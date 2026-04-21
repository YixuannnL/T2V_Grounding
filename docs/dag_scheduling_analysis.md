# Quality-Aware Shot Scheduling via Agentic Planning

## 1. 核心洞察：为什么线性执行顺序不是最优？

### 1.1 问题本质

在 multi-shot video generation 中，存在一个关键矛盾：

> **叙事顺序 ≠ 最优生成顺序**

传统方法按 shot 1 → shot 2 → ... → shot N 线性执行，隐含假设是：
- 实体首次出现时能获得足够好的参考
- Reference 质量在 shot 间均匀分布

但现实中，**同一实体在不同 shot 的 grounding 质量差异巨大**：

```
Shot 1 (wide shot):     实体占画面 5%  → 低分辨率参考
Shot 2 (medium shot):   实体占画面 20% → 中等质量参考
Shot 3 (close-up):      实体占画面 60% → 高质量参考
```

如果按线性顺序，Shot 1 的低质量参考会"污染"整个生成链。

---

## 2. Case Studies: 何时 DAG Scheduling 有显著优势？

### Case 1: Close-up 后置场景（最典型）

**Script Example:**
```
Shot 1: Wide shot - Sarah walks through crowded marketplace
Shot 2: Medium shot - Sarah stops at a fruit stall
Shot 3: Close-up - Sarah's face as she examines an apple
Shot 4: Wide shot - Sarah continues walking, apple in hand
```

**线性执行的问题：**
```
Shot 1 Grounding: Sarah 在人群中，仅占画面 3%
                  → 参考图分辨率: ~50x100 pixels
                  → 人脸细节几乎不可见

Shot 2, 3, 4 使用这个低质量参考
→ 结果：人物一致性逐渐漂移
```

**DAG 优化执行：**
```
Step 1: 分析 script，识别 Shot 3 是 Sarah 的 "reference source"
Step 2: 先执行 Shot 3 (close-up)
        → 获得高质量人脸参考（~400x400 pixels）
Step 3: 用 Shot 3 的参考作为 anchor，执行 Shot 1, 2, 4
→ 结果：所有 shot 的人物一致性显著提升
```

**量化收益预估：**
| 指标 | 线性执行 | DAG 执行 | 提升 |
|------|----------|----------|------|
| Anchor 分辨率 | 50x100 | 400x400 | 64x |
| CLIP-I 一致性 | ~0.65 | ~0.85 | +30% |
| Retry 次数 | 3-5 | 0-1 | -70% |

---

### Case 2: 渐进式揭示（Progressive Reveal）

**Script Example:**
```
Shot 1: A mysterious figure in shadows, face obscured by hood
Shot 2: The figure steps into dim streetlight, silhouette visible
Shot 3: The figure lowers hood, revealing it's Detective Chen
Shot 4: Chen walks toward camera, determined expression
```

**核心挑战：** 同一实体在前几个 shot 被刻意遮挡/隐藏

**线性执行的问题：**
```
Shot 1: Grounding 失败 - 人脸被兜帽遮挡
Shot 2: Grounding 部分成功 - 仅侧影
→ Shot 3, 4 没有好的参考可用
→ Chen 的外观在 shot 间不一致
```

**DAG 优化：**
```
依赖分析：
  Shot 3 → provides → anchor for "Chen"
  Shot 4 → depends on → Shot 3 (best reference)
  Shot 1, 2 → can use → Shot 3's reference (even for obscured version)

执行顺序：3 → 4 → 1 → 2
```

**关键洞察：** 即使叙事上 Shot 1 的人物是"神秘的"，但生成时仍需要 reference 保证是同一个人。先生成 Shot 3 获得 Chen 的正脸，再用它约束 Shot 1, 2 中的"神秘人物"，确保揭示时身份一致。

---

### Case 3: 多实体交叉出场

**Script Example:**
```
Shot 1: Alex and Bob meet at the cafe (wide, two-shot)
Shot 2: Close-up of Bob explaining his plan
Shot 3: Alex's reaction shot (close-up)
Shot 4: Both stand up to leave (medium shot)
```

**实体-Shot 矩阵分析：**
```
           Shot 1    Shot 2    Shot 3    Shot 4
Alex       medium    absent    HIGH      medium
Bob        medium    HIGH      absent    medium
```

**最优执行顺序：**
```
1. Shot 2 → 获得 Bob 高质量参考
2. Shot 3 → 获得 Alex 高质量参考
3. Shot 1 → 用两个高质量参考生成双人镜头
4. Shot 4 → 用两个高质量参考生成双人镜头

DAG 结构：
    Shot 2 ──┐
             ├──→ Shot 1 ──→ Shot 4
    Shot 3 ──┘
```

**为什么不先生成 Shot 1？**
- Shot 1 是 wide two-shot，两人都较小
- 如果先生成 Shot 1，两人的参考质量都是 medium
- Close-up shots (2, 3) 生成时会与 Shot 1 的低质量参考不一致

---

### Case 4: 场景建立依赖（Environment Bootstrapping）

**Script Example:**
```
Shot 1: Establishing shot - exterior of Gothic cathedral at dusk
Shot 2: Interior - camera moves through empty nave
Shot 3: Close-up of stained glass window, light streaming through
Shot 4: Wide interior shot with protagonist entering
```

**场景一致性挑战：**
- Cathedral 的建筑风格、光线色调需要跨 shot 一致
- Shot 1 (exterior) vs Shot 2, 3, 4 (interior) 风格需要协调

**DAG 分析：**
```
Shot 2 (interior wide) → provides → location anchor
Shot 3 (detail) → depends on → Shot 2's color palette
Shot 4 (interior + character) → depends on → Shot 2's architecture

执行：2 → 3 → 1 → 4
```

**关键决策：** 先生成 interior wide shot 建立场景基调，再用它约束 detail shot 和 establishing shot。

---

### Case 5: 动作序列的端点约束

**Script Example:**
```
Shot 1: Character crouches, preparing to jump
Shot 2: Character mid-air, arms spread
Shot 3: Character lands on rooftop, rolls forward
```

**常规方法：** 线性生成，每个 shot 独立

**DAG 方法（端点约束策略）：**
```
1. 先生成 Shot 1 (起始姿态) 和 Shot 3 (结束姿态)
   → 这两个 shot 姿态相对"稳定"，更容易生成一致
2. 再生成 Shot 2，用 Shot 1, 3 作为时序约束
   → Mid-air shot 有明确的 "从哪来、到哪去" 约束
```

**数学直觉：** 类似于曲线拟合，先确定端点，再插值中间点。

---

## 3. 形式化：Shot Scheduling 优化问题

### 3.1 符号定义

- $S = \{s_1, s_2, ..., s_n\}$: Shot 集合
- $E = \{e_1, e_2, ..., e_m\}$: 实体集合
- $Q(s_i, e_j) \in [0, 1]$: Shot $s_i$ 对实体 $e_j$ 的预测 grounding 质量
- $A(e_j)$: 实体 $e_j$ 最终使用的 anchor 参考质量

### 3.2 优化目标

**目标函数：最大化所有实体的 anchor 质量**

$$\max_{\pi} \sum_{e_j \in E} A_\pi(e_j)$$

其中 $\pi$ 是 shot 执行顺序的排列。

**约束条件：**
1. 叙事依赖（如果 shot B 的内容直接引用 shot A 的结果）
2. 时序一致性（某些 action sequence 需要保持顺序）

### 3.3 Anchor 质量计算

对于实体 $e_j$，其 anchor 质量取决于执行顺序中**第一个包含该实体的高质量 shot**：

$$A_\pi(e_j) = Q(s_{\text{first}}, e_j)$$

其中 $s_{\text{first}} = \arg\min_{s_i \in \pi} \{ \text{rank}_\pi(s_i) : e_j \in s_i \land Q(s_i, e_j) > \tau \}$

### 3.4 LLM 作为启发式求解器

精确求解是 NP-hard（类似于带约束的排列优化）。我们使用 LLM 作为启发式求解器：

```python
SCHEDULING_PROMPT = """
分析以下 script，确定最优 shot 执行顺序。

## 输入
- Shots: {shots_with_descriptions}
- Entities: {entities_list}

## 分析维度
1. **Grounding Quality Prediction**: 预测每个 shot 对每个实体的 grounding 质量
   - Close-up > Medium shot > Wide shot
   - 良好光线 > 暗光/逆光
   - 无遮挡 > 部分遮挡 > 完全遮挡

2. **Reference Source Identification**: 识别每个实体的"最佳参考来源" shot

3. **Dependency Graph**: 构建执行依赖图

## 输出
{
  "quality_matrix": {
    "shot_1": {"entity_a": 0.3, "entity_b": 0.8},
    "shot_2": {"entity_a": 0.9, "entity_b": 0.2},
    ...
  },
  "reference_sources": {
    "entity_a": "shot_2",
    "entity_b": "shot_1",
    ...
  },
  "execution_order": ["shot_2", "shot_1", "shot_3", ...],
  "reasoning": "..."
}
"""
```

---

## 4. 与 Existing Work 的对比

| 方法 | 执行顺序 | Reference 选择 | 一致性策略 |
|------|----------|----------------|------------|
| **VideoDirector** | 线性 | 首次出现 | 全局 embedding |
| **MovieGen** | 线性 | 用户提供 | Multi-reference fusion |
| **StoryDiffusion** | 线性 | 首次出现 | Cross-attention |
| **Ours** | **DAG (动态)** | **最优出现 (LLM 决策)** | Grounding-in-the-loop |

**核心差异化：**

> 现有方法：Reference quality is a **given constraint** to work around
> Our approach: Reference quality is an **optimization variable** to maximize

---

## 5. 实验设计建议

### 5.1 Ablation Study: Linear vs DAG

**Setup:**
- 相同的 script 集合
- 相同的 generation model
- 变量：执行顺序（Linear vs DAG-optimized）

**Metrics:**
- Per-entity CLIP-I consistency
- FID between shots
- Human evaluation: "Is this the same person?"
- Retry rate per shot

### 5.2 Case-Stratified Evaluation

将 test scripts 分类：

| Script Type | 预期 DAG 优势 | 示例 |
|-------------|---------------|------|
| Close-up Heavy | 高 | 人物访谈、情感戏 |
| Wide Shot Heavy | 低 | 风景、建筑纪录片 |
| Progressive Reveal | 高 | 悬疑、惊悚片 |
| Multi-character | 中-高 | 对话场景、群戏 |
| Action Sequence | 中 | 动作片、追逐戏 |

**预期结果：** DAG scheduling 在 Close-up Heavy 和 Progressive Reveal 类型上显著优于 Linear。

### 5.3 Scheduling Quality Correlation

验证 LLM 预测的 grounding quality 与实际质量的相关性：

1. LLM 预测 quality matrix
2. 实际执行 grounding，获得真实 quality
3. 计算 Spearman correlation

---

## 6. 论文写作建议

### Contribution 表述

**Contribution 2: Quality-Aware Agentic Shot Scheduling**

> We observe that the quality of visual grounding varies dramatically across shots due to camera distance, lighting, and occlusion. This creates a critical dependency: shots with high-quality grounding should be generated first to establish reliable reference anchors. We propose an agentic planning approach where an LLM analyzes the script to: (1) predict grounding quality for each entity-shot pair, (2) identify optimal reference sources, and (3) construct a dependency-aware execution DAG. This enables "reference bootstrapping"—generating close-up shots first to establish strong anchors, even when they appear later in narrative order.

### Figure 建议

**Figure 3: DAG Scheduling Example**

```
(a) Script with 4 shots
    Shot 1: Wide - Alex walks into cafe
    Shot 2: Medium - Alex orders coffee
    Shot 3: Close-up - Alex's face
    Shot 4: Wide - Alex leaves cafe

(b) Grounding Quality Matrix (LLM predicted)
    [Heat map showing quality scores]

(c) Linear Execution
    1 → 2 → 3 → 4
    Anchor quality: 0.3 (from shot 1)

(d) DAG Execution
    3 → 1 → 2 → 4
    Anchor quality: 0.9 (from shot 3)

(e) Consistency Comparison
    [Side-by-side frames showing drift in linear vs stability in DAG]
```

---

## 7. 潜在 Limitations & Future Work

### Limitations
1. **LLM 预测不准确**：Grounding quality 预测是启发式的
2. **叙事约束冲突**：某些 shot 必须按顺序生成（如动作连续性）
3. **Computational Overhead**：DAG 分析增加 planning 时间

### Future Work
1. **Learning to Schedule**：用 RL 学习 scheduling policy
2. **Iterative Refinement**：生成后根据实际质量动态调整后续 schedule
3. **Parallelization**：DAG 中独立子图可并行执行

---

## 8. 代码实现草案

```python
class AgenticScheduler:
    """
    Quality-Aware Shot Scheduler

    使用 LLM 分析 script，构建最优执行 DAG
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    def analyze_script(self, shots: List[Shot], entities: List[Entity]) -> ScheduleResult:
        """
        分析 script，返回优化后的执行顺序

        Returns:
            ScheduleResult:
                - quality_matrix: Dict[shot_id, Dict[entity_id, float]]
                - reference_sources: Dict[entity_id, shot_id]
                - execution_order: List[shot_id]
                - dependency_graph: DAG structure
        """
        prompt = self._build_analysis_prompt(shots, entities)
        response = self.llm.chat(prompt, system=SCHEDULING_SYSTEM_PROMPT)
        return self._parse_schedule_response(response)

    def _predict_grounding_quality(self, shot: Shot, entity: Entity) -> float:
        """
        预测单个 shot-entity pair 的 grounding 质量

        Heuristics:
        - shot_type: close-up (0.9) > medium (0.6) > wide (0.3)
        - lighting: good (1.0x) > dim (0.7x) > backlit (0.5x)
        - occlusion: none (1.0x) > partial (0.6x) > full (0.1x)
        """
        base_score = {"close-up": 0.9, "medium": 0.6, "wide": 0.3}.get(shot.type, 0.5)
        # ... 更多启发式调整
        return base_score

    def build_dag(self, quality_matrix, entities) -> nx.DiGraph:
        """
        基于 quality matrix 构建执行 DAG

        Rules:
        1. 每个实体的 reference source shot 没有入边（最先执行）
        2. 其他包含该实体的 shot 依赖 reference source
        3. 合并多实体依赖，形成最终 DAG
        """
        dag = nx.DiGraph()

        for entity in entities:
            # 找到该实体的最佳 reference source
            best_shot = max(quality_matrix.keys(),
                           key=lambda s: quality_matrix[s].get(entity.id, 0))

            # 其他包含该实体的 shot 依赖 best_shot
            for shot_id, entity_scores in quality_matrix.items():
                if entity.id in entity_scores and shot_id != best_shot:
                    dag.add_edge(best_shot, shot_id)

        return dag

    def get_execution_order(self, dag) -> List[str]:
        """拓扑排序获得执行顺序"""
        return list(nx.topological_sort(dag))
```

---

## 9. 总结

**DAG Scheduling 的核心价值：**

1. **打破"首次出现即锚定"的假设**
   - 允许用"最佳出现"而非"首次出现"作为 anchor

2. **将 Reference Quality 从约束变为优化变量**
   - 不是被动接受参考质量，而是主动选择最佳参考来源

3. **体现 Agentic 特征**
   - LLM 主动分析、规划、决策
   - 不是执行固定 pipeline，而是动态适应每个 script

4. **可量化、可验证**
   - Ablation 实验设计清晰
   - 预期在特定场景（close-up heavy, progressive reveal）有显著收益
