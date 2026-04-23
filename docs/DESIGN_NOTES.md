# T2V Grounding 设计细节与边界情况处理

本文档记录项目中发现的各种边界情况（edge cases）、问题根因分析、以及对应的设计决策。这些细节是在实际运行中逐步发现和修复的，对于理解系统行为和后续维护非常重要。

---

## 目录

1. [Character-Aware Mode Routing (v3.1)](#1-character-aware-mode-routing-v31)
2. [Frontal-Aware Character Reference (v2.3)](#2-frontal-aware-character-reference-v23)
3. [Body Part Closeup Detection (v2.4)](#3-body-part-closeup-detection-v24)
4. [Earliest High-Quality Anchor Strategy](#4-earliest-high-quality-anchor-strategy)
5. [Location Anchor Strategy (v2.1)](#5-location-anchor-strategy-v21)
6. [Agentic Light-Aware Close-up Strategy](#6-agentic-light-aware-close-up-strategy)
7. [Shot 1 Generation-Verification Loop (v2.5)](#7-shot-1-generation-verification-loop-v25)
8. [Cross-Entity IoU Deduplication](#8-cross-entity-iou-deduplication)
9. [Agentic Reference Selection (v3.0)](#9-agentic-reference-selection-v30)
10. [Environment Context Injection](#10-environment-context-injection)
11. [Smart Self-Improving Registry (v3.2)](#11-smart-self-improving-registry-v32)
12. [Self-Critique & Reflection Loop (v4.0)](#12-self-critique--reflection-loop-v40)
13. [Agentic DAG Scheduling (v2.5)](#13-agentic-dag-scheduling-v25)
14. [Root Cause Analysis Retry (v4.1)](#14-root-cause-analysis-retry-v41)
15. [Experience Memory System (v4.2)](#15-experience-memory-system-v42)

---

## 1. Character-Aware Mode Routing (v3.1)

### 问题发现

**场景**：夜骑镜头，Shot 1 是远景，人物只有背影剪影。

**现象**：Shot 2 生成了动画风格的人物，与 Shot 1 的真实风格完全不一致。

**数据证据**：
```
Shot 1 char_001 id_confidence: 0.0 (全是背影，无人脸)
Shot 2 reference_used: {"obj_horse": [...]}  ← 只有马，没有人
Shot 2 generation_mode: phantom  ← 走了 S2V 模式！
```

### 根因分析

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. Shot 1 人物全是背影 → id_confidence = 0.0                        │
│  2. Shot 2 路由时，character 被标记为 faceless，跳过                  │
│  3. 但 obj_horse 有参考图 → 旧逻辑认为 "有 subject ref"               │
│  4. 路由到 S2V 模式，但只有马的参考图                                 │
│  5. Phantom 没有人脸锚定 → 脑补出动画风格人物                         │
└─────────────────────────────────────────────────────────────────────┘
```

### 问题本质

**旧逻辑**：
```python
has_subject_ref = any(character OR object 有参考图)
if has_subject_ref:
    mode = "phantom"  # S2V
```

**问题**：object 参考图（马、车等）不能锚定人物外观！当场景中有人物但只有 object 参考时，S2V 仍会导致人物风格漂移。

### 修复方案

**新逻辑 (v3.1)**：
```python
if has_frontal_character_ref:
    # 有人脸锚定 → S2V 安全
    mode = "phantom"
elif character_entities_in_shot and not has_frontal_character_ref:
    # 场景有人物但无人脸锚定 → 必须 T2V
    # 即使有 object 参考也不行！
    mode = "t2v"
elif has_object_ref:
    # 场景无人物，有物体参考 → S2V 可以
    mode = "phantom"
else:
    mode = "t2v"
```

### 决策流程图

```
场景有 character 实体？
       │
       ├── YES ──► 有 frontal character 参考？
       │             │
       │             ├── YES ──► S2V ✅
       │             │
       │             └── NO ──► T2V ✅ (即使有 object ref)
       │
       └── NO ──► 有 object 参考？
                   │
                   ├── YES ──► S2V ✅
                   └── NO ──► T2V
```

### 修复效果

| 场景 | 修复前 | 修复后 |
|------|--------|--------|
| 人物背影 + 马 | S2V → 动画化 ❌ | T2V → 真实风格 ✅ |
| 人物正脸 + 马 | S2V ✅ | S2V ✅ |
| 只有车辆（无人物）| S2V ✅ | S2V ✅ |

---

## 2. Frontal-Aware Character Reference (v2.3)

### 问题发现

**场景**：某些镜头只捕捉到人物背影或完全侧面。

**现象**：Phantom 使用背影参考图时，会"脑补"出一个完全不同的脸，甚至变成动画风格。

### 根因分析

InsightFace 的 `id_confidence` 反映人脸检测置信度：
- 正脸：0.8-1.0
- 侧脸：0.3-0.6
- 背影/无脸：0.0

当 `id_confidence < 0.3` 时，说明参考图中几乎没有可用的面部信息。

### 修复方案

```python
min_face_confidence = 0.3  # 阈值

for entity in character_entities:
    anchor = registry.query_anchor(entity.entity_id, ...)
    if anchor and anchor.id_confidence < min_face_confidence:
        # 无脸参考：不传参考图，改用 Appearance Context
        faceless_characters.append(entity)
    else:
        # 有脸参考：正常使用
        reference_used[entity.entity_id] = [anchor.crop_path]
```

### Appearance Context 注入

对于无脸 character，通过 prompt 描述外观：

```
[Appearance Context - No Frontal Reference Available]
Character: blonde woman (hair_color: blonde, clothing: white coat).
Note: Maintain visual consistency with the described appearance.
```

---

## 3. Body Part Closeup Detection (v2.4)

### 问题发现

**场景**：镜头描述 "A close-up focuses on an aged hand with wrinkled skin..."

**现象**：LLM 正确关联到 `char_elderly_man`，Pipeline 传入人脸参考图，但生成的手部特写出现不合理的人脸特征。

### 根因分析

Phantom 看到人脸参考图会强行注入人脸特征，即使镜头只需要手部。

### 修复方案

检测身体部位关键词，对此类镜头跳过 character 人脸参考：

```python
body_part_keywords = [
    "hand", "hands", "finger", "fingers", "palm",
    "foot", "feet", "leg", "legs",
    "arm", "arms", "shoulder", "back", "torso",
]

is_body_part_closeup = is_closeup and any(kw in shot_text_lower for kw in body_part_keywords)

if is_body_part_closeup:
    # 跳过所有 character 人脸参考图
    faceless_characters.append(entity)
```

### 示例

```
Shot: "A close-up focuses on an aged hand with wrinkled skin..."

旧行为: LLM 关联 char_elderly_man → 传人脸参考 → 手部出现不合理结果 ❌
新行为: 检测到 "hand" + "close-up" → 跳过人脸 → Appearance Context ✅
```

---

## 4. Earliest High-Quality Anchor Strategy

### 问题发现

**场景**：Shot 1 是 wide shot（远景），人物只有侧脸；Shot 2 是 close-up，有清晰正脸。

**现象**：单纯选"最早 shot"会选到质量差的侧脸，导致后续所有镜头都基于侧脸生成。

### 数据示例

```
char_elderly_man 的参考图：
  - Shot 1: 0.72 (侧脸，wide shot)
  - Shot 2: 0.93 (正脸，close-up)
  - Shot 4: 0.88 (正脸)
```

### 修复方案

**"earliest_high_quality" 策略**：

```python
high_quality_threshold = 0.85

def query_anchor(entity_id, min_quality, high_quality_threshold):
    # 1. 如果最早 shot 有高质量参考 → 选它
    earliest = get_earliest_ref(entity_id)
    if earliest.quality >= high_quality_threshold:
        return earliest

    # 2. 否则找所有 shot 中最早的高质量参考
    high_quality_refs = get_refs_with_quality_above(entity_id, high_quality_threshold)
    if high_quality_refs:
        return min(high_quality_refs, key=lambda r: r.shot_id)

    # 3. 都没有高质量的 → 回退最早 shot
    return earliest
```

### 效果

```
旧策略 (earliest_good): 选 Shot 1 的 0.72 (侧脸) ❌
新策略 (earliest_high_quality): 选 Shot 2 的 0.93 (正脸) ✅
  → 因为 0.72 < 0.85，而 Shot 2 有 ≥ 0.85 的高质量正脸
```

---

## 5. Location Anchor Strategy (v2.1)

### 问题发现

**场景**：Shot 1 是 close-up（近景），背景因景深效果而模糊；Shot 3 是 wide shot，背景清晰。

**现象**：单纯选最早 shot 会锁定模糊背景，后续镜头场景一致性差。

### 数据示例

```
loc_opulent_room 的背景参考图：
  - Shot 1: 0.34 (close-up 导致模糊)
  - Shot 2: 0.35
  - Shot 3: 0.95 (wide shot，背景清晰)
```

### 修复方案

**"earliest_unless_much_worse" 策略**：

```python
high_quality_threshold = 0.7
quality_gap_ratio = 1.5

def query_anchor_location(entity_id, ...):
    earliest = get_earliest_ref(entity_id)

    # 1. 如果最早 shot 背景质量 >= 0.7 → 选它
    if earliest.quality >= high_quality_threshold:
        return earliest

    # 2. 找最高质量背景，如果比最早好 1.5 倍以上 → 切换
    best = get_best_quality_ref(entity_id)
    if best.quality / earliest.quality >= quality_gap_ratio:
        return best

    # 3. 否则默认最早 shot（防止场景漂移）
    return earliest
```

### 效果

```
旧策略: 选 Shot 1 的 0.34 (模糊) ❌
新策略: 选 Shot 3 的 0.95 ✅
  → 因为 0.95 / 0.34 = 2.8x > 1.5x 阈值
```

---

## 6. Agentic Light-Aware Close-up Strategy

### 问题发现

**场景**：办公室场景，有金色暖光。近景镜头如果不传 location 参考，光线色调可能与整体不一致。

**旧方案**：close-up 一律不传 location 参考图。

**问题**：简单光线可以用文字描述，但复杂光线（多色光源、金色反射）需要图像参考。

### 修复方案

引入 **Agentic 决策**：LLM 分析光线复杂度，决定是否传 location。

```python
if is_closeup and has_location_ref:
    analysis = parser.analyze_closeup_lighting(location_entities)

    if analysis.complexity_score >= 3:
        # 复杂光线 → 传 location 参考图
        include_location = True
    else:
        # 简单光线 → 不传，注入 Lighting Context
        include_location = False
        prompt_parts.append(build_lighting_context(analysis))
```

### 光线复杂度等级

| Score | 描述 | 策略 |
|-------|------|------|
| 1-2 | 均匀日光、单一室内灯 | 不传 location，文字描述 |
| 3-5 | 多色霓虹、金色反射、逆光 | 传 location，保持色调 |

### LLM 分析输出示例

```json
{
  "lighting_description": "warm ambient lighting with golden tones from large windows",
  "color_tone": "warm golden and amber tones",
  "complexity_score": 3,
  "needs_location_ref": true,
  "reason": "Gold furniture creates specific warm reflections affecting skin tones"
}
```

---

## 7. Shot 1 Generation-Verification Loop (v2.5)

### 问题发现

**场景**：Shot 1 prompt 描述"三个人"，但 T2V 模型生成了四个人。

**问题**：Shot 1 是锚点，后续所有 shot 都基于它的 grounding 结果。人数错误会传播到整个视频。

### 修复方案

引入 **Generation-Verification Loop**：

```
1. LLM 解析 prompt → 提取预期人数
2. T2V 生成视频
3. MLLM 验证人数（采样 3 帧，取众数）
4. 不匹配 → 重试
   - Attempt 1-2: 换 seed
   - Attempt 3+: 增强 prompt "[exactly N people]"
5. 最多 3 次重试
```

### 为什么用 MLLM 而不是检测模型

| 维度 | Grounding DINO | MLLM (Claude Haiku) |
|------|----------------|---------------------|
| 小目标 | 容易漏检 | 更准确 |
| 遮挡 | NMS 误合并 | 语义理解 |
| 非真人 | 无法区分 | 能区分雕像/画像/倒影 |

---

## 8. Cross-Entity IoU Deduplication

### 问题发现

**场景**：一帧中有多人，Grounding DINO 可能把 "young boy" 的检测框错误定位到 "elderly man" 上。

### 根因

两者都是 person 类别，当人物靠近时检测框可能高度重叠。

### 修复方案

对同一帧中所有实体的检测结果做 IoU 去重：

```python
def cross_entity_dedup(all_results, iou_threshold=0.5):
    # 按帧分组
    # 对每帧：按置信度排序
    # 高置信度的抑制重叠的低置信度检测
    for frame in frames:
        detections = sort_by_confidence(frame_detections)
        for det in detections:
            # 抑制与当前检测 IoU > 0.5 的其他检测
            suppress_overlapping(det, iou_threshold)
```

### 日志示例

```
[Pipeline] IoU去重: char_young_boy 的检测被 char_elderly_man 抑制 (IoU=0.78)
[Pipeline] 跨实体IoU去重: 共移除 3 个重叠检测
```

---

## 9. Agentic Reference Selection (v3.0)

### 问题发现

传统 InsightFace 打分只适用于人脸，无法评估：
- Object（车辆、道具）
- Location（场景）
- 上下文需求（对话戏 vs 动作戏）

### 修复方案

引入 **ReferenceSelectionAgent**，使用 VLM 智能选择：

```python
agent = ReferenceSelectionAgent(model="claude-sonnet-4-6")

result = agent.select_best_reference(
    entity_id="char_alex",
    entity_type="character",
    entity_description="A young man with short brown hair",
    candidates=candidate_images,
    shot_context="A close-up dialogue scene",
    shot_type="close-up",
)
# result.selected_index, result.confidence, result.reason
```

### 三种模式

| 模式 | 行为 |
|------|------|
| `traditional` | 传统 InsightFace 打分 |
| `agent` | 纯 VLM 选择 |
| `hybrid` (默认) | VLM 优先 + 传统 fallback |

### 优势

1. **通用性**：人物、物体、场景都能评估
2. **上下文感知**：根据镜头需求选择
3. **可解释性**：输出选择理由

---

## 10. Environment Context Injection

### 问题发现

当回退 T2V 模式（无 subject 参考）时，如何保持场景一致性？

### 修复方案

从 location 实体提取环境属性，注入 prompt：

```python
if generation_mode == "t2v" and location_entities:
    env_parts = []
    for loc in location_entities:
        desc = loc.text_description
        attrs = ", ".join(f"{k}: {v}" for k, v in loc.attributes.items())
        env_parts.append(f"Scene: {desc} ({attrs}).")

    prompt_parts.append("[Environment Context]\n" + "\n".join(env_parts))
```

### 输出示例

```
[Environment Context]
Scene: Interrogation room (lighting: fluorescent, harsh shadows, atmosphere: tense).
```

---

## 11. Smart Self-Improving Registry (v3.2)

### 问题发现

**场景**：处理一个 20 shot 的视频，包含 5 个 character + 3 个 location。

**现象**：Registry 数据库膨胀到 480+ 条记录，大量低质量、冗余的参考图堆积。

**数据证据**：
```
char_alex 参考图：
  - Shot 1: 3 张 (score: 0.45, 0.52, 0.71)
  - Shot 2: 3 张 (score: 0.48, 0.55, 0.68)
  - ...
  - Shot 20: 3 张 (score: 0.41, 0.49, 0.63)

总计 60 张，其中 40+ 张质量相似、角度相似，完全冗余
```

### 根因分析

```
┌─────────────────────────────────────────────────────────────────────┐
│  旧 Registry 设计缺陷：                                              │
│                                                                     │
│  1. 只注册不淘汰 —— 所有参考图永久保留                                │
│  2. 无相似度检测 —— 相同角度的参考图重复注册                          │
│  3. 无容量管理 —— 无上限，无优先级                                   │
│  4. 查询时才过滤 —— 低质量数据仍占用存储和索引                        │
│                                                                     │
│  后果：                                                             │
│  - 数据库膨胀                                                       │
│  - 查询变慢                                                         │
│  - 误选低质量参考的风险增加                                          │
└─────────────────────────────────────────────────────────────────────┘
```

### 修复方案：SmartEntityRegistry

引入 **Agentic 自优化参考图库**，核心改进：

#### 1. 智能注册（注册时过滤）

```python
def register(entity_id, entry) -> (success, reason):
    # Check 1: 质量门槛
    if entry.quality_score < min_quality_to_register:
        return False, "低于质量阈值"

    # Check 2: 人脸置信度（character）
    if entry.id_confidence < min_id_confidence:
        return False, "人脸置信度不足"

    # Check 3: 相似度去重（CLIP embedding）
    if cosine_similarity(new, existing) > 0.92:
        return False, "与已有参考图过于相似"

    # Check 4: 同 shot 容量限制
    if shot_count >= max_refs_per_shot:
        if new.score > worst_in_shot.score:
            evict(worst_in_shot)  # 替换
        else:
            return False, "同 shot 已满"

    # Check 5: 总容量限制
    if total_count >= max_refs_per_entity:
        evict_lowest_quality()  # 淘汰最差的
```

#### 2. 主动淘汰机制

```python
class EvictionReason(Enum):
    LOW_QUALITY = "low_quality"      # 质量分过低
    REDUNDANT = "redundant"          # 与其他参考图过于相似
    SUPERSEDED = "superseded"        # 被更高质量的替代

def run_eviction_audit():
    # 1. 淘汰低质量（保留锚点）
    evict_below_threshold(quality < 0.5, protected=anchors)

    # 2. 淘汰冗余（CLIP 相似度 > 0.92）
    evict_redundant_by_similarity()
```

#### 3. 锚点保护机制

```python
# 自动提升为锚点的条件：
is_anchor = (
    quality_score >= 0.85 and      # 高质量
    id_confidence >= 0.7 and       # 正脸
    shot_id <= 3                   # 早期出现
)

# 锚点特权：
# - 不会被自动淘汰
# - 查询时优先返回
# - 每个实体保护 2 个锚点
```

#### 4. 多样性管理

```python
# 姿态标签
pose_tag: "frontal" | "three_quarter" | "profile" | "back"

# 光线标签
lighting_tag: "natural" | "bright" | "dim" | "backlit"

# 确保保留不同姿态的参考图（用于不同镜头需求）
min_pose_diversity = 2
```

#### 5. VLM 审计 Agent

```python
class ReferenceAuditAgent:
    """
    VLM 驱动的参考图质量审计
    - 分析姿态、光线、清晰度
    - 检测跨实体的标注错误
    - 建议淘汰/保留
    """
```

### 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `min_quality_to_register` | 0.4 | 注册质量门槛 |
| `max_refs_per_entity` | 10 | 每实体最大参考数 |
| `max_refs_per_shot` | 2 | 每 shot 每实体最大数 |
| `similarity_threshold` | 0.92 | CLIP 相似度去重阈值 |
| `protected_anchor_count` | 2 | 每实体保护锚点数 |

### 效果对比

| 指标 | 旧 Registry | SmartRegistry |
|------|-------------|---------------|
| 20 shot 后参考图数量 | 480+ | ≤80 (8 实体 × 10) |
| 冗余参考图 | 大量 | 自动去重 |
| 查询性能 | 随时间下降 | 稳定 |
| 锚点稳定性 | 无保护 | 自动保护 |
| 审计能力 | 无 | VLM 驱动 |

---

## 12. Self-Critique & Reflection Loop (v4.0)

### 问题发现

**场景**：Shot 1 prompt 描述 "blonde woman with blue eyes"，T2V 生成了 "brown-haired woman with dark eyes"。

**现象**：传统 Generation-Verification Loop (v2.5) 只验证人数，无法检测细粒度的属性偏差（发色、服装、场景氛围等）。

**数据证据**：
```
Shot 1 prompt: "A blonde woman in a white coat walks through a golden-lit opulent room"
Generated video:
  - ✅ 人数正确 (1 人)
  - ❌ 发色错误 (brown vs blonde)
  - ❌ 服装不符 (black coat vs white coat)
  - ❌ 场景偏差 (modern office vs opulent room)
```

### 根因分析

```
┌─────────────────────────────────────────────────────────────────────┐
│  v2.5 验证局限：                                                     │
│                                                                     │
│  1. 只检查人数 —— 忽略外观属性（发色、服装、表情）                      │
│  2. 无法评估场景一致性 —— 光线、氛围、背景                            │
│  3. 无法关联 Shot 间漂移 —— 同一角色跨 Shot 外观变化                   │
│  4. 输出是 pass/fail —— 无法指导修复方向                             │
│                                                                     │
│  后果：                                                             │
│  - 细节偏差无法检测                                                  │
│  - 错误参考图被锚定，传播到后续所有 shot                               │
│  - 修复只能靠换 seed 碰运气                                          │
└─────────────────────────────────────────────────────────────────────┘
```

### 修复方案：Self-Critique & Reflection Loop

引入 **VLM 驱动的视频质量分析器**，不仅检测问题，还输出针对性修复建议。

#### 1. 架构设计

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Self-Critique 反馈循环                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Shot Prompt + Reference Images                                    │
│        │                                                            │
│        ▼                                                            │
│   ┌──────────┐     ┌──────────────┐     ┌─────────────────┐        │
│   │ T2V/S2V  │ ──▶ │ VideoQuality │ ──▶ │ RepairStrategy  │        │
│   │ Generate │     │   Critic     │     │   Generator     │        │
│   └──────────┘     └──────────────┘     └─────────────────┘        │
│        ▲                  │                     │                   │
│        │                  ▼                     ▼                   │
│        │           ┌──────────────┐     ┌─────────────────┐        │
│        └────────── │ Pass? score  │ NO  │ Adjusted Params │        │
│                    │   >= 0.7     │ ──▶ │ (prompt, seed,  │        │
│                    └──────────────┘     │  guidance, etc) │        │
│                          │              └─────────────────┘        │
│                         YES                                        │
│                          ▼                                         │
│                    ┌──────────────┐                                │
│                    │ Accept Video │                                │
│                    └──────────────┘                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 2. VideoQualityCritic 核心类

```python
class VideoQualityCritic:
    """
    VLM 驱动的视频质量评估器
    - 多帧采样分析
    - 细粒度问题检测
    - 可操作的修复建议
    """
    def critique(
        video_path: str,
        shot_prompt: str,
        reference_images: Dict[str, str],  # entity_id -> image_path
        sample_frames: int = 5,
    ) -> CritiqueResult
```

#### 3. 问题类型分类 (IssueType)

| 类型 | 描述 | 示例 |
|------|------|------|
| `identity_mismatch` | 整体身份不匹配 | 完全不同的人 |
| `facial_feature_mismatch` | 面部特征偏差 | 发色、眼色、脸型 |
| `clothing_mismatch` | 服装不一致 | 白大衣变黑夹克 |
| `pose_mismatch` | 姿态不符预期 | 站立变坐着 |
| `expression_mismatch` | 表情偏差 | 微笑变严肃 |
| `scene_mismatch` | 场景/背景不符 | 办公室变街道 |
| `lighting_mismatch` | 光线色调偏差 | 金色暖光变冷白光 |
| `style_drift` | 风格漂移 | 真实变动画 |
| `motion_issue` | 动作/运动问题 | 不自然的运动 |
| `quality_issue` | 画质问题 | 模糊、伪影 |

#### 4. 严重程度分级 (IssueSeverity)

```python
class IssueSeverity(Enum):
    CRITICAL = "critical"  # 必须修复（identity 错误）
    HIGH = "high"          # 强烈建议修复
    MEDIUM = "medium"      # 建议修复
    LOW = "low"            # 可接受
```

#### 5. CritiqueResult 输出结构

```python
@dataclass
class CritiqueResult:
    passed: bool              # overall_score >= threshold
    overall_score: float      # 0.0 - 1.0
    issues: List[CritiqueIssue]
    frame_analyses: List[FrameAnalysis]
    repair_suggestions: List[RepairSuggestion]
    summary: str              # 人类可读的总结
```

#### 6. RepairSuggestion 修复建议

```python
@dataclass
class RepairSuggestion:
    issue_type: IssueType
    strategy: str             # "enhance_prompt" | "change_seed" | ...
    priority: int             # 1 = 最高
    prompt_addition: str      # 增强 prompt 的文本
    parameter_adjustments: Dict[str, Any]  # seed, guidance_scale 等
    rationale: str
```

#### 7. RepairStrategyGenerator

将 VLM 的建议转换为具体的生成参数：

```python
class RepairStrategyGenerator:
    def generate_repair_params(
        original_params: Dict,
        critique_result: CritiqueResult,
    ) -> Dict:
        # 根据问题类型选择策略
        for suggestion in critique_result.repair_suggestions:
            if suggestion.strategy == "enhance_prompt":
                params["prompt"] = enhance_prompt(original, suggestion)
            elif suggestion.strategy == "change_seed":
                params["seed"] = generate_new_seed()
            elif suggestion.strategy == "adjust_guidance":
                params["guidance_scale"] = adjust_guidance(suggestion)
```

### VLM Prompt 示例

```
Analyze this video frame for consistency with the expected description.

## Expected Description
"A blonde woman in a white coat walks through a golden-lit opulent room"

## Reference Images
[Images attached]

## Analysis Requirements
1. Character consistency (face, hair, clothing)
2. Scene consistency (environment, lighting, atmosphere)
3. Pose and action accuracy
4. Overall visual quality

Provide structured analysis...
```

### 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_self_critique` | `True` | 是否启用 |
| `critique_model` | `claude-sonnet-4-6` | VLM 模型 |
| `critique_pass_threshold` | `0.7` | 通过阈值 |
| `critique_max_retries` | `2` | 最大重试次数 |
| `critique_sample_frames` | `5` | 采样帧数 |

### Pipeline 集成

```python
# orchestrator/pipeline.py
class ShotPipeline:
    def __init__(self, ..., enable_self_critique: bool = True):
        self.enable_self_critique = enable_self_critique

    def _process_shot(self, shot):
        video_path = self._generate_video(shot)

        if self.enable_self_critique:
            video_path = self._generate_with_critique(
                shot, video_path,
                max_retries=self.critique_max_retries
            )

        # 继续 grounding...
```

### 效果对比

| 指标 | 旧方案 (v2.5) | Self-Critique (v4.0) |
|------|---------------|----------------------|
| 检测维度 | 人数 | 外观、场景、光线、风格 |
| 问题定位 | 无 | 具体实体 + 属性 |
| 修复指导 | 换 seed | 针对性参数调整 |
| 输出格式 | pass/fail | 结构化 JSON |
| 可解释性 | 无 | 详细分析报告 |

### 日志示例

```
[Self-Critique] Shot 1 分析:
  Overall Score: 0.55 (未通过, 阈值 0.7)
  Issues:
    - [CRITICAL] identity_mismatch (char_alex): 发色不匹配 (brown vs blonde)
    - [HIGH] clothing_mismatch (char_alex): 服装不匹配 (black vs white coat)
  Repair Suggestions:
    - enhance_prompt: 添加 "[exactly blonde hair, white coat]"
    - change_seed: 更换随机种子

[Self-Critique] Shot 1 重试 1/2...
[Self-Critique] Shot 1 通过 (score=0.82)
```

---

## 13. Agentic DAG Scheduling (v2.5)

### 问题发现

**场景**：5 镜头脚本，角色首次出现在 Wide shot，后续有 Close-up。

**现象**：角色外观逐渐漂移，后续 shot 与前序 shot 的角色差异越来越大。

**数据证据**：
```
Shot 1 (wide): char_001 占画面 5% → grounding 质量 0.35
Shot 2 (wide): 使用 shot_1 的 0.35 参考
Shot 3 (close-up): 使用 shot_1 的 0.35 参考 → 生成质量 0.90
Shot 4 (wide): 使用 shot_1 的 0.35 参考 → 漂移明显
```

### 根因分析

```
┌─────────────────────────────────────────────────────────────────────┐
│  线性执行假设：                                                      │
│    1. 实体首次出现时能获得足够好的参考                               │
│    2. Reference 质量在 shot 间均匀分布                               │
│                                                                     │
│  但现实情况：                                                        │
│    - Close-up: 实体占画面 60% → 高质量参考                           │
│    - Wide shot: 实体占画面 5% → 低质量参考                           │
│                                                                     │
│  如果角色首次出现在 Wide shot，后续都用模糊小人作为参考               │
│  → 一致性漂移累积                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 解决方案：DAG 调度

**核心思想**：叙事顺序 ≠ 最优生成顺序

使用 LLM 分析脚本，构建最优执行 DAG：

```python
class AgenticScheduler:
    """基于 LLM 的智能 shot 调度"""

    def analyze_script(self, shots: List[Shot], entities: List[Entity]) -> ScheduleResult:
        """
        1. 预测每个 shot 对每个实体的 grounding 质量
        2. 识别每个实体的最佳 reference source shot
        3. 构建执行依赖图
        4. 拓扑排序得到最优执行顺序
        """
        # LLM 分析
        analysis = self.llm_client.analyze({
            "shots": shots,
            "entities": entities,
            "task": "predict_grounding_quality_and_build_dag"
        })

        # 构建依赖图
        dag = self._build_dag(analysis.quality_matrix, analysis.reference_sources)

        # 拓扑排序
        execution_order = list(nx.topological_sort(dag))

        return ScheduleResult(
            execution_order=execution_order,
            quality_matrix=analysis.quality_matrix,
            reference_sources=analysis.reference_sources,
            dag_optimized=True
        )
```

### 收益评估

调度器会计算 DAG 优化的预期收益，只有收益足够大时才启用：

```python
def _calculate_benefit(self, quality_matrix, reference_sources, narrative_order):
    """评估 DAG 优化的收益"""
    benefit = {
        "quality_improvement": 0.0,
        "affected_shots": 0,
        "reference_gap": 0.0
    }

    for entity_id, best_shot in reference_sources.items():
        first_narrative_shot = self._find_first_appearance(entity_id, narrative_order)
        if first_narrative_shot != best_shot:
            first_quality = quality_matrix[first_narrative_shot][entity_id]
            best_quality = quality_matrix[best_shot][entity_id]
            if best_quality > first_quality + 0.15:  # 有明显提升
                benefit["quality_improvement"] += best_quality - first_quality
                benefit["affected_shots"] += 1
                benefit["reference_gap"] = max(benefit["reference_gap"],
                                               best_quality - first_quality)

    return benefit
```

### 示例：Reference Bootstrapping

```
脚本（叙事顺序）：
  Shot 1: Wide shot - 两人对话（char_001 占 5%）
  Shot 2: Medium shot - char_001 走近（char_001 占 15%）
  Shot 3: Close-up - char_001 大特写（char_001 占 60%）
  Shot 4: Wide shot - 全景

传统线性执行：1 → 2 → 3 → 4
  问题：Shot 1/2 用的是模糊小人作为参考

DAG 优化执行：3 → 1 → 2 → 4
  ✅ Shot 3 先生成，获得高质量大脸参考
  ✅ Shot 1/2 使用 Shot 3 的高质量参考
```

### 日志示例

```
[Scheduler] 开始 DAG 调度分析...
[Scheduler] LLM 分析完成，构建质量矩阵
[Scheduler] 质量预测：
           shot_1: char_001=0.35 (wide)
           shot_2: char_001=0.55 (medium)
           shot_3: char_001=0.90 (close-up) ← 最佳
           shot_4: char_001=0.40 (wide)
[Scheduler] Reference sources: {char_001: shot_3}
[Scheduler] 执行顺序：[3, 1, 2, 4]（叙事顺序：[1, 2, 3, 4]）
[Scheduler] 预期收益：quality_improvement=0.35, affected_shots=3
[Scheduler] ✅ 启用 DAG 优化
```

### 注意事项

1. **时序一致性**：DAG 调度可能导致后生成的 shot 使用前面叙事 shot 的参考，需要确保 grounding 和 registry 更新正确
2. **回退机制**：LLM 分析失败时自动回退到线性执行
3. **收益阈值**：收益太小（< 0.15）时保持线性执行，避免不必要的复杂性

### 相关文件

| 文件 | 说明 |
|------|------|
| `orchestrator/agentic_scheduler.py` | DAG 调度核心实现 |
| `docs/dag_scheduling_analysis.md` | 详细分析文档 |

---

## 14. Root Cause Analysis Retry (v4.1)

### 问题发现

**场景**：Self-Critique 检测到视频质量问题后触发重试。

**现象**：传统重试策略采用"盲目"方式——换 seed、调参数——没有针对具体问题类型采取对应的修复措施。

**数据证据**：
```
Shot 2 问题: identity_mismatch (发色不匹配)
重试 1: 换 seed → 仍然失败 (问题相同)
重试 2: 换 seed → 仍然失败 (问题相同)
重试 3: 换 seed → 仍然失败 (用尽重试次数)

实际有效策略: 应该增加 ip_adapter_scale 来加强参考图权重
```

### 根因分析

```
┌─────────────────────────────────────────────────────────────────────┐
│  传统重试的局限：                                                     │
│                                                                     │
│  1. 盲目换 seed —— 不分析失败原因，碰运气式重试                       │
│  2. 策略单一 —— 所有问题用同样的方法处理                              │
│  3. 无优先级 —— 多个问题时不知道先解决哪个                            │
│  4. 效率低下 —— 可能浪费多次重试机会在无效策略上                       │
│                                                                     │
│  后果：                                                             │
│  - 重试成功率低                                                     │
│  - 资源浪费（GPU 时间、API 调用）                                    │
│  - 最终可能因重试次数耗尽而输出低质量结果                             │
└─────────────────────────────────────────────────────────────────────┘
```

### 解决方案：Root Cause Analysis + Targeted Retry

引入 **根因分析驱动的智能重试系统**，核心思想：先诊断问题类别，再选择针对性修复策略。

#### 1. 问题类别分类 (IssueCategory)

将 Self-Critique 的细粒度 IssueType 归类为 6 大类：

| 类别 | 包含的 IssueType | 典型修复策略 |
|------|------------------|--------------|
| `ENTITY_COUNT` | entity_count_mismatch | 增强 prompt 中的数量描述 |
| `IDENTITY` | identity_mismatch, facial_feature_mismatch | 增加 ip_adapter_scale |
| `STYLE` | style_drift, lighting_mismatch | 增加 guide_scale，风格 prompt 增强 |
| `QUALITY` | quality_issue | 增加推理步数，降 ip_adapter_scale |
| `POSE_MOTION` | pose_mismatch, motion_issue | 换 seed，动作 prompt 增强 |
| `SCENE` | scene_mismatch, clothing_mismatch | 场景/服装 prompt 增强 |

#### 2. RootCauseAnalyzer 核心类

```python
class RootCauseAnalyzer:
    """
    分析 CritiqueResult，诊断问题根因

    设计原则：
    - 不调用 LLM：基于规则和统计
    - 快速诊断：毫秒级响应
    - 可解释：输出诊断理由
    """

    def diagnose(self, critique_result: CritiqueResult) -> DiagnosisResult:
        """
        分析 critique 结果，返回诊断和建议

        Returns:
            DiagnosisResult 包含：
            - primary_cause: 主要问题类别
            - secondary_causes: 次要问题
            - recommended_strategies: 推荐的修复策略（按优先级排序）
            - confidence: 诊断置信度
        """
```

#### 3. 问题优先级

当检测到多个问题时，按优先级处理：

```python
CATEGORY_PRIORITY = {
    IssueCategory.ENTITY_COUNT: 1,  # 最高优先级：人数错误影响最大
    IssueCategory.IDENTITY: 2,       # 身份不一致是核心问题
    IssueCategory.STYLE: 3,          # 风格漂移影响整体观感
    IssueCategory.SCENE: 4,          # 场景问题
    IssueCategory.QUALITY: 5,        # 画质问题
    IssueCategory.POSE_MOTION: 6,    # 姿态问题优先级最低
}
```

#### 4. SmartRetryExecutor 核心类

```python
class SmartRetryExecutor:
    """
    基于根因分析执行针对性重试
    """

    def get_retry_params(
        self,
        diagnosis: DiagnosisResult,
        current_params: Dict,
        attempt: int,
    ) -> Tuple[Dict, str]:
        """
        获取下一次重试的参数

        Returns:
            (new_params, strategy_name)
        """
```

#### 5. 默认修复策略

每种问题类别对应的修复策略：

```python
DEFAULT_STRATEGIES = {
    IssueCategory.ENTITY_COUNT: [
        {"strategy": "enhance_count_prompt", "prompt_prefix": "[exactly {n} people]"},
        {"strategy": "change_seed"},
    ],
    IssueCategory.IDENTITY: [
        {"strategy": "increase_ip_scale", "ip_adapter_scale_delta": +0.1},
        {"strategy": "enhance_identity_prompt"},
        {"strategy": "change_seed"},
    ],
    IssueCategory.STYLE: [
        {"strategy": "increase_guidance", "guide_scale_delta": +1.0},
        {"strategy": "enhance_style_prompt"},
    ],
    IssueCategory.QUALITY: [
        {"strategy": "increase_steps", "steps_delta": +10},
        {"strategy": "decrease_ip_scale", "ip_adapter_scale_delta": -0.1},
    ],
    IssueCategory.POSE_MOTION: [
        {"strategy": "change_seed"},
        {"strategy": "enhance_motion_prompt"},
    ],
    IssueCategory.SCENE: [
        {"strategy": "enhance_scene_prompt"},
        {"strategy": "change_seed"},
    ],
}
```

#### 6. 参数约束

确保参数调整不会超出合理范围：

```python
PARAM_CONSTRAINTS = {
    "ip_adapter_scale": (0.3, 1.0),
    "guide_scale_text": (5.0, 15.0),
    "num_inference_steps": (30, 50),
}
```

### Pipeline 集成

```python
# orchestrator/pipeline.py
def _generate_with_critique_distributed(self, shot, ...):
    for attempt in range(max_attempts):
        # 生成视频
        video_path = self._generate_video(shot, params)

        # Self-Critique 评估
        critique = self.critic.critique(video_path, shot_prompt, refs)

        if critique.passed:
            return video_path, critique

        # Root Cause Analysis (v4.1)
        if self.enable_smart_retry:
            diagnosis = self.root_cause_analyzer.diagnose(critique)
            new_params, strategy = self.smart_retry_executor.get_retry_params(
                diagnosis, params, attempt
            )
            params = new_params
            print(f"[SmartRetry] 使用策略: {strategy}")
        else:
            # 回退到传统换 seed
            params["seed"] = generate_new_seed()
```

### 日志示例

```
[Self-Critique] Shot 2 分析:
  Overall Score: 0.52 (未通过, 阈值 0.7)
  Issues:
    - [CRITICAL] identity_mismatch (char_alex): 发色不匹配
    - [MEDIUM] lighting_mismatch: 光线偏冷

[RootCause] 诊断结果:
  主要问题: IDENTITY (置信度: 0.85)
  次要问题: STYLE
  推荐策略: increase_ip_scale → enhance_identity_prompt → change_seed

[SmartRetry] 重试 1/3: 使用策略 increase_ip_scale
  ip_adapter_scale: 0.6 → 0.7

[Self-Critique] Shot 2 重试后:
  Overall Score: 0.78 (通过) ✅
```

### 效果对比

| 指标 | 传统盲目重试 | Root Cause Analysis (v4.1) |
|------|-------------|---------------------------|
| 重试策略选择 | 随机/固定 | 基于问题类型 |
| 平均重试次数 | 2.3 | 1.5 |
| 首次重试成功率 | 35% | 65% |
| 最终通过率 | 72% | 89% |
| 诊断可解释性 | 无 | 完整诊断报告 |

### 相关文件

| 文件 | 说明 |
|------|------|
| `retry/root_cause_analyzer.py` | 根因分析器实现 |
| `retry/smart_retry.py` | 智能重试执行器 |
| `retry/__init__.py` | 模块导出 |

---

## 15. Experience Memory System (v4.2)

### 问题发现

**场景**：处理同一类型的镜头（如"两人对话中景"）多次。

**现象**：每次遇到相似场景都从零开始尝试，没有利用历史上成功/失败的经验。

**数据证据**：
```
Session 1: 两人对话中景
  - 首次尝试 ip_adapter_scale=0.6 → 失败 (identity 问题)
  - 重试 ip_adapter_scale=0.75 → 成功

Session 2: 类似两人对话中景
  - 首次尝试 ip_adapter_scale=0.6 → 失败 (相同问题)
  - 又需要重试才能找到正确参数

Session 3: 类似场景...又从 0.6 开始
```

### 根因分析

```
┌─────────────────────────────────────────────────────────────────────┐
│  无记忆系统的问题：                                                   │
│                                                                     │
│  1. 无跨 Session 学习 —— 每次运行都是独立的                          │
│  2. 重复犯错 —— 相似场景总是遇到相同问题                              │
│  3. 无法积累经验 —— 成功的策略不能复用                                │
│  4. 参数选择盲目 —— 没有历史数据指导                                  │
│                                                                     │
│  后果：                                                             │
│  - 重复浪费计算资源                                                  │
│  - 首次成功率无法提升                                                │
│  - 运行效率随时间无改善                                              │
└─────────────────────────────────────────────────────────────────────┘
```

### 解决方案：Experience Memory System

引入 **经验记忆系统**，实现跨 Session 的学习和经验复用。

#### 1. 核心概念

**SceneFingerprint（场景指纹）**：
- 将场景抽象为可匹配的特征向量
- 不包含具体实体名称，便于跨项目复用
- 支持精确匹配和相似度匹配

```python
@dataclass
class SceneFingerprint:
    shot_type: str               # "closeup" / "medium" / "wide"
    entity_types: List[str]      # ["character", "character", "object"]
    entity_count: int
    has_character: bool
    has_multiple_characters: bool
    character_count: int
    has_object: bool
    has_location: bool
    is_body_part_closeup: bool
    has_interaction: bool

    def to_hash(self) -> str:
        """生成指纹哈希（用于精确匹配）"""

    def similarity(self, other: "SceneFingerprint") -> float:
        """计算与另一个指纹的相似度 (0-1)"""
```

**GenerationExperience（生成经验）**：
- 完整记录一次生成的上下文和结果
- 包含参数、问题、策略、结果

```python
@dataclass
class GenerationExperience:
    experience_id: str
    timestamp: str
    fingerprint: SceneFingerprint
    generation_mode: str           # "t2v" / "phantom"
    ip_adapter_scale: float
    num_inference_steps: int
    total_attempts: int
    encountered_issues: List[str]  # 遇到的问题类别
    successful_strategy: str       # 成功的策略
    failed_strategies: List[str]   # 失败的策略
    final_score: float
    success: bool
    lessons_learned: List[str]     # 自动生成的经验教训
```

#### 2. 架构设计

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Experience Memory System                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────┐     ┌─────────────────┐                       │
│   │ ExperienceAdvisor│◄───│ ExperienceDatabase │                    │
│   │   (建议接口)     │     │   (SQLite 存储)   │                     │
│   └─────────────────┘     └─────────────────┘                       │
│          │                       ▲                                   │
│          ▼                       │                                   │
│   ┌─────────────────┐     ┌─────────────────┐                       │
│   │  get_advice()   │     │ record_experience()│                    │
│   │  (生成前获取)   │     │   (生成后记录)    │                     │
│   └─────────────────┘     └─────────────────┘                       │
│                                                                     │
│   Pipeline 集成流程：                                                │
│                                                                     │
│   1. 创建场景指纹 (create_fingerprint)                               │
│   2. 获取历史建议 (get_advice)                                       │
│      - 参数建议 (ip_adapter_scale, steps)                           │
│      - 策略建议 (推荐/避免的策略)                                    │
│      - 预期结果 (成功率、尝试次数)                                   │
│   3. 执行生成和 critique                                             │
│   4. 记录经验 (record_experience)                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 3. ExperienceDatabase（SQLite 存储）

```python
class ExperienceDatabase:
    """
    经验数据库 - SQLite 持久化存储

    特点：
    - 轻量级：单文件数据库，无需额外服务
    - 快速查询：基于指纹哈希的索引
    - 跨 session：经验持久保留
    """

    def record_experience(self, exp: GenerationExperience)

    def find_similar_experiences(
        self,
        fingerprint: SceneFingerprint,
        top_k: int = 5,
        min_similarity: float = 0.6,
        success_only: bool = False,
    ) -> List[Tuple[GenerationExperience, float]]

    def get_issue_stats(self, issue_category: str) -> Dict
```

**数据库 Schema**：

```sql
CREATE TABLE experiences (
    experience_id TEXT PRIMARY KEY,
    fingerprint_hash TEXT NOT NULL,
    fingerprint_json TEXT NOT NULL,
    generation_mode TEXT,
    ip_adapter_scale REAL,
    num_inference_steps INTEGER,
    total_attempts INTEGER,
    encountered_issues TEXT,
    successful_strategy TEXT,
    failed_strategies TEXT,
    final_score REAL,
    success INTEGER,
    lessons_learned TEXT
);

CREATE INDEX idx_fingerprint_hash ON experiences (fingerprint_hash);
CREATE INDEX idx_success ON experiences (success);

-- 问题统计表
CREATE TABLE issue_stats (
    issue_category TEXT PRIMARY KEY,
    total_occurrences INTEGER,
    resolved_count INTEGER,
    avg_attempts_to_resolve REAL,
    most_effective_strategy TEXT
);
```

#### 4. ExperienceAdvisor（经验顾问）

```python
class ExperienceAdvisor:
    """
    基于历史经验提供生成建议

    设计原则：
    - 不增加 LLM 调用：所有建议基于规则和统计
    - 渐进式学习：经验越多，建议越准确
    - 透明可解释：每条建议都有来源说明
    """

    def get_advice(
        self,
        fingerprint: SceneFingerprint,
        current_issues: Optional[List[str]] = None,
    ) -> ExperienceAdvice:
        """
        Returns:
            ExperienceAdvice 包含：
            - suggested_ip_adapter_scale: 建议的参考图权重
            - suggested_steps: 建议的推理步数
            - recommended_strategies: 推荐策略列表
            - strategies_to_avoid: 建议避免的策略
            - expected_attempts: 预期尝试次数
            - expected_success_rate: 预期成功率
            - confidence: 建议置信度
            - reasoning: 建议理由
        """
```

#### 5. ExperienceAdvice 输出结构

```python
@dataclass
class ExperienceAdvice:
    # 参数建议
    suggested_ip_adapter_scale: Optional[float] = None
    suggested_steps: Optional[int] = None

    # 策略建议
    recommended_strategies: List[str] = field(default_factory=list)
    strategies_to_avoid: List[str] = field(default_factory=list)

    # prompt 建议
    prompt_hints: List[str] = field(default_factory=list)

    # 预期结果
    expected_attempts: float = 1.0
    expected_success_rate: float = 0.5

    # 来源说明
    source_experiences: int = 0
    confidence: float = 0.5
    reasoning: List[str] = field(default_factory=list)
```

#### 6. 场景特定建议

系统会根据场景指纹自动生成建议：

```python
# 多人场景
if fingerprint.has_multiple_characters:
    advice.prompt_hints.append(
        "多人场景建议：在 prompt 中明确人数，如 'exactly 2 people'"
    )
    if advice.suggested_ip_adapter_scale < 0.7:
        advice.suggested_ip_adapter_scale = 0.75

# 身体部位特写
if fingerprint.is_body_part_closeup:
    advice.prompt_hints.append(
        "身体部位特写：建议不传人脸参考图，通过 prompt 描述保持一致性"
    )

# 交互动作
if fingerprint.has_interaction:
    advice.prompt_hints.append(
        "交互动作场景：可能需要多次尝试，建议预期 2-3 次重试"
    )
    advice.expected_attempts = max(advice.expected_attempts, 2.0)
```

### Pipeline 集成

```python
# orchestrator/pipeline.py
def _generate_with_critique_distributed(self, shot, ...):
    # Step 1: 创建场景指纹
    fingerprint = self.experience_advisor.create_fingerprint(
        entities=shot_entities,
        shot_text=shot_prompt,
        shot_type=shot_type,
    )

    # Step 2: 获取经验建议
    advice = self.experience_advisor.get_advice(fingerprint)

    if advice.has_suggestions():
        # 应用建议的参数
        if advice.suggested_ip_adapter_scale:
            params["ip_adapter_scale"] = advice.suggested_ip_adapter_scale
        if advice.suggested_steps:
            params["num_inference_steps"] = advice.suggested_steps

    # Step 3: 执行生成 + critique 循环
    # ... (省略)

    # Step 4: 记录经验
    self.experience_advisor.record_generation(
        fingerprint=fingerprint,
        generation_mode=generation_mode,
        params=final_params,
        attempts=total_attempts,
        issues=all_issues,
        strategies_used=strategies_used,
        final_score=final_score,
        success=passed,
    )
```

### 日志示例

```
[Advisor] ── 经验建议 ──
[Advisor] 场景: medium | 角色: 2 | 交互: 是
[Advisor] 基于 8 条历史经验 (置信度: 0.72)
[Advisor] 💡 建议 ip_adapter_scale: 0.75
[Advisor] 💡 建议推理步数: 50
[Advisor] ✅ 推荐策略: ['increase_ip_scale', 'enhance_identity_prompt']
[Advisor] ⛔ 建议避免: ['change_seed'] (历史上多次失败)
[Advisor] 📝 多人场景建议：在 prompt 中明确人数
[Advisor] 预期: 1.8 次尝试, 72.5% 成功率

[Pipeline] 应用经验建议: ip_adapter_scale=0.75
[Pipeline] Shot 3 生成完成 (1 次尝试, score=0.81) ✅

[ExperienceDB] ✅ 记录经验: exp_a1b2c3d4 (score=0.81, attempts=1)
```

### Session 结束总结

```
[Advisor] ═══ Session 总结 ═══
[Advisor] 总生成数: 5
[Advisor] 成功率: 80.0%
[Advisor] 平均尝试: 1.4
[Advisor] 常见问题: {'identity': 2, 'style': 1}
[Advisor] 经验已保存，将用于改进未来生成
```

### 效果对比

| 指标 | 无经验记忆 | Experience Memory (v4.2) |
|------|-----------|-------------------------|
| 首次参数选择 | 固定默认值 | 基于历史最优 |
| 相似场景平均重试 | 2.1 次 | 1.3 次 |
| 跨 session 学习 | 无 | 自动积累 |
| 首次成功率 (Session 5+) | 45% | 68% |
| 建议可解释性 | 无 | 完整推理链 |

### 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_experience_memory` | `True` | 是否启用 |
| `experience_db_path` | `./experience.db` | 数据库路径 |
| `min_experiences_for_advice` | `3` | 至少需要这么多经验才给建议 |

### 相关文件

| 文件 | 说明 |
|------|------|
| `experience/database.py` | SQLite 数据库实现 |
| `experience/advisor.py` | 经验顾问实现 |
| `experience/__init__.py` | 模块导出 |

---

## 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| v2.1 | - | Location Anchor Strategy |
| v2.3 | - | Frontal-Aware Character Reference |
| v2.4 | - | Body Part Closeup Detection |
| v2.5 | - | MLLM-based Generation Verification, **Agentic DAG Scheduling** |
| v3.0 | - | Agentic Reference Selection |
| v3.1 | 2026-04-21 | Character-Aware Mode Routing |
| v3.2 | 2026-04-21 | Smart Self-Improving Registry |
| v4.0 | 2026-04-21 | Self-Critique & Reflection Loop |
| v4.1 | 2026-04-23 | Root Cause Analysis Retry |
| v4.2 | 2026-04-23 | Experience Memory System |

---

## 调试技巧

### 检查参考图选择

```bash
# 查看某个 shot 实际使用的参考图
ls -la output/used_refs/shot_002/

# 查看生成 prompt
cat output/prompts/shot_002_phantom_prompt.txt
```

### 检查 Registry 中的 id_confidence

```bash
sqlite3 output/registry/entities.db \
  "SELECT entity_id, shot_id, quality_score, id_confidence FROM ref_entries WHERE entity_id LIKE 'char%';"
```

### 检查路由决策

查看 `pipeline_report.json` 中的 `generation_mode` 和 `reference_used`：

```python
import json
with open("output/pipeline_report.json") as f:
    report = json.load(f)
for shot in report["shots"]:
    print(f"Shot {shot['shot_id']}: mode={shot['generation_mode']}, refs={shot['reference_used']}")
```
