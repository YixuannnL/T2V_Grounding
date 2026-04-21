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
| `critique_model` | `claude-sonnet-4-6-Anthropic` | VLM 模型 |
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

## 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| v2.1 | - | Location Anchor Strategy |
| v2.3 | - | Frontal-Aware Character Reference |
| v2.4 | - | Body Part Closeup Detection |
| v2.5 | - | MLLM-based Generation Verification |
| v3.0 | - | Agentic Reference Selection |
| v3.1 | 2026-04-21 | Character-Aware Mode Routing |
| v3.2 | 2026-04-21 | Smart Self-Improving Registry |
| v4.0 | 2026-04-21 | Self-Critique & Reflection Loop |

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
