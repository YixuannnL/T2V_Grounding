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

## 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| v2.1 | - | Location Anchor Strategy |
| v2.3 | - | Frontal-Aware Character Reference |
| v2.4 | - | Body Part Closeup Detection |
| v2.5 | - | MLLM-based Generation Verification |
| v3.0 | - | Agentic Reference Selection |
| v3.1 | 2026-04-21 | Character-Aware Mode Routing |

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
