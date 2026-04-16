# T2V Grounding — 多镜头一致性视频生成系统

## 这个项目解决什么问题

当前主流的文本生成视频（T2V）模型（如 Wan2.1、Sora）存在一个根本性缺陷：**每次生成都是独立的**。如果你用同样的文字描述 "一个穿蓝色夹克的男人" 生成 5 个视频片段，每个片段里这个男人的长相、发型、夹克颜色都会不同。这在电影/短剧等需要跨镜头角色一致性的场景下完全不可用。

**本项目的目标：** 给定一段多镜头的分镜脚本（如 5 个镜头描述同一个故事），让系统自动生成每个镜头的视频，并保证同一个角色在所有镜头中外观一致。

**解决思路：** 生成第 1 个镜头后，从视频中"认出"角色，截取角色的图像；生成第 2 个镜头时，把这张图传给模型说"这个角色就长这样"，让模型参照外观生成——如此滚动传递，实现跨镜头一致性。

---

## 核心流程详解

整个 Pipeline 按镜头顺序执行，每个镜头经历 **解析 → 路由 → 生成 → Grounding → 入库** 五个阶段。

### 阶段一：LLM 实体解析

**文件**：`entity_parser/parser.py`

用 LLM（Claude）读取当前镜头的文字描述，从中识别出所有需要跨镜头保持一致的"实体"：

| 实体类型 | 示例 | Grounding 优先级 |
|---------|------|-----------------|
| `character` | "留胡子的男人"、"戴白色面具的人" | high |
| `object` | "狙击步枪"、"皮质日记本" | medium |
| `location` | "沙漠废墟"、"私人书房" | medium |
| `style` | "赛博朋克风格" | low |

**每个实体的数据结构**：
```python
@dataclass
class Entity:
    entity_id: str            # 跨镜头唯一 ID，如 "char_bearded_man"
    type: str                 # character / object / location / style
    text_description: str     # 原文描述
    attributes: dict          # {"gender": "male", "age": "40s", "weapon": "sniper rifle"}
    is_new: bool              # 是否首次出现
    grounding_priority: str   # high / medium / low
    aliases: List[str]        # 别名列表 ["the man", "he", "the armed man"]
```

**跨镜头共指消解**：
- Shot 1 写了 "a bearded man with a sniper rifle"
- Shot 2 写了 "the armed man runs forward"
- LLM 会把 "the armed man" 识别为同一个 `char_bearded_man`，而不是创建新实体

**global_caption 预解析**（推荐）：
```yaml
global_caption: >
  The video depicts a tense scene where a bearded man armed with
  a sniper rifle pursues a mysterious masked figure in black.
```
Pipeline 在正式处理镜头前，会：
1. 用 `shot_id=0` 解析 global_caption，提取所有实体建立"实体图谱"
2. 调用 `_extract_global_context()` 提取全局语义（风格、氛围、叙事背景）

后续每个 shot 解析时参照实体图谱做共指消解，生成 prompt 时注入全局语义上下文。

### 阶段二：路由决策

**文件**：`orchestrator/pipeline.py`

查询 **Entity Registry**（SQLite 数据库），判断当前镜头中的实体是否已有历史参考图。

**实体分类处理**：
- **非 location 实体**（character, object）：按 high → medium 优先级排序，最多取 3 个
- **location 实体**：单独处理，确保场景一致性
  - 如果 registry 中已有该 location 的参考图 → **必须使用**（非新场景）
  - 如果 registry 中无记录 → 新场景，生成后会自动 grounding 入库

**Agentic Light-Aware Close-up Strategy（智能光线感知近景策略）**：

传统方案对 close-up 镜头一律不传 location 参考图，但这会导致近景镜头的光线/色调与整体视频不融合（例如，办公室场景的金色暖光在近景中丢失）。

本项目引入 **Agentic 决策机制**：由 LLM 分析场景光线复杂度，智能决定是否传入 location 参考图。

**决策流程**：
```
┌─────────────────────────────────────────────────────────────┐
│              Close-up Agentic Light Analysis                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 检测镜头类型（close-up / wide / medium）                 │
│                                                             │
│  2. 若为 close-up 且有 location 参考图可用：                 │
│     │                                                       │
│     └─► LLM 分析光线复杂度 (complexity_score 1-5)           │
│         ├── 1-2 (简单): 均匀光、单一光源                    │
│         │   → 不传 location，通过 prompt 描述光线           │
│         │                                                   │
│         └── 3-5 (复杂): 多色光源、特殊色调、逆光             │
│             → 传 location 参考图，保持色调一致               │
│                                                             │
│  3. 根据决策结果调整 prompt 层次                             │
│     - 不传 location → 添加 [Lighting Context] 层             │
│     - 传 location → 正常 4 层 prompt                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Prompt 分层结构（四层）**：
| 层级 | 名称 | 来源 | 作用 |
|------|------|------|------|
| Layer 1 | Global Context | `global_caption` 提取 | 风格、氛围、叙事背景 |
| Layer 2 | Lighting Context | LLM 光线分析 | 仅 close-up + 不传 location 时注入 |
| Layer 3 | Shot Entities | 本 shot 实体描述 | 角色、物体、场景 |
| Layer 4 | Shot Text | 原始镜头描述 | 动作、构图 |

**光线分析输出示例**：
```json
{
  "lighting_description": "warm ambient lighting with golden tones from large windows and decorative lamps",
  "color_tone": "warm golden and amber tones",
  "complexity_score": 3,
  "needs_location_ref": true,
  "reason": "The ornate room with gold furniture and large windows creates specific warm reflections that would affect skin tones in close-ups"
}
```

**设计原理**：
- **简单光线（1-2）**：均匀日光、单一室内灯，文字描述足够还原
- **复杂光线（3-5）**：多色霓虹、金色反射、逆光等，需要图像参考保持色调
- Phantom 最多支持 4 张参考图，close-up 不传 location 时角色可用满 4 张

| 镜头类型 | Location 决策 | 说明 |
|----------|---------------|------|
| close-up | **Agentic 决策** | LLM 分析光线复杂度后决定 |
| wide shot | 始终传 | 场景一致性 |
| 其他（中景） | 始终传 | 场景一致性 |

```
┌────────────────────────────────────────────────────────────┐
│ Step 1: 分离 location 和非 location 实体                    │
│                                                            │
│ Step 2: 检测镜头类型（close-up / wide / medium）            │
│   close-up → 不传 location，角色最多 4 个                   │
│   wide     → 传 location，角色最多 3 个                     │
│   medium   → 传 location，角色最多 3 个                     │
│                                                            │
│ Step 3: 处理非 location 实体（使用 earliest_high_quality 锚点策略）│
│   对于 character 类型：                                     │
│     Registry.query_anchor(entity_id, min_quality=0.4,      │
│                           high_quality_threshold=0.85)     │
│     【锚点策略】优先"最早的大正脸"，而非单纯最早 shot       │
│     → 若最早 shot 质量 < 0.85，会去找后续 shot 的高质量正脸 │
│                                                            │
│ Step 4: 处理 location 实体（场景一致性）                    │
│   Registry.query(loc_entity_id, anchor_strategy="earliest")│
│     有参考图 → 必须使用（保证场景一致性）                   │
│     无参考图 → 新场景，跳过                                 │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │ reference_used 列表有图？                │
        │                                         │
        │   YES ──▶ generation_mode = "phantom"   │
        │           使用 Phantom S2V（参考图+文本）│
        │                                         │
        │   NO  ──▶ generation_mode = "t2v"       │
        │           使用 WanT2V（纯文本生成）      │
        └─────────────────────────────────────────┘
```

**锚点策略（防止误差累积）**：

这是本系统的核心设计之一。观察发现：越往后的 shot，生成的人脸越可能偏离原始外观。如果每次都用最近 shot 的 grounding 结果作为参考图，误差会逐步累积。

**解决方案**：对于 character 类型实体，使用 **"earliest_high_quality"** 策略选择锚点参考图——优先选择"最早的大正脸"，而非单纯选最早 shot。

**策略细节**：
```
┌─────────────────────────────────────────────────────────────────┐
│            Earliest High-Quality Anchor Strategy                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  问题：Shot 1 可能是侧脸/部分脸，Shot 2 的 close-up 可能        │
│       有更清晰的正脸。单纯选"最早"会选到质量差的侧脸。          │
│                                                                 │
│  规则（按优先级）：                                              │
│  1. 如果最早 shot 有 ≥ high_quality_threshold (0.85) 的参考图   │
│     → 选它（最早的高质量正脸）                                  │
│                                                                 │
│  2. 如果最早 shot 只有中等质量 (< 0.85)：                       │
│     - 查找所有 shot 中最早的高质量参考图 (≥ 0.85)               │
│     - 如果有 → 选它（高质量正脸优先于早期出现）                 │
│     - 如果没有 → 回退选最早 shot（防止误差累积）                │
│                                                                 │
│  3. 同等高质量时，选 shot_id 更小的（更早出现）                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**示例**：
```
char_elderly_man 的参考图：
  - Shot 1: 0.72 (侧脸，wide shot)
  - Shot 2: 0.93 (正脸，close-up)
  - Shot 4: 0.88 (正脸)

旧策略 (earliest_good): 选 Shot 1 的 0.72 (侧脸) ❌
新策略 (earliest_high_quality): 选 Shot 2 的 0.93 (正脸) ✅
  → 因为 Shot 1 的 0.72 < 0.85 阈值，而 Shot 2 有 ≥ 0.85 的高质量正脸
```

```python
# 旧逻辑（只看 shot_id，可能选到侧脸）
refs = registry.query(entity_id, top_k=1)  # ORDER BY shot_id ASC

# 新逻辑（earliest_high_quality 锚点策略）
anchor = registry.query_anchor(
    entity_id,
    min_quality=0.4,
    high_quality_threshold=0.85  # 高质量正脸阈值
)
# 日志：选择后续高质量正脸 (shot=2, score=0.93) 优于最早shot (shot=1, score=0.72)
```

**路由逻辑要点**：
- Shot 1 必定走 `t2v`（首镜头，Registry 为空）
- Shot 2+ 若历史有参考图则走 `phantom`
- 新角色首次出现也走 `t2v`，生成后入库，下一镜头就有参考了
- **每个 shot 使用递增的 seed**（`base_seed + shot_id`），增加生成多样性

### 阶段 2.5：Shot 1 实体数量验证（Generation-Verification Loop）

**文件**：`verification/entity_count_verifier.py`、`orchestrator/pipeline.py`

**问题背景**：
Shot 1 是纯 T2V 生成，没有参考图约束。如果 prompt 描述"三个人"，但模型生成了四个人，这个错误会传播到所有后续 shot（因为后续 shot 都基于 Shot 1 的 grounding 结果作为 anchor reference）。这是典型的**错误累积问题**。

**解决方案**：引入 Generation-Verification Loop

```
┌─────────────────────────────────────────────────────────────┐
│              Shot 1 Generation-Verification Loop             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. LLM 解析 prompt → 提取预期人数 (expected_count)          │
│       例: "three men in suits" → expected_count = 3         │
│                                                             │
│  2. T2V 生成 → video_shot1.mp4                              │
│                                                             │
│  3. 验证模块：                                               │
│     - 采样 3 帧（跳过首尾 10%）                              │
│     - Person Detection (Grounding DINO)                     │
│     - NMS 去重                                              │
│     - 取众数作为 actual_count                               │
│                                                             │
│  4. 比较 expected_count vs actual_count：                   │
│     ├── 匹配 → ✅ 通过，继续 grounding 入库                 │
│     └── 不匹配 → ❌ 重试                                    │
│           ├── 尝试 1-2: 换 seed 重试                        │
│           └── 尝试 3+: 使用增强 prompt                      │
│               "[IMPORTANT: exactly 3 people] + original"   │
│                                                             │
│  5. 最多重试 3 次，超出则警告并使用最后结果                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**配置参数**：
```python
pipeline = T2VGroundingPipeline(
    enable_shot1_verification=True,   # 开启 Shot 1 验证
    shot1_max_retries=3,              # 最大重试次数
    shot1_verify_person_count=True,   # 验证人数
)
```

**验证报告**（保存在 `pipeline_report.json`）：
```json
{
  "shots": [{
    "shot_id": 1,
    "metadata": {
      "verification": {
        "verified": true,
        "final_attempt": 2,
        "final_passed": true,
        "expected_person_count": 3,
        "actual_person_count": 3,
        "retry_history": [
          {"attempt": 1, "seed": 43, "status": "count_mismatch", "expected": {"person": 3}, "actual": {"person": 4}},
          {"attempt": 2, "seed": 1043, "status": "passed", "expected": {"person": 3}, "actual": {"person": 3}}
        ]
      }
    }
  }]
}
```

**设计考量**：
- 只验证 T2V 模式（S2V 模式有 reference 约束，人数错误概率低）
- 人数验证是最关键的（物体数量通常不那么敏感）
- 增强 prompt 在第 3 次尝试才启用，避免过度约束影响画面质量

### 阶段三：四层 Prompt 构建

**文件**：`orchestrator/pipeline.py`、`entity_parser/parser.py`

生成 prompt 时**不直接使用 global_caption 原文**（因为它包含整个故事的叙事，会把后续镜头的角色注入当前镜头），而是分**四层**精细构建：

#### Layer 1: Global Context（全局语义上下文）

由 `build_global_context_prompt()` 从 `global_caption` 中提取，包含风格、氛围、叙事背景等**跨越所有镜头的共性信息**，不包含具体角色/物体描述：

```python
def build_global_context_prompt() -> str:
    """
    输出格式：
    [Global Context]
    Visual style: cinematic, dramatic lighting.
    Mood: tense, suspenseful.
    Setting: sandy desolate desert environment.
    Narrative: a standoff and pursuit sequence.
    """
```

#### Layer 2: Lighting Context（光线引导，仅 close-up 不传 location 时）

当 close-up 镜头由 Agentic 决策不传 location 参考图时，由 `build_closeup_lighting_prompt()` 注入光线描述，确保近景镜头的光线和色调与整体视频一致：

```python
def build_closeup_lighting_prompt(analysis: CloseupLightingAnalysis) -> str:
    """
    输出格式：
    [Lighting Context for Close-up]
    Lighting: warm ambient lighting with golden tones from large windows.
    Color tone: warm golden and amber tones.
    The background should be softly blurred while maintaining consistent lighting.
    """
```

#### Layer 3: Shot Context（当前镜头实体描述）

由 `build_shot_context()` 构建，**只包含本 shot 实际出现的实体**，从实体图谱中查找完整属性：

```python
def build_shot_context(parse_result) -> str:
    """
    输出格式：
    [Shot Entities]
    Character: bearded man (gender: male, age: 40s, weapon: sniper rifle).
    Scene: sandy desolate environment (lighting: harsh sunlight).
    """
```

#### Layer 4: Shot Text（原始镜头描述）

直接使用 config 中的镜头文本描述：

```
A close-up shot shows the bearded man looking through the scope of his rifle.
```

#### 最终 Prompt 组装

```python
prompt_parts = []
prompt_parts.append(global_context)      # Layer 1: 全局语义
if is_closeup and not include_location and closeup_lighting_analysis:
    prompt_parts.append(lighting_prompt) # Layer 2: 光线引导（仅特定条件）
prompt_parts.append(shot_context)        # Layer 3: 实体描述
prompt_parts.append(shot.text)           # Layer 4: 动作描述
gen_prompt = "\n\n".join(prompt_parts)
```

**实际输出示例（Close-up 镜头，不传 location）**：
```
[Global Context]
Visual style: cinematic, warm lighting.
Mood: professional, conversational.
Setting: ornate office with gold furniture.
Narrative: a business meeting.

[Lighting Context for Close-up]
Lighting: warm ambient lighting with golden tones from large windows and decorative lamps.
Color tone: warm golden and amber tones.
The background should be softly blurred while maintaining consistent lighting on the subject.

[Shot Entities]
Character: blonde woman (gender: female, features: bright red lipstick, gold stud earrings).

A close-up of a blonde woman with bright red lipstick and gold stud earrings,
looking attentively off-camera with a neutral expression.
```

**Prompt Debug 功能**：每个 shot 生成的最终 prompt 会保存到 `output/prompts/shot_XXX_prompt.txt`，方便调试和分析。

### 阶段四：视频生成

**文件**：`generator/ref2video.py`

#### 4.1 WanT2V（纯文本生成）

当 `generation_mode = "t2v"` 时调用，用于：
- 首个镜头（Registry 为空）
- 新角色首次出现

```python
def _phantom_t2v_generate(self, prompt, output_path):
    frames = self._t2v_pipeline.generate(
        input_prompt=prompt,
        size=(832, 480),
        frame_num=81,           # 81帧 ≈ 3.4秒 @24fps
        sampling_steps=50,
        guide_scale=7.5,        # 文本对齐强度
    )
```

#### 4.2 Phantom S2V（参考图+文本生成）

当 `generation_mode = "phantom"` 且有参考图时调用。

**关键特性**：Phantom 只从参考图提取**外观特征**（脸型、发色、服装），**姿态和动作由文本决定**，不会像 I2V 模型那样锁定第一帧构图。

```python
def _phantom_s2v_generate(self, prompt, references, output_path):
    # 参考图预处理：resize + 白边 padding → 832×480
    ref_imgs = [self._preprocess_ref_image(img, 832, 480)
                for img in references[:4]]  # 最多 4 张

    frames = self._pipeline.generate(
        input_prompt=prompt,
        ref_images=ref_imgs,
        guide_scale_img=5.0,    # 参考图外观强度
        guide_scale_text=7.5,   # 文本对齐强度
    )
```

**参考图预处理**（与 Phantom 官方一致）：
```
原图 (任意尺寸)
    │
    ▼ 保持宽高比 resize
    │
    ▼ 白色填充到 832×480
    │
    ▼ VAE encode → latent concat 到生成序列
```

#### 4.3 多卡并行配置

| 配置项 | 说明 |
|-------|------|
| `use_usp=true` | USP 序列并行（需要 FlashAttention）|
| `dit_fsdp=true` | DiT 模型参数分片到多卡 |
| `t5_fsdp=true` | T5 文本编码器参数分片 |

**8 卡 H800 + USP** 可达 **16 FPS**，比纯 FSDP 快 2-3 倍。

### 阶段五：Visual Grounding（生成后执行）

**文件**：`visual_grounding/grounder.py`

生成完视频后，对视频做"角色定位"，把高质量截图存入 Registry 供下一个镜头使用。

#### Step 1：抽帧

```python
def extract_frames(video_path, output_dir, fps=1.0):
    """按指定 fps 从视频中均匀采帧"""
    # 81帧视频 @24fps ≈ 3.4秒
    # fps=1.0 → 提取约 3-4 帧
```

#### Step 2：Grounding DINO 检测

开放词汇目标检测，根据实体的 `text_description` 定位边界框：

```python
boxes, logits, phrases = predict(
    model=self._gdino_model,
    image=image_tensor,
    caption="bearded man with sniper rifle",  # entity.text_description
    box_threshold=0.35,
    text_threshold=0.25,
)
# 返回：边界框坐标 + 置信度分数
```

#### Step 3：SAM2 精细分割

对检测到的边界框做像素级分割，提取干净的前景：

```python
def _extract_with_sam2(self, image_rgb, bbox):
    self._sam2_predictor.set_image(image_rgb)
    masks, scores, _ = self._sam2_predictor.predict(
        box=np.array([[x1, y1, x2, y2]]),
        multimask_output=False,
    )
    # 前景保留，背景填白
    result = image_rgb.copy()
    result[mask == 0] = 255
    return result[y1:y2, x1:x2]
```

**Location 类型特殊处理**：
```python
if entity_type == "location":
    # 检测所有前景 → SAM2 得到 union mask → cv2.inpaint 填充
    # 返回干净的背景帧作为场景参考
    bg_img = self._extract_background(image_source, image_tensor)
```

#### Step 4：Re-ID 质量评分

**文件**：`visual_grounding/reid.py`

对所有裁切图打分，过滤低质量截图：

```python
@dataclass
class QualityScore:
    sharpness: float       # Laplacian 方差，越高越清晰
    id_confidence: float   # InsightFace 人脸检测置信度
    frontal_score: float   # 正面程度（yaw 角接近 0 则高）
    final_score: float     # 加权综合分

# 综合分计算
final = 0.4 * sharpness + 0.4 * id_confidence + 0.2 * frontal_score
```

**评分维度详解**：

| 维度 | 计算方式 | 作用 |
|-----|---------|------|
| 清晰度 | `cv2.Laplacian(gray).var() / 500` | 过滤模糊帧 |
| 人脸置信度 | InsightFace `det_score` | 确保人脸可辨识 |
| 正面程度 | `1 - abs(yaw) / 90` | 优先选正脸 |
| bbox 面积 | 检测框占图像比例 | 过滤太小的远景人物 |

**过滤阈值**：`min_ref_quality=0.4`，低于此分数的截图不入库。

#### Step 4.5：跨实体 IoU 去重（多人场景）

**问题**：当一帧中有多人时，Grounding DINO 可能把 "young boy" 的检测框错误地定位到 "elderly man" 上（因为都是 person 类别）。

**解决方案**：对同一帧中所有实体的检测结果做 IoU 去重：

```python
def _cross_entity_dedup(all_ground_results, iou_threshold=0.5):
    """
    跨实体 IoU 去重：同一帧中，如果多个实体的检测框高度重叠，
    只保留置信度最高的那个实体的检测结果。
    """
    # 按帧分组所有检测结果
    # 对每帧：按置信度排序 → 高置信度的抑制重叠的低置信度检测
```

**效果**：
```
[Pipeline] IoU去重: char_young_boy 的检测被 char_elderly_man 抑制 (IoU=0.78, frame=frame_000024.jpg)
[Pipeline] 跨实体IoU去重: 共移除 3 个重叠检测
```

#### Step 5：入库

**文件**：`reference_manager/registry.py`

```python
@dataclass
class ReferenceEntry:
    entity_id: str        # "char_bearded_man"
    shot_id: int          # 来自哪个镜头
    crop_path: str        # 裁切图路径
    quality_score: float  # 综合质量分
    source: str           # "grounding" | "bootstrapped" | "manual"
```

```python
registry.register(entity_id, ReferenceEntry(
    entity_id="char_bearded_man",
    shot_id=1,
    crop_path="crops/shot_001/char_bearded_man_frame_000024_det00_crop.jpg",
    quality_score=0.72,
    source="grounding",
))
```

---

## 完整流程图

```
输入脚本 (YAML)
  │
  ├─ [可选] global_caption
  │     │
  │     └─► LLM 解析 (Shot 0)
  │           ├─ 提取所有实体，建立实体图谱
  │           └─ 例：char_bearded_man、char_masked_figure、loc_desert
  │
  ├─ Shot 1 描述文本
  │     │
  │     ├─► [1] LLM 解析
  │     │     ├─ 提取实体，查已知列表做共指消解
  │     │     └─ → {char_bearded_man: 首次出现, is_new=true}
  │     │
  │     ├─► [2] Registry 查询
  │     │     ├─ char_bearded_man 无参考图
  │     │     └─ → 路由：WanT2V（纯文本生成）
  │     │
  │     ├─► [3] Prompt 构建
  │     │     └─ gen_prompt = shot_context + shot.text
  │     │
  │     ├─► [4] WanT2V 生成 → shot_001.mp4
  │     │
  │     ├─► [4.5] Shot 1 验证（新增）
  │     │     ├─ 预期人数: 1（从实体解析）
  │     │     ├─ 检测实际人数: Person Detection + NMS
  │     │     ├─ 匹配 → 继续
  │     │     └─ 不匹配 → 换 seed 重试（最多 3 次）
  │     │
  │     └─► [5] Post-Grounding
  │           ├─ 抽帧 (1fps) → 3-4 帧
  │           ├─ Grounding DINO 检测 → 边界框
  │           ├─ SAM2 分割 → 白底前景图
  │           ├─ Re-ID 评分 → 过滤低质量
  │           └─ Registry 入库：char_bearded_man: [crop_A, crop_B]
  │
  ├─ Shot 2 描述文本
  │     │
  │     ├─► [1] LLM 解析
  │     │     └─ "the armed man" → char_bearded_man（共指消解）
  │     │
  │     ├─► [2] Registry 查询
  │     │     ├─ char_bearded_man 有 2 张参考图
  │     │     └─ → 路由：Phantom S2V
  │     │
  │     ├─► [3] 参考图预处理
  │     │     └─ resize + 白边 padding → 832×480
  │     │
  │     ├─► [4] Phantom S2V 生成
  │     │     ├─ 参考图提供外观
  │     │     ├─ 文本决定动作
  │     │     └─ → shot_002.mp4
  │     │
  │     └─► [5] Post-Grounding → 更新 Registry
  │
  └─ Shot N ... （同上，Registry 滚动积累）

输出：
  output/
  ├── videos/shot_001.mp4, shot_002.mp4, ...
  ├── prompts/shot_001_prompt.txt, ...    # 每个 shot 的最终生成 prompt（含 seed、镜头类型）
  ├── frames/shot_001/, shot_002/, ...      # 抽帧结果
  ├── crops/shot_001/, shot_002/, ...       # Grounding 裁切图
  ├── selected_refs/{entity_id}/            # Grounding 后入库的所有达标参考图（候选池）
  ├── used_refs/shot_001/, shot_002/, ...   # 每个 shot 实际输入模型的参考图
  ├── verify_frames/shot_001/, ...          # 【新增】Shot 1 验证用的采样帧
  ├── registry/entities.db                  # SQLite 参考图数据库
  └── pipeline_report.json                  # 详细运行报告（含验证历史）
```

**`selected_refs/` vs `used_refs/` 目录区别**：

| 目录 | 内容 | 用途 |
|------|------|------|
| `selected_refs/{entity_id}/` | 每个 shot grounding 后**所有达标的候选图** | 累积的参考池 |
| `used_refs/shot_XXX/` | 该 shot **实际输入模型的参考图** | 明确知道选了哪张 |

`used_refs/` 目录文件命名格式：`{entity_id}_{idx}.jpg`，一目了然。

---

## 为什么 Grounding 在生成之后做

一个容易误解的设计点：为什么不在生成前就从互联网或用户上传图里找参考图？

**原因一：风格一致性。** 如果 Shot 1 是模型生成的赛博朋克风画面，但参考图是真实照片，两者风格完全不同，Phantom 同时接收风格差异极大的输入会产生不自然的融合。从自己生成的视频里取截图，风格天然一致。

**原因二：没有外部依赖。** 不需要用户提前准备角色图，输入只有文字脚本，完全自动化。

**原因三：真实外观。** LLM 解析出的文字描述是 "bearded man"，这个描述够模糊、够抽象。第 1 个镜头生成后，胡子的颜色、长度、脸型就被具体化了，后续镜头参照具体图像生成，细节一致性比纯文字描述传递要好得多。

---

## 系统组件

| 组件 | 文件 | 职责 |
|------|------|------|
| LLM 实体解析器 | `entity_parser/parser.py` | 调用 Claude 从文本提取实体，跨镜头共指消解，全局语义提取，**实体数量提取**，**Close-up 光线分析** |
| 参考图库 | `reference_manager/registry.py` | SQLite 数据库，存储每个实体的参考截图路径和质量分 |
| 视觉 Grounding | `visual_grounding/grounder.py` | Grounding DINO 检测 + SAM2 分割，从生成视频中定位并裁切实体 |
| 质量评分 | `visual_grounding/reid.py` | 对裁切图打分（清晰度、人脸置信度、正面程度、bbox 面积），过滤低质量截图 |
| **Shot 1 验证器** | `verification/entity_count_verifier.py` | **【新增】** T2V 生成后验证人数是否匹配预期，支持自动重试 |
| 视频生成器 | `generator/ref2video.py` | 封装 WanT2V 和 Phantom S2V，含参考图预处理、T2V/S2V 动态切换、VRAM 管理、多卡支持 |
| 主 Pipeline | `orchestrator/pipeline.py` | 串联以上所有组件，**四层 Prompt 构建**，prompt 保存，多卡分布式逻辑，**验证循环**，**Agentic 光线决策** |
| LLM 客户端 | `utils/llm_client.py` | 封装公司内部 OpenAI 兼容 API，含模型别名映射和限流重试 |

---

## 目录结构

```
T2V_Grounding/
├── README.md                        # 本文档
├── configs/
│   ├── config.yaml                  # 全局配置（模型路径、API key、多卡参数等）
│   ├── demo_script.yaml             # 测试脚本01：单角色 4 镜头
│   ├── test_aba_scene.yaml          # 测试脚本02：A→B→A 场景回归
│   ├── test_sniper_standoff.yaml    # 测试脚本03：global_caption + 5 镜头
│   ├── test_dual_character.yaml     # 测试脚本04：双角色 5 镜头
│   └── test_scene_consistency.yaml  # 测试脚本05：场景一致性
├── phase1_poc/
│   ├── run_demo.py                  # 主入口（单卡/多卡通用）
│   ├── entity_parser/
│   │   └── parser.py                # LLM 实体提取 + 共指消解 + 全局语义提取 + 实体数量提取 + 光线分析
│   ├── visual_grounding/
│   │   ├── grounder.py              # Grounding DINO + SAM2
│   │   └── reid.py                  # Re-ID 质量打分
│   ├── verification/                # 【新增】Shot 1 验证模块
│   │   └── entity_count_verifier.py # 实体数量验证 + 重试逻辑
│   ├── reference_manager/
│   │   └── registry.py              # Entity Registry（SQLite）
│   ├── generator/
│   │   └── ref2video.py             # WanT2V / Phantom S2V 封装
│   ├── orchestrator/
│   │   └── pipeline.py              # 主 Pipeline + 四层 Prompt 构建 + 验证循环 + Agentic 光线决策
│   ├── utils/
│   │   └── llm_client.py            # LLM API 封装
│   └── weights/                     # 模型权重
│       ├── groundingdino_swinb_cogcoor.pth
│       ├── GroundingDINO_SwinB_cfg.py
│       ├── bert-based-uncased/
│       └── sam2_hiera_large.pt
├── phase2_system/
│   └── agent_orchestrator.py        # Agentic Loop（Phase 2，开发中）
└── evaluation/
    ├── metrics.py                   # CLIP-I / FaceID 评测指标
    └── eval_pipeline.py             # 自动化评测入口
```

---

## 快速开始

### Step 1：环境配置

```bash
# 基础依赖
pip install openai pyyaml opencv-python pillow easydict ftfy imageio imageio-ffmpeg

# 视觉模型
pip install groundingdino-py
pip install git+https://github.com/facebookresearch/sam2.git
pip install insightface onnxruntime-gpu

# FlashAttention（USP 序列并行需要）
# Ubuntu 18.04 需要用预编译 wheel：
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

### Step 2：Mock 模式验证

```bash
cd phase1_poc

python run_demo.py \
  --script ../configs/test_sniper_standoff.yaml \
  --mock \
  --output ./output_mock
```

检查实体解析是否正确：
```bash
python -c "
import json
with open('./output_mock/pipeline_report.json') as f:
    r = json.load(f)
for s in r['shots']:
    print(f'Shot {s[\"shot_id\"]}: mode={s[\"generation_mode\"]:8s} entities={s[\"entity_ids\"]}')
"
```

### Step 3：真实生成（8 卡）

```bash
cd phase1_poc

torchrun --nproc_per_node=8 run_demo.py \
  --script ../configs/test_sniper_standoff.yaml \
  --config ../configs/config.yaml \
  --backend phantom \
  --output ./output_sniper
```

检查路由是否正确：
```bash
python -c "
import json
with open('./output_sniper/pipeline_report.json') as f:
    r = json.load(f)
for s in r['flow_summary']:
    print(f'Shot {s[\"shot_id\"]}: mode={s[\"mode\"]:8s} refs_used={s[\"refs_used\"]} grounded={s[\"crops_grounded\"]}')
"
```

期望输出：
```
Shot 1: mode=t2v      refs_used=0  grounded=6   ← 首镜头无参考，生成后入库
Shot 2: mode=phantom  refs_used=1  grounded=4   ← 用 Shot 1 的截图
Shot 3: mode=phantom  refs_used=2  grounded=5
Shot 4: mode=phantom  refs_used=2  grounded=4
Shot 5: mode=phantom  refs_used=2  grounded=6
```

---

## 配置说明

`configs/config.yaml` 关键参数：

```yaml
llm:
  model: "claude-haiku-4-5"        # 实体解析 LLM（haiku 成本低）

generator:
  num_inference_steps: 30          # 推理步数（30 步质量接近 50 步，快 40%）
  guide_scale_text: 7.5            # 文本对齐强度
  guide_scale_img:  5.0            # 参考图外观强度
                                   # 3.0 = 弱约束，动作更自然
                                   # 5.0 = 平衡
                                   # 7.0 = 强约束，外观更一致但动作偏僵
  use_usp:   true                  # USP 序列并行（需要 FlashAttention）
  dit_fsdp:  true                  # DiT 模型参数分片
  t5_fsdp:   true                  # T5 参数分片

grounding:
  box_threshold: 0.35              # DINO 检测阈值，越低框越多

reid:
  min_quality_threshold: 0.4       # 参考图最低质量分
```

---

## Debug 功能

### Prompt 保存

每个 shot 生成前，Pipeline 会将最终发送给模型的 prompt 保存到文件：

```
output/prompts/shot_001_prompt.txt
output/prompts/shot_002_prompt.txt
...
```

文件内容示例：
```
=== Shot 1 Generation Prompt ===
Mode: t2v
Seed: 43 (base=42)
Shot type: medium
Reference images: 0

==================================================
FINAL PROMPT TO MODEL:
==================================================

[Global Context]
Visual style: cinematic, dramatic lighting.
...

[Shot Entities]
Character: bearded man (gender: male, age: 40s).
...

A close-up shot shows a bearded man intently looking through...
```

### 实际使用的参考图记录

每个 shot 实际输入模型的参考图会保存到 `used_refs/` 目录：

```
output/used_refs/
├── shot_001/           # Shot 1（通常无参考图，T2V模式）
├── shot_002/
│   ├── char_elderly_man_00.jpg
│   └── loc_living_room_00.jpg
├── shot_003/
│   ├── char_elderly_man_00.jpg    # 锚点：来自 shot 1
│   ├── char_elderly_woman_00.jpg  # 锚点：来自 shot 1
│   └── loc_living_room_00.jpg
...
```

### Pipeline Report

运行完成后，`pipeline_report.json` 包含完整的运行信息：

```json
{
  "total_shots": 5,
  "registry_stats": {...},
  "flow_summary": [...],
  "shots": [
    {
      "shot_id": 1,
      "generation_mode": "t2v",
      "entity_ids": ["char_bearded_sniper", "obj_sniper_rifle"],
      "reference_used": {},
      "grounded_entities": {...},
      "metadata": {
        "gen_prompt": "完整的生成 prompt..."
      }
    }
  ]
}
```

---

## 常见问题

**Q：`AssertionError: FlashAttention is not available`**

A：需要安装 FlashAttention。Ubuntu 18.04 使用预编译 wheel：
```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```
或者禁用 USP（`use_usp: false`），使用纯 FSDP。

**Q：Shot 2 路由走了 t2v 而不是 phantom**

A：Shot 1 的 Grounding 没有找到角色。检查 `pipeline_report.json` 中 Shot 1 的 `crops_grounded`：
- 若为 0，调低 `box_threshold` 到 0.25
- 检查 Grounding DINO 权重路径是否正确

**Q：同一角色被识别成不同 ID（如 `char_alex` 和 `char_man`）**

A：LLM 共指消解失败。解决：
1. 添加 `global_caption` 统一描述
2. 每个 shot 保持角色描述一致，避免只用代词

**Q：CUDA OOM**

A：确认多卡配置正确：
- `dit_fsdp: true` 和 `t5_fsdp: true` 必须开启
- 使用 `torchrun --nproc_per_node=8` 启动

---

## 测试脚本说明

| 脚本 | 镜头数 | 有无 global_caption | 测试重点 |
|------|--------|---------------------|---------|
| `demo_script.yaml` | 4 | 无 | 基础功能，单角色 |
| `test_aba_scene.yaml` | 3 | 无 | 场景回归：书房→仓库→书房 |
| `test_sniper_standoff.yaml` | 5 | **有** | 代词共指消解 |
| `test_dual_character.yaml` | 5 | 无 | 双角色互不干扰 |
| `test_office_conversation.yaml` | 6 | **有** | 双角色对话、close-up/wide 切换、**Agentic 光线决策测试** |
| `test_scene_consistency.yaml` | 4 | 无 | 场景一致性 |
