# T2V Grounding — 多镜头一致性视频生成系统

## 这个项目解决什么问题

当前主流的文本生成视频（T2V）模型（如 Wan2.1、Sora）存在一个根本性缺陷：**每次生成都是独立的**。如果你用同样的文字描述 "一个穿蓝色夹克的男人" 生成 5 个视频片段，每个片段里这个男人的长相、发型、夹克颜色都会不同。这在电影/短剧等需要跨镜头角色一致性的场景下完全不可用。

**本项目的目标：** 给定一段多镜头的分镜脚本（如 5 个镜头描述同一个故事），让系统自动生成每个镜头的视频，并保证同一个角色在所有镜头中外观一致。

**解决思路：** 生成第 1 个镜头后，从视频中"认出"角色，截取角色的图像；生成第 2 个镜头时，把这张图传给模型说"这个角色就长这样"，让模型参照外观生成——如此滚动传递，实现跨镜头一致性。

---

## 核心流程详解

整个 pipeline 按镜头顺序执行，每个镜头分为"生成"和"入库"两个阶段。

### 阶段一：实体解析（每个镜头执行一次）

用 LLM（Claude）读取当前镜头的文字描述，从中识别出所有需要跨镜头保持一致的"实体"：

- **人物**（character）：如"留胡子的男人"、"戴白色面具的人"
- **重要物体**（object）：如"狙击步枪"、"皮质日记本"
- **场景**（location）：如"沙漠废墟"、"私人书房"

每个实体被分配一个唯一 ID（如 `char_bearded_man`），并记录外观属性。这些 ID 在所有镜头中保持一致，是后续"认人"的关键。

> **跨镜头共指消解**：Shot 1 写了"一个留胡子的男人拿着狙击枪"，Shot 2 写了"这名男子向前跑去"——LLM 会把"这名男子"识别为同一个 `char_bearded_man`，而不是创建新实体。

### 阶段二：路由决策（决定用哪个模型生成）

查询"参考图库"（Entity Registry）——一个 SQLite 数据库，存储着历史镜头中每个实体的截图：

- **库里没有这个角色的图** → 用 **WanT2V**（纯文字→视频）生成。这发生在首个镜头，或角色第一次出现时。
- **库里已有这个角色的图** → 用 **Phantom S2V**（图+文字→视频）生成。把截图作为外观参考传给模型。

> **为什么不用图转视频（I2V）？** I2V 模型把参考图当成视频第 0 帧，会直接锁死构图、背景、姿态——角色只能静止不动。Phantom S2V 不同，它只从参考图中提取外观特征（脸型、发色、服装），姿态和动作完全由文字描述决定，更灵活。

### 阶段三：视频生成

把组装好的 prompt（= 全局背景描述 + 当前镜头描述）和参考图送给模型，生成当前镜头的视频（默认 81 帧，约 3.4 秒，分辨率 832×480）。

多卡环境下（4 张 A100），所有 GPU 共同参与推理（Phantom 14B 模型的参数分布在 4 张卡上），速度约为单卡的 3 倍。

### 阶段四：视觉 Grounding（生成之后才做）

生成完视频后，对这个视频做"角色定位"，把截图存入库中供下一个镜头使用：

**Step 1 — 抽帧**：按 1fps 从视频中提取若干帧（如 4 帧）

**Step 2 — Grounding DINO 检测**：用开放词汇目标检测模型，根据实体的文字描述（如 "bearded man with sniper rifle"）在每一帧中画出边界框，定位角色位置

**Step 3 — SAM2 分割**：对检测到的边界框做精细的像素级分割，裁切出干净的角色图像

**Step 4 — Re-ID 质量评分**：对所有裁切图打分，过滤掉质量差的：
- **清晰度**：Laplacian 方差（模糊的图扣分）
- **人脸置信度**：InsightFace 检测人脸（人物类实体，找不到人脸则得 0 分）
- **bbox 面积**：如果裁切框太小（短边 < 80px），说明检测到的是远景小人，pad 到 832×480 后主体面积不足 1%，传给模型没有意义，给低分过滤掉

**Step 5 — 参考图预处理**：将筛选出的高质量截图 resize+白边 padding 到 832×480，保证所有参考图经 VAE 编码后维度一致（否则多张图 `torch.cat` 时会报维度不匹配）

**Step 6 — 入库**：写入 Entity Registry，供下一个镜头的 Phantom S2V 使用

### 完整流程图

```
输入脚本 (YAML)
  │
  ├─ [可选] global_caption
  │     │
  │     └─► LLM 解析 (Shot 0)
  │           提取所有实体，写入已知实体列表
  │           例：char_bearded_man、char_masked_figure、loc_desert
  │
  ├─ Shot 1 描述文本
  │     │
  │     ├─► LLM 解析：提取实体，查已知列表做共指消解
  │     │     → {char_bearded_man: 首次出现}
  │     │
  │     ├─► Registry 查询：char_bearded_man 无参考图
  │     │     → 路由：WanT2V（纯文本生成）
  │     │
  │     ├─► WanT2V 生成 → shot_001.mp4
  │     │
  │     └─► Post-Grounding（生成后）
  │           抽帧 → DINO 检测 → SAM2 分割 → 质量评分 → 预处理
  │           → Registry 入库：char_bearded_man: [crop_A, crop_B, crop_C]
  │
  ├─ Shot 2 描述文本
  │     │
  │     ├─► LLM 解析："the armed man" → char_bearded_man（共指消解成功）
  │     │
  │     ├─► Registry 查询：char_bearded_man 有 3 张参考图
  │     │     → 路由：Phantom S2V（参考图 + 文本生成）
  │     │
  │     ├─► 参考图预处理：resize+pad → 832×480
  │     │
  │     ├─► Phantom S2V 生成（参考图提供外观，文本决定动作）→ shot_002.mp4
  │     │
  │     └─► Post-Grounding → 更新 Registry
  │
  └─ Shot N ... （同上，Registry 滚动积累）

输出：output/videos/shot_001.mp4, shot_002.mp4, ..., pipeline_report.json
```

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
| LLM 实体解析器 | `entity_parser/parser.py` | 调用 Claude 从文本提取实体，跨镜头共指消解，支持 global_caption 预解析 |
| 参考图库 | `reference_manager/registry.py` | SQLite 数据库，存储每个实体的参考截图路径和质量分 |
| 视觉 Grounding | `visual_grounding/grounder.py` | Grounding DINO 检测 + SAM2 分割，从生成视频中定位并裁切实体 |
| 质量评分 | `visual_grounding/reid.py` | 对裁切图打分（清晰度、人脸置信度、bbox 面积），过滤低质量截图 |
| 视频生成器 | `generator/ref2video.py` | 封装 WanT2V 和 Phantom S2V，含参考图预处理和多卡 rank 守卫 |
| 主 Pipeline | `orchestrator/pipeline.py` | 串联以上所有组件，管理多卡分布式逻辑（rank 0 负责解析和入库，所有 rank 参与推理）|
| LLM 客户端 | `utils/llm_client.py` | 封装公司内部 OpenAI 兼容 API，含模型别名映射和限流重试 |

---

## 目录结构

```
T2V_Grounding/
├── README.md                        # 本文档
├── ROADMAP.md                       # 三阶段路线图
├── configs/
│   ├── config.yaml                  # 全局配置（模型路径、API key、多卡参数等）
│   ├── demo_script.yaml             # 测试脚本01：单角色 4 镜头
│   ├── test_aba_scene.yaml          # 测试脚本02：A→B→A 场景回归（3 镜头）
│   ├── test_sniper_standoff.yaml    # 测试脚本03：global_caption + 5 镜头
│   └── test_dual_character.yaml     # 测试脚本04：双角色 5 镜头
├── scripts/
│   └── run_multi_seed.sh            # 多 seed 批量生成脚本
├── phase1_poc/
│   ├── run_demo.py                  # 主入口（单卡/多卡通用）
│   ├── entity_parser/
│   │   └── parser.py                # LLM 实体提取 + 跨镜头共指消解
│   ├── visual_grounding/
│   │   ├── grounder.py              # Grounding DINO + SAM2
│   │   └── reid.py                  # Re-ID 质量打分（含 bbox 面积惩罚）
│   ├── reference_manager/
│   │   └── registry.py              # Entity Registry（SQLite）
│   ├── generator/
│   │   └── ref2video.py             # WanT2V / Phantom S2V 封装 + 路由
│   ├── orchestrator/
│   │   └── pipeline.py              # 主 Pipeline（支持多卡分布式）
│   ├── utils/
│   │   └── llm_client.py            # 公司内部 API 统一封装
│   └── weights/                     # Grounding DINO / SAM2 / BERT 权重
│       ├── groundingdino_swinb_cogcoor.pth
│       ├── GroundingDINO_SwinB_cfg.py
│       ├── bert-based-uncased/      # Grounding DINO 文本编码器
│       └── sam2_hiera_large.pt
├── phase2_system/
│   └── agent_orchestrator.py        # Agentic Loop（Phase 2，开发中）
└── evaluation/
    ├── metrics.py                   # CLIP-I / FaceID 评测指标
    └── eval_pipeline.py             # 自动化评测入口
```

---

## 输入脚本格式

### 基础格式

每个 shot 写一段文字描述这个镜头发生了什么。**第 1 个镜头应完整描述角色外观**（发色、服装等），后续镜头可以用简短指代。

```yaml
shots:
  - shot_id: 1
    text: >
      Alex Chen, a male detective in his early 30s with short black hair
      and a dark navy trench coat, walks into a brightly lit police station.
      He carries a worn leather briefcase.
    duration_seconds: 3.0

  - shot_id: 2
    text: >
      Alex sits at a cluttered desk, studying a photograph under a desk lamp.
      His trench coat hangs on a rack behind him.
    duration_seconds: 3.0

  - shot_id: 3
    text: >
      Alex walks quickly through a busy outdoor market, weaving through the crowd.
    duration_seconds: 3.0
```

### 带 global_caption 格式（推荐用于复杂场景）

当脚本里有多个角色、或 shot 描述使用了大量代词（"the man"、"he"、"they"），建议在文件开头加 `global_caption`。

```yaml
global_caption: >
  The video depicts a tense scene in a sandy, desolate environment where a
  bearded man armed with a sniper rifle is engaged in a standoff. He is seen
  aiming his weapon, reacting to his surroundings, and pursuing a mysterious
  masked figure dressed in black with a white face mask.

shots:
  - shot_id: 1
    text: >
      A close-up shot shows a bearded man intently looking through the scope
      of a large sniper rifle, with a distorted "STOP" sign and sandy background.
    duration_seconds: 3.0

  - shot_id: 2
    text: >
      The camera pulls back. The man stands up, holding his rifle, while two
      bodies lie on the ground behind him amidst sandbags.
    duration_seconds: 3.0

  - shot_id: 3
    text: >
      The armed man runs forward, following the masked figure between sandbags.
    duration_seconds: 3.0
```

**global_caption 做了三件事：**

1. **Shot 0 预解析**：Pipeline 正式处理镜头之前，先把 global_caption 单独送给 LLM 解析一次。LLM 从中提取所有出现的角色、物体、场景，为每个实体分配唯一 ID（`char_bearded_man`、`char_masked_figure`、`loc_desert`）并记录外观属性（胡子、服装、持有物等）。这份清单写入已知实体列表，供后续每个 shot 查用。

2. **共指消解**：Shot 2 写的是 "The man stands up"——没有任何外观描述。解析 Shot 2 时，LLM 同时看到 global_caption，能把 "The man" 与已知实体 `char_bearded_man` 对应起来，而不是错误地创建一个新实体 `char_man`。

3. **生成 prompt 增强**：发给视频生成模型的 prompt 是 `global_caption + shot_text`，模型在生成 Shot 2 时同时看到"这是个拿狙击枪的留胡子男人"和"他站起来向侧面看"，外观描述和动作描述都有，生成质量更稳定。

---

## 环境与权重

### 依赖安装

```bash
# 基础依赖
pip install openai pyyaml opencv-python pillow easydict ftfy imageio imageio-ffmpeg

# 视觉模型
pip install groundingdino-py
pip install git+https://github.com/facebookresearch/sam2.git
pip install insightface onnxruntime-gpu

# 视频生成
pip install diffusers transformers accelerate

# 注意：flash_attn 在 glibc 2.27 环境下无法安装
# Phantom 内置 fallback，直接跳过：pip uninstall flash_attn -y
```

### 权重文件

| 模型 | 路径 | 用途 |
|------|------|------|
| Grounding DINO | `phase1_poc/weights/groundingdino_swinb_cogcoor.pth` | 从视频帧中检测实体位置 |
| BERT | `phase1_poc/weights/bert-based-uncased/` | Grounding DINO 的文本编码器 |
| SAM2 | `phase1_poc/weights/sam2_hiera_large.pt` | 精细分割角色轮廓 |
| Wan2.1-T2V-14B | 配置在 `config.yaml` | 首镜头纯文本生成 |
| Phantom-Wan-14B | 配置在 `config.yaml` | 后续镜头参考图外观 conditioning |

### LLM API

```bash
# 查询当前可用模型
curl -s -H "Authorization: Bearer $API_KEY" \
  http://yy.dbh.baidu-int.com/v1/models | python3 -c "
import json,sys; data=json.load(sys.stdin)
[print(m['id']) for m in data.get('data',[]) if 'claude' in m['id']]
"
```

当前可用 Claude 模型（2026-04）：

| 模型 ID | 说明 |
|---------|------|
| `claude-sonnet-4-6-Anthropic` | **默认推荐** |
| `claude-sonnet-4-6-chuangzuo` | 备用渠道 |
| `claude-opus-4-6` | 更强推理能力，速度较慢 |

---

## 快速开始

### Step 1：验证 LLM 连通性

```bash
cd phase1_poc
python utils/llm_client.py
# 输出 "使用模型: claude-sonnet-4-6-Anthropic" 并有正常回复则通过
```

### Step 2：Mock 模式（无 GPU，验证 Pipeline 逻辑）

Mock 模式跳过视频生成（输出彩色占位视频），但完整执行 LLM 解析、路由决策、报告生成，用来快速验证脚本解析是否正确。

```bash
cd phase1_poc

python run_demo.py \
  --script ../configs/demo_script.yaml \
  --mock \
  --output ./output_mock
```

**检查实体解析：同一角色是否跨镜头 ID 一致**

```bash
python -c "
import json
with open('./output_mock/pipeline_report.json') as f:
    r = json.load(f)
for s in r['shots']:
    print(f'Shot {s[\"shot_id\"]}: mode={s[\"generation_mode\"]:8s}  entities={s[\"entity_ids\"]}')
"
```

期望输出（`char_alex` 在全部镜头中 ID 相同）：
```
Shot 1: mode=mock      entities=['char_alex', 'obj_briefcase', 'loc_police_station']
Shot 2: mode=mock      entities=['char_alex', 'loc_office', 'obj_desk_lamp']
Shot 3: mode=mock      entities=['char_alex', 'loc_outdoor_market']
Shot 4: mode=mock      entities=['char_alex', 'loc_street_night']
```

### Step 3：4 卡真实生成

```bash
cd phase1_poc

# A→B→A 场景回归测试（3 镜头，约 30 分钟）
torchrun --nproc_per_node=4 run_demo.py \
  --script ../configs/test_aba_scene.yaml \
  --config ../configs/config.yaml \
  --backend phantom \
  --output ./output_aba

# 带 global_caption 的 5 镜头测试（约 50 分钟）
torchrun --nproc_per_node=4 run_demo.py \
  --script ../configs/test_sniper_standoff.yaml \
  --config ../configs/config.yaml \
  --backend phantom \
  --output ./output_sniper
```

**检查路由是否正确（Shot 1 应为 t2v，后续应为 phantom）：**

```bash
python -c "
import json
with open('./output_aba/pipeline_report.json') as f:
    r = json.load(f)
for s in r['flow_summary']:
    print(f'Shot {s[\"shot_id\"]}: mode={s[\"mode\"]:8s}  refs_used={s[\"refs_used\"]}  grounded={s[\"crops_grounded\"]}')
"
```

期望输出：
```
Shot 1: mode=t2v      refs_used=0  grounded=6   ← 首镜头无参考，生成后入库 6 张截图
Shot 2: mode=phantom  refs_used=1  grounded=4   ← 用 Shot 1 的截图作参考
Shot 3: mode=phantom  refs_used=2  grounded=6   ← 用 Shot 1+2 的截图作参考
```

### Step 4：多 seed 批量生成（对比多组结果）

```bash
# 用法: run_multi_seed.sh [运行次数] [输出目录] [脚本文件名]

# 跑 5 次（seed=42/1337/2025/314159/777），输出到 output_sniper_multi/
bash scripts/run_multi_seed.sh 5 ./output_sniper_multi test_sniper_standoff.yaml
```

输出结构：
```
output_sniper_multi/
├── run_001_seed42/       ← videos/, crops/, registry/, pipeline_report.json
├── run_002_seed1337/
├── run_003_seed2025/
├── run_004_seed314159/
└── run_005_seed777/
```

---

## 配置说明

`configs/config.yaml` 关键参数：

```yaml
llm:
  model: "claude-sonnet-4-6-Anthropic"   # 实体解析用的 LLM

generator:
  num_inference_steps: 10    # 调试用 10 步（快），正式生成用 50 步（质量好）
  guide_scale_text: 7.5      # 文本 prompt 的影响强度，越大越贴近文字描述
  guide_scale_img:  5.0      # 参考图外观的影响强度（核心调参项）
                             # 3.0 = 弱约束，角色外观可能漂移但动作更自然
                             # 5.0 = 默认平衡值
                             # 7.0 = 强约束，外观一致但动作可能偏僵硬
  dit_fsdp: true             # 14B 模型参数分片到多卡，4 卡运行必须开启
  t5_fsdp:  true             # 同上，T5 文本编码器也分片

grounding:
  box_threshold: 0.35        # Grounding DINO 检测阈值，越低检测到的框越多
                             # 如果 grounded=0，尝试调低到 0.25
```

---

## 常见问题

**Q：运行时报 `AssertionError: pipeline model parallel group is not initialized`**

`config.yaml` 中 `use_usp` 必须为 `false`。`use_usp: true` 需要 xfuser 初始化额外的并行组，目前未启用。

**Q：CUDA out of memory（加载 Phantom S2V 时报 OOM）**

- 确认 `dit_fsdp: true` 和 `t5_fsdp: true` 已开启（14B 模型必须 4 卡分片，否则每卡需要 56GB）
- 用 `nvidia-smi` 检查是否有其他进程占用显存
- 确认使用 `torchrun --nproc_per_node=4` 启动，而不是直接 `python`

**Q：Shot 2 开始视频风格变成动画/插画风**

原因：从 Shot 1 的视频中裁切到的角色截图太小（角色在画面中只占很小一部分），截图 pad 成 832×480 后角色主体面积不足 1%，Phantom 无法从中提取有效外观信息，就退化为"凭感觉生成"，而模型本身有插画风格偏好。

解法：调低 `grounding.box_threshold`（如 0.25），让 DINO 在更宽松的阈值下找到更大的检测框。

**Q：Shot 2 路由走了 t2v 而不是 phantom**

说明 Shot 1 的 Grounding 没有找到任何角色。检查 `pipeline_report.json` 中 Shot 1 的 `crops_grounded`，如果为 0，原因通常是：
1. Grounding DINO 权重路径配置错误
2. `box_threshold` 太高，降低到 0.25 试试

**Q：同一个角色在不同镜头被识别成两个不同 ID（如 `char_alex` 和 `char_man`）**

LLM 共指消解失败。解法：
1. 在 YAML 里加 `global_caption`，统一角色描述，LLM 在全局上下文下消解更准确
2. 或者在每个 shot 的文字描述中保持角色描述一致，不要只写代词

**Q：API 报 `model_not_found` 503 错误**

模型 ID 写错了。以 `http://yy.dbh.baidu-int.com/v1/models` 接口返回的列表为准，当前可用的是 `claude-sonnet-4-6-Anthropic` 而不是 `claude-sonnet-4-6`。

---

## 测试脚本说明

| 脚本 | 镜头数 | 有无 global_caption | 测试重点 |
|------|--------|---------------------|---------|
| `demo_script.yaml` | 4 | 无 | 基础功能验证，单角色跨室内/户外/雨夜 |
| `test_aba_scene.yaml` | 3 | 无 | **场景回归**：书房 → 仓库 → 回到书房，验证场景和角色同时保持一致 |
| `test_sniper_standoff.yaml` | 5 | **有** | global_caption + 代词共指消解，验证 Shot 2-5 的 "the man" 都能正确识别 |
| `test_dual_character.yaml` | 5 | 无 | 双角色，验证两个角色分别入库、互不干扰 |

**`test_aba_scene.yaml` 预期执行过程：**
```
Shot 1 → WanT2V（无参考，首镜头）
         生成书房场景，Marcus 坐在木质大桌前
         → Grounding 入库：char_marcus（人物截图）、loc_study_room（背景截图）

Shot 2 → Phantom S2V（char_marcus 有参考图）
         换到仓库场景，外观参考 Shot 1 的 Marcus 截图
         → Grounding 入库：char_marcus 更新，loc_warehouse 新增

Shot 3 → Phantom S2V（char_marcus + loc_study_room 都有参考图）
         回到书房，同时参考人物截图和书房背景截图
         → 验证：Marcus 外观与 Shot 1 一致，书房视觉风格与 Shot 1 接近
```
