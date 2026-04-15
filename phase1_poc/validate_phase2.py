"""
阶段二验收脚本
测试流程: 采帧 → Grounding DINO 检测 → SAM2 分割 → Re-ID 质量评分 → 输出报告
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))

VIDEO_PATH  = "/root/paddlejob/workspace/env_run/output/lyx/T2V_Grounding/data/test/video.mp4"
OUTPUT_DIR  = "/root/paddlejob/workspace/env_run/output/lyx/T2V_Grounding/data/test/phase2_output"
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")

# 测试的实体（文本描述 → entity_id），根据视频内容可改
ENTITIES = [
    ("person", "char_main"),
]

# ─── Step 1: 采帧 ────────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: 从视频采帧 (2fps)")
print("=" * 60)
from visual_grounding.grounder import extract_frames, VideoGrounder

frames_dir = os.path.join(OUTPUT_DIR, "frames")
t0 = time.time()
frames = extract_frames(VIDEO_PATH, frames_dir, fps=2.0)
print(f"  → 采样帧数: {len(frames)}  耗时: {time.time()-t0:.1f}s")
assert len(frames) > 0, "采帧失败，请检查视频路径"

# ─── Step 2: Grounding DINO + SAM2 ──────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: Grounding DINO + SAM2 检测分割")
print("=" * 60)

gdino_cfg = os.path.join(os.path.dirname(__file__),
    "visual_grounding/../weights/GroundingDINO_SwinB_cfg.py")
# 使用本地修改过的 config（text_encoder_type 指向本地 bert 路径）
gdino_cfg = os.path.join(WEIGHTS_DIR, "GroundingDINO_SwinB_cfg.py")

os.environ["TRANSFORMERS_OFFLINE"] = "1"   # 禁止联网，强制用本地缓存

grounder = VideoGrounder(
    gdino_config=gdino_cfg,
    gdino_weights=os.path.join(WEIGHTS_DIR, "groundingdino_swinb_cogcoor.pth"),
    sam2_config="sam2_hiera_l.yaml",
    sam2_weights=os.path.join(WEIGHTS_DIR, "sam2_hiera_large.pt"),
    device="cuda",
)

all_grounding_results = {}
for entity_text, entity_id in ENTITIES:
    print(f"\n  检测实体: '{entity_text}' (id={entity_id})")
    crops_dir = os.path.join(OUTPUT_DIR, "crops", entity_id)
    t0 = time.time()
    results = grounder.ground_in_frames(frames, entity_text, entity_id, crops_dir, max_results=5)
    elapsed = time.time() - t0
    print(f"  → 检测到 {len(results)} 个结果  耗时: {elapsed:.1f}s")
    for r in results:
        print(f"     score={r.score:.3f}  bbox={r.bbox}  crop={os.path.basename(r.crop_path)}")
    all_grounding_results[entity_id] = results

# ─── Step 3: Re-ID 质量评分 ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: Re-ID 质量评分")
print("=" * 60)

from visual_grounding.reid import ReferenceQualityScorer
scorer = ReferenceQualityScorer(device="cuda")

all_scores = {}
for entity_text, entity_id in ENTITIES:
    results = all_grounding_results.get(entity_id, [])
    if not results:
        print(f"  [{entity_id}] 无裁切图，跳过")
        continue
    crop_paths = [r.crop_path for r in results]
    ranked = scorer.rank_references(crop_paths, entity_type="character")
    all_scores[entity_id] = ranked
    print(f"\n  [{entity_id}] 质量排序:")
    for i, s in enumerate(ranked):
        print(f"    #{i+1} final={s.final_score:.3f}  sharp={s.sharpness:.3f}  "
              f"id_conf={s.id_confidence:.3f}  frontal={s.frontal_score:.3f}")
        print(f"         {os.path.basename(s.crop_path)}")

# ─── Step 4: 生成验收报告 ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: 验收报告")
print("=" * 60)

report = {
    "video": VIDEO_PATH,
    "frames_extracted": len(frames),
    "entities": {}
}

PASS = True
for entity_text, entity_id in ENTITIES:
    grounding_ok = len(all_grounding_results.get(entity_id, [])) > 0
    best_score = all_scores[entity_id][0].final_score if entity_id in all_scores and all_scores[entity_id] else 0.0
    best_crop  = all_scores[entity_id][0].crop_path   if entity_id in all_scores and all_scores[entity_id] else None
    status = "PASS" if grounding_ok and best_score >= 0.4 else "FAIL"
    if status == "FAIL":
        PASS = False
    report["entities"][entity_id] = {
        "entity_text": entity_text,
        "grounding_hits": len(all_grounding_results.get(entity_id, [])),
        "best_quality_score": round(best_score, 3),
        "best_crop": best_crop,
        "status": status,
    }
    print(f"  [{entity_id}]  grounding_hits={report['entities'][entity_id]['grounding_hits']}  "
          f"best_score={best_score:.3f}  → {status}")

report["overall"] = "PASS" if PASS else "FAIL"
os.makedirs(OUTPUT_DIR, exist_ok=True)
report_path = os.path.join(OUTPUT_DIR, "phase2_report.json")
import numpy as np
class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.floating, np.integer)): return o.item()
        return super().default(o)
with open(report_path, "w") as f:
    json.dump(report, f, ensure_ascii=False, indent=2, cls=_Encoder)

print(f"\n报告已保存: {report_path}")
print(f"\n{'✅ 阶段二验收通过！' if PASS else '❌ 阶段二验收未通过，请检查报告'}")
