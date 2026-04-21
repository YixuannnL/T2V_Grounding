#!/usr/bin/env python3
"""
run_demo.py — T2V Grounding Pipeline 一键运行入口

用法:
    # Mock 模式（不需要 GPU，测试 pipeline 逻辑）
    python run_demo.py --mock

    # 真实生成（需要 GPU + 模型权重）
    python run_demo.py --script ../configs/demo_script.yaml --backend wan2

    # 指定输出目录
    python run_demo.py --mock --output ./my_output
"""

import argparse
import json
import os
import sys
import yaml

# 将 phase1_poc 加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator.pipeline import T2VGroundingPipeline, ShotConfig


# ── 内置 Demo 脚本（--mock 时使用）─────────────────────────────────────────
DEMO_SHOTS = [
    ShotConfig(
        shot_id=1,
        text=(
            "Alex Chen, a male detective in his 30s wearing a dark trench coat, "
            "enters the busy police station lobby. He looks determined and focused."
        ),
    ),
    ShotConfig(
        shot_id=2,
        text=(
            "Inside the interrogation room, Alex sits across the table from a nervous suspect. "
            "The fluorescent light casts harsh shadows. Alex's trench coat is draped over his chair."
        ),
    ),
    ShotConfig(
        shot_id=3,
        text=(
            "Alex walks out into a rain-soaked alley at night. "
            "His coat is drenched. Neon signs reflect on the wet pavement. "
            "He lights a cigarette and stares into the darkness."
        ),
    ),
]


def load_script(script_path: str):
    with open(script_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    global_caption = data.get("global_caption", "").strip()
    shots = []
    for item in data.get("shots", []):
        shots.append(ShotConfig(
            shot_id=item["shot_id"],
            text=item["text"],
            duration_seconds=item.get("duration_seconds", 3.0),
        ))
    return shots, global_caption


def main():
    parser = argparse.ArgumentParser(description="T2V Grounding Demo")
    parser.add_argument("--script", type=str, default=None,
                        help="镜头脚本 YAML 路径（不指定则使用内置 demo）")
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml"),
                        help="全局配置文件路径（默认: ../configs/config.yaml）")
    parser.add_argument("--output", type=str, default="./output_demo",
                        help="输出目录（默认: ./output_demo）")
    parser.add_argument("--backend", type=str, default="mock",
                        choices=["mock", "phantom", "cogvideo"],
                        help="生成后端（默认: mock，不需要 GPU）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备（默认: cuda）")
    parser.add_argument("--mock", action="store_true",
                        help="强制使用 mock 后端（等同于 --backend mock）")
    parser.add_argument("--llm-model", type=str, default=None,
                        help="LLM 模型（默认读 config.yaml）")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机 seed（覆盖 config.yaml 中的 seed；-1=随机）")
    # ── Agentic 参考图选择 ──
    parser.add_argument("--ref-selection-mode", type=str, default=None,
                        choices=["traditional", "agent", "hybrid"],
                        help="参考图选择模式: traditional(传统InsightFace打分), agent(VLM智能选择), hybrid(Agent优先+fallback)")
    parser.add_argument("--ref-selection-model", type=str, default=None,
                        help="Agent 使用的 VLM 模型（默认: claude-sonnet-4-6）")
    args = parser.parse_args()

    if args.mock:
        args.backend = "mock"

    # 加载全局配置
    cfg = {}
    config_path = os.path.abspath(args.config)
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        print(f"[Demo] 加载配置: {config_path}")
    else:
        print(f"[Demo] 未找到配置文件 {config_path}，使用默认值")

    # 加载脚本
    global_caption = ""
    if args.script:
        shots, global_caption = load_script(args.script)
        print(f"[Demo] 从 {args.script} 加载了 {len(shots)} 个镜头")
        if global_caption:
            print(f"[Demo] global_caption: {global_caption[:80]}...")
    else:
        shots = DEMO_SHOTS
        print(f"[Demo] 使用内置 demo 脚本（{len(shots)} 个镜头）")

    print(f"[Demo] 后端: {args.backend}  |  输出目录: {args.output}")
    if args.ref_selection_mode and args.ref_selection_mode != "traditional":
        print(f"[Demo] 🤖 Agentic 参考图选择: mode={args.ref_selection_mode}")
    print("=" * 60)

    # 运行 pipeline
    _gcfg = cfg.get("generator", {})
    _agcfg = cfg.get("agentic", {})  # Agentic 配置
    pipeline = T2VGroundingPipeline(
        output_dir=args.output,
        device=args.device,
        gen_backend=args.backend,
        llm_model=args.llm_model or cfg.get("llm", {}).get("model", "claude-opus-4-6-qmt"),
        wan2_t2v_dir=_gcfg.get("wan2_t2v_dir", "weights/Wan2.1-T2V-14B"),
        phantom_ckpt=_gcfg.get("phantom_ckpt", "weights/Phantom"),
        phantom_task=_gcfg.get("phantom_task", "s2v-14B"),
        use_usp=_gcfg.get("use_usp", False),
        dit_fsdp=_gcfg.get("dit_fsdp", False),
        t5_fsdp=_gcfg.get("t5_fsdp", False),
        num_inference_steps=_gcfg.get("num_inference_steps", 50),
        num_frames=_gcfg.get("num_frames", 81),
        fps=_gcfg.get("fps", 24),
        width=_gcfg.get("width", 832),
        height=_gcfg.get("height", 480),
        guide_scale_text=_gcfg.get("guide_scale_text", 7.5),
        guide_scale_img=_gcfg.get("guide_scale_img", 5.0),
        # 命令行 --seed 优先，其次 config.yaml，默认 -1（随机）
        seed=args.seed if args.seed is not None else _gcfg.get("seed", -1),
        # ── Agentic 参考图选择参数 ──
        # 优先级: 命令行 > config.yaml > 默认值(traditional)
        ref_selection_mode=(
            args.ref_selection_mode
            or _agcfg.get("ref_selection_mode", "traditional")
        ),
        ref_selection_model=(
            args.ref_selection_model
            or _agcfg.get("ref_selection_model", "claude-sonnet-4-6")
        ),
    )
    results = pipeline.run(shots, global_caption=global_caption)

    # 打印总结
    print("\n" + "=" * 60)
    print(f"[Demo] 完成！共生成 {len(results)} 个视频镜头：")
    for r in results:
        refs_info = {k: len(v) for k, v in r.reference_used.items()}
        print(f"  Shot {r.shot_id}: {r.video_path}")
        print(f"    entities={r.entity_ids}  references_used={refs_info}")

    report_path = os.path.join(args.output, "pipeline_report.json")
    print(f"\n详细报告: {report_path}")

    # 如果有评测模块，自动触发评测
    eval_script = os.path.join(os.path.dirname(__file__), "..", "evaluation", "eval_pipeline.py")
    if os.path.exists(eval_script):
        print(f"\n运行自动评测... (python {eval_script} --report {report_path})")
        os.system(f"python {eval_script} --report {report_path}")


if __name__ == "__main__":
    main()
