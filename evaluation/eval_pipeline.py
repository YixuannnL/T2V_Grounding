#!/usr/bin/env python3
"""
evaluation/eval_pipeline.py — 自动化评测入口
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase1_poc"))

from metrics import ConsistencyEvaluator


def main():
    parser = argparse.ArgumentParser(description="T2V Grounding 自动化评测")
    parser.add_argument("--report", required=True, help="pipeline_report.json 路径")
    parser.add_argument("--output", default=None, help="评测结果保存路径")
    args = parser.parse_args()

    print("\n[Eval] 开始自动化评测...")
    evaluator = ConsistencyEvaluator()
    eval_report = evaluator.evaluate_pipeline(args.report)

    print(f"\n{'='*50}")
    print(f"[Eval] 评测结果汇总（共 {eval_report.total_shots} 个镜头）:")
    print(f"  avg CLIP-I (主体一致性): {eval_report.avg_clip_i:.4f}")
    print(f"  avg CLIP-T (文本对齐):   {eval_report.avg_clip_t:.4f}")
    print(f"  avg FaceID (人脸一致性): {eval_report.avg_faceid:.4f}" if eval_report.avg_faceid >= 0 else "  avg FaceID: N/A")

    result = {
        "summary": {
            "total_shots": eval_report.total_shots,
            "avg_clip_i": eval_report.avg_clip_i,
            "avg_clip_t": eval_report.avg_clip_t,
            "avg_faceid": eval_report.avg_faceid,
        },
        "per_shot": [
            {
                "shot_id": m.shot_id,
                "clip_i": m.clip_i_score,
                "clip_t": m.clip_t_score,
                "faceid": m.faceid_score,
            }
            for m in eval_report.per_shot
        ]
    }

    output_path = args.output or args.report.replace("pipeline_report.json", "eval_report.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Eval] 评测报告已保存: {output_path}")


if __name__ == "__main__":
    main()
