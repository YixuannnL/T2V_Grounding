"""
orchestrator/pipeline.py

正确的 T2V Grounding 流程：

  Shot N 处理步骤
  ─────────────────────────────────────────────────────
  1. LLM 解析 prompt_N → 提取实体列表
  2. 路由决策：
       - registry 里有实体的参考图 → Phantom (S2V)，传入参考图
       - registry 完全空（Shot 1 或新实体首次出现） → T2V，纯文本生成
  3. 生成 video_N
  4. 【生成后】对 video_N 做 Grounding：
       - 从视频抽帧
       - 用 Grounding DINO 定位每个实体
       - Re-ID 质量评分，筛选最优 crops
       - 存入 Entity Registry
  5. registry 里的 crops 供 Shot N+1 使用

  关键原则
  ─────────────────────────────────────────────────────
  · Shot 1 → T2V（registry 必然为空）
  · Shot 2+ → Phantom，参考图来自上一步 grounding 结果
  · 新实体首次出现 → T2V 生成，随后入库，下一镜就有 ref 了
  · Grounding 永远发生在生成之后，不影响当前镜头的生成
"""

import os
import sys
import json
import shutil
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from PIL import Image
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entity_parser.parser import EntityParser, ParseResult, Entity
from visual_grounding.grounder import VideoGrounder, extract_frames
from visual_grounding.reid import ReferenceQualityScorer
from reference_manager.registry import EntityRegistry, ReferenceEntry
from generator.ref2video import Reference2VideoGenerator, GenerationConfig


@dataclass
class ShotConfig:
    shot_id: int
    text: str
    duration_seconds: float = 3.0


@dataclass
class ShotResult:
    shot_id: int
    text: str
    video_path: str
    generation_mode: str        # "t2v" | "phantom"
    entity_ids: List[str]
    reference_used: dict        # entity_id -> [crop_path, ...]
    grounded_entities: dict     # entity_id -> [crop_path, ...]  （本镜 grounding 结果）
    metadata: dict = field(default_factory=dict)


class T2VGroundingPipeline:
    """
    T2V Grounding 主 Pipeline

    用法:
        pipeline = T2VGroundingPipeline(output_dir="./output")
        results = pipeline.run(shots)
    """

    def __init__(
        self,
        output_dir: str = "./output",
        device: str = "cuda",
        gen_backend: str = "phantom",
        llm_model: str = "claude-opus-4-6-qmt",
        grounding_threshold: float = 0.35,
        min_ref_quality: float = 0.4,
        max_refs_per_entity: int = 3,
        wan2_t2v_dir: str = "weights/Wan2.1-T2V-14B",
        phantom_ckpt: str = "weights/Phantom",
        phantom_task: str = "s2v-14B",
        use_usp: bool = False,
        dit_fsdp: bool = False,
        t5_fsdp: bool = False,
        num_inference_steps: int = 50,
        num_frames: int = 81,
        fps: int = 24,
        width: int = 832,
        height: int = 480,
        guide_scale_text: float = 7.5,
        guide_scale_img: float = 5.0,
        seed: int = -1,
    ):
        import torch
        import torch.distributed as dist

        # 检测 torchrun 分布式环境
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_rank0 = (self.rank == 0)

        self.output_dir = output_dir
        self.device = device
        self.mock_mode = (gen_backend == "mock")
        self.min_ref_quality = min_ref_quality
        self.max_refs_per_entity = max_refs_per_entity

        if self.is_rank0:
            for sub in ["frames", "crops", "videos", "registry", "selected_refs"]:
                os.makedirs(os.path.join(output_dir, sub), exist_ok=True)
            # rank 0 等目录创建完再让其他 rank 继续
        if self.world_size > 1 and dist.is_initialized():
            dist.barrier()

        # LLM / Grounding / Registry 只在 rank 0 初始化
        if self.is_rank0:
            self.parser = EntityParser(model=llm_model)
            self.registry = EntityRegistry(
                os.path.join(output_dir, "registry", "entities.db")
            )
            if not self.mock_mode:
                self.grounder = VideoGrounder(
                    device=device, box_threshold=grounding_threshold
                )
                self.scorer = ReferenceQualityScorer(device=device)
            else:
                self.grounder = None
                self.scorer = None

        # 所有 rank 都初始化 generator（参与模型推理）
        gen_config = GenerationConfig(
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            fps=fps,
            width=width,
            height=height,
            guide_scale_text=guide_scale_text,
            guide_scale_img=guide_scale_img,
            seed=seed if seed >= 0 else int(torch.randint(0, 2**31, (1,)).item()),
        )
        self.generator = Reference2VideoGenerator(
            backend=gen_backend,
            device=device,
            config=gen_config,
            wan2_t2v_dir=wan2_t2v_dir,
            phantom_ckpt=phantom_ckpt,
            phantom_task=phantom_task,
            use_usp=use_usp,
            dit_fsdp=dit_fsdp,
            t5_fsdp=t5_fsdp,
        )

        if self.is_rank0:
            print(f"[Pipeline] 初始化完成 | output={output_dir} | "
                  f"mock={self.mock_mode} | world_size={self.world_size}")

    # ── 主入口 ────────────────────────────────────────────────────────────────

    def run(self, shots: List[ShotConfig], global_caption: str = "") -> List[ShotResult]:
        import torch.distributed as dist

        # global_caption 仅用于 entity parser 预建实体图谱（shot_id=0 解析）
        # pipeline 本身不再保留原始文本，生成 prompt 改由 build_shot_context() 按需过滤
        if self.is_rank0 and global_caption:
            self.parser.set_global_caption(global_caption)

        results = []
        for shot in shots:
            if self.is_rank0:
                print(f"\n{'='*60}")
                print(f"[Pipeline] ── Shot {shot.shot_id} ──")
            result = self._process_shot(shot)
            if self.is_rank0:
                results.append(result)

        if self.is_rank0:
            self._save_report(results)
        return results

    def _process_shot(self, shot: ShotConfig) -> Optional[ShotResult]:
        import torch.distributed as dist

        # ── Step 1 & 2: rank 0 解析实体 + 收集参考图路径 ─────────────────────
        if self.is_rank0:
            parse_result = self.parser.parse(shot.text, shot.shot_id)
            high_priority = [
                e for e in parse_result.entities
                if e.grounding_priority in ("high", "medium")
            ]
            print(f"[Pipeline] 实体: {[e.entity_id for e in high_priority]}")

            reference_used: dict = {}
            all_ref_paths: List[str] = []
            # 按 high → medium 优先级排序，每个实体只取 best 1 张，最多 4 个实体
            priority_order = {"high": 0, "medium": 1}
            sorted_entities = sorted(
                high_priority,
                key=lambda e: priority_order.get(e.grounding_priority, 2)
            )
            for entity in sorted_entities[:4]:
                refs = self.registry.query(
                    entity.entity_id, top_k=1,
                    min_quality=self.min_ref_quality
                )
                if refs:
                    reference_used[entity.entity_id] = [refs[0].crop_path]
                    all_ref_paths.append(refs[0].crop_path)

            generation_mode = "phantom" if all_ref_paths and not self.mock_mode else \
                              ("t2v" if not self.mock_mode else "mock")
            print(f"[Pipeline] 模式: {generation_mode} | 参考图: {len(all_ref_paths)} 张")

            # ── 构建生成 prompt ────────────────────────────────────────────────
            # 不直接使用原始 global_caption 全文，因为它描述整个视频叙事，
            # 会把当前 shot 不应出现的实体（如后续 shot 才登场的角色）注入 prompt，
            # 导致视频模型生成出不属于本 shot 的内容。
            #
            # 正确做法：从 entity graph 中只取本 shot 实际出现的实体的外观/属性描述，
            # 精准注入，保证 prompt 语义与本 shot 内容完全对应。
            shot_context = self.parser.build_shot_context(parse_result)
            if shot_context:
                gen_prompt = f"{shot_context}\n\n{shot.text}"
                print(f"[Pipeline] Shot context (filtered):\n{shot_context}")
            else:
                gen_prompt = shot.text

            # 广播给其他 rank
            broadcast_data = [gen_prompt, all_ref_paths, generation_mode]
        else:
            broadcast_data = [None, None, None]

        if self.world_size > 1 and dist.is_initialized():
            dist.broadcast_object_list(broadcast_data, src=0)

        prompt, all_ref_paths, generation_mode = broadcast_data

        # ── Step 3: 所有 rank 加载参考图并生成 ───────────────────────────────
        all_ref_images: List[Image.Image] = []
        for p in (all_ref_paths or []):
            try:
                all_ref_images.append(Image.open(p).convert("RGB"))
            except Exception:
                pass

        video_path = os.path.join(
            self.output_dir, "videos", f"shot_{shot.shot_id:03d}.mp4"
        )
        self.generator.generate(
            text_prompt=prompt,
            references=all_ref_images,
            output_path=video_path,
        )

        # ── Step 4: rank 0 做 grounding → 入库 ───────────────────────────────
        if not self.is_rank0:
            return None

        if self.is_rank0:
            print(f"[Pipeline] 视频已生成: {video_path}")

        grounded_entities: dict = {}
        if not self.mock_mode:
            grounded_entities = self._ground_and_register(
                video_path=video_path,
                parse_result=parse_result,
                shot_id=shot.shot_id,
            )
            print(f"[Pipeline] Grounding 入库: "
                  f"{sum(len(v) for v in grounded_entities.values())} 张 crops")

        return ShotResult(
            shot_id=shot.shot_id,
            text=shot.text,
            video_path=video_path,
            generation_mode=generation_mode,
            entity_ids=[e.entity_id for e in parse_result.entities],
            reference_used=reference_used,
            grounded_entities=grounded_entities,
        )

    # ── 生成后 Grounding & 入库 ───────────────────────────────────────────────

    def _ground_and_register(
        self,
        video_path: str,
        parse_result: ParseResult,
        shot_id: int,
    ) -> dict:
        """
        对已生成的视频做 Grounding，提取每个实体的 crop，
        质量评分后存入 Entity Registry，供后续镜头使用。
        """
        frames_dir = os.path.join(
            self.output_dir, "frames", f"shot_{shot_id:03d}"
        )
        crops_dir = os.path.join(
            self.output_dir, "crops", f"shot_{shot_id:03d}"
        )
        frames = extract_frames(video_path, frames_dir, fps=1.0)
        if not frames:
            print(f"[Pipeline] 抽帧失败: {video_path}")
            return {}

        grounded: dict = {}

        for entity in parse_result.entities:
            if entity.grounding_priority == "low":
                continue

            # Grounding DINO 定位
            ground_results = self.grounder.ground_in_frames(
                frame_paths=frames,
                entity_text=entity.text_description,
                entity_id=entity.entity_id,
                output_dir=crops_dir,
                max_results=5,
                entity_type=entity.type,
            )
            if not ground_results:
                print(f"[Pipeline] {entity.entity_id}: grounding 未找到")
                continue

            # Re-ID 质量评分（传入 ground_results 以利用 bbox 面积）
            crop_paths = [r.crop_path for r in ground_results]
            scored = self.scorer.rank_references(crop_paths, entity.type, ground_results=ground_results)
            good = [s for s in scored if s.final_score >= self.min_ref_quality]

            if not good:
                print(f"[Pipeline] {entity.entity_id}: 所有 crops 质量不足")
                continue

            # 入库 + 同步复制到 selected_refs/ 供人工检查
            selected_dir = os.path.join(
                self.output_dir, "selected_refs", entity.entity_id
            )
            os.makedirs(selected_dir, exist_ok=True)
            registered_paths = []
            for s in good[:self.max_refs_per_entity]:
                entry = ReferenceEntry(
                    entity_id=entity.entity_id,
                    shot_id=shot_id,
                    frame_path="",
                    crop_path=s.crop_path,
                    quality_score=float(s.final_score),
                    source="grounding",
                )
                self.registry.register(entity.entity_id, entry)
                registered_paths.append(s.crop_path)

                # 复制到 selected_refs/{entity_id}/shot{N}_score{X:.2f}.jpg
                ext = os.path.splitext(s.crop_path)[1] or ".jpg"
                dst_name = f"shot{shot_id:03d}_score{s.final_score:.2f}{ext}"
                shutil.copy2(s.crop_path, os.path.join(selected_dir, dst_name))

            grounded[entity.entity_id] = registered_paths
            print(f"[Pipeline] {entity.entity_id}: "
                  f"入库 {len(registered_paths)} 张 "
                  f"(best score={good[0].final_score:.3f})")

        return grounded

    # ── 报告 ──────────────────────────────────────────────────────────────────

    def _save_report(self, results: List[ShotResult]):
        report_path = os.path.join(self.output_dir, "pipeline_report.json")
        report = {
            "total_shots": len(results),
            "registry_stats": self.registry.stats(),
            "flow_summary": [
                {
                    "shot_id": r.shot_id,
                    "mode": r.generation_mode,
                    "refs_used": sum(len(v) for v in r.reference_used.values()),
                    "crops_grounded": sum(len(v) for v in r.grounded_entities.values()),
                }
                for r in results
            ],
            "shots": [asdict(r) for r in results],
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[Pipeline] 报告已保存: {report_path}")
