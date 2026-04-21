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

from entity_parser.parser import EntityParser, ParseResult, Entity, EntityCountInfo, CloseupLightingAnalysis
from visual_grounding.referdino_grounder import ReferDINOGrounder, extract_frames, MultiEntityGroundingResult
from visual_grounding.reid import ReferenceQualityScorer
from reference_manager.registry import EntityRegistry, ReferenceEntry
from generator.ref2video import Reference2VideoGenerator, GenerationConfig
from verification.entity_count_verifier import (
    EntityCountVerifier,
    EntityCountExpectation,
    VerificationResult,
    VerificationStatus,
    build_count_enhanced_prompt,
)


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

    Agentic 模式（参考图智能选择）:
        pipeline = T2VGroundingPipeline(
            output_dir="./output",
            ref_selection_mode="agent",  # "traditional" | "agent" | "hybrid"
        )
    """

    def __init__(
        self,
        output_dir: str = "./output",
        device: str = "cuda",
        gen_backend: str = "phantom",
        llm_model: str = "claude-opus-4-6-qmt",
        grounding_threshold: float = 0.35,
        min_ref_quality: float = 0.4,
        high_quality_threshold: float = 0.85,  # 高质量正脸阈值，用于锚点选择
        min_face_confidence: float = 0.3,      # 最低人脸置信度，低于此值视为"无脸参考"
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
        # ── Shot 1 验证参数 ──
        enable_shot1_verification: bool = True,
        shot1_max_retries: int = 3,
        shot1_verify_person_count: bool = True,
        # ── Agentic 参考图选择参数 ──
        ref_selection_mode: str = "traditional",  # "traditional" | "agent" | "hybrid"
        ref_selection_model: str = "claude-sonnet-4-6",  # Agent 使用的 VLM 模型
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
        self.high_quality_threshold = high_quality_threshold
        self.min_face_confidence = min_face_confidence  # 人脸置信度阈值
        self.max_refs_per_entity = max_refs_per_entity

        # Shot 1 验证参数
        self.enable_shot1_verification = enable_shot1_verification
        self.shot1_max_retries = shot1_max_retries
        self.shot1_verify_person_count = shot1_verify_person_count

        # Agentic 参考图选择参数
        self.ref_selection_mode = ref_selection_mode
        self.ref_selection_model = ref_selection_model
        self._ref_selection_agent = None  # 延迟初始化

        if self.is_rank0:
            for sub in ["frames", "crops", "videos", "registry", "selected_refs", "prompts", "used_refs"]:
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
                self.grounder = ReferDINOGrounder(device=device)
                self.scorer = ReferenceQualityScorer(device=device)
                # 初始化验证器（仍使用 ReferDINO）
                self.verifier = EntityCountVerifier(
                    grounder=self.grounder,
                    detection_threshold=grounding_threshold,
                    device=device,
                )
            else:
                self.grounder = None
                self.scorer = None
                self.verifier = None

        # 所有 rank 都初始化 generator（参与模型推理）
        # 保存 base_seed，每个 shot 会递增使用不同的 seed
        self.base_seed = seed if seed >= 0 else int(torch.randint(0, 2**31, (1,)).item())
        gen_config = GenerationConfig(
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            fps=fps,
            width=width,
            height=height,
            guide_scale_text=guide_scale_text,
            guide_scale_img=guide_scale_img,
            seed=self.base_seed,
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
            if self.ref_selection_mode != "traditional":
                print(f"[Pipeline] 🤖 Agentic 参考图选择已启用 (mode={self.ref_selection_mode}, model={self.ref_selection_model})")

    # ── Agentic 参考图选择 ─────────────────────────────────────────────────────

    @property
    def ref_selection_agent(self):
        """获取参考图选择 Agent（延迟初始化）"""
        if self._ref_selection_agent is None and self.ref_selection_mode != "traditional":
            try:
                from agents.reference_selection_agent import ReferenceSelectionAgent
                self._ref_selection_agent = ReferenceSelectionAgent(
                    model=self.ref_selection_model,
                    enable_fallback=(self.ref_selection_mode == "hybrid"),
                    verbose=True,
                )
                print(f"[Pipeline] ReferenceSelectionAgent 初始化成功")
            except Exception as e:
                print(f"[Pipeline] ⚠️ ReferenceSelectionAgent 初始化失败: {e}")
                print(f"[Pipeline] 回退到传统参考图选择模式")
                self.ref_selection_mode = "traditional"
        return self._ref_selection_agent

    def _agent_select_reference(
        self,
        entity,
        candidates: List[ReferenceEntry],
        shot_text: str,
        shot_type: str,
    ) -> Optional[ReferenceEntry]:
        """
        使用 Agent 从候选参考图中选择最佳参考

        Args:
            entity: Entity 对象
            candidates: 候选 ReferenceEntry 列表
            shot_text: 当前镜头描述
            shot_type: 镜头类型 (close-up / medium / wide)

        Returns:
            选中的 ReferenceEntry，或 None
        """
        if not candidates:
            return None

        agent = self.ref_selection_agent
        if agent is None:
            # Agent 不可用，回退到传统方法
            return candidates[0] if candidates else None

        # 调用 Agent 选择
        crop_paths = [r.crop_path for r in candidates]
        result = agent.select_best_reference(
            entity_id=entity.entity_id,
            entity_type=entity.type,
            entity_description=entity.text_description,
            candidates=crop_paths,
            shot_context=shot_text,
            shot_type=shot_type,
        )

        if result.selected_index >= 0 and result.selected_index < len(candidates):
            selected = candidates[result.selected_index]
            print(f"[Pipeline] 🤖 Agent 选择 {entity.entity_id}: shot={selected.shot_id}, "
                  f"confidence={result.confidence:.2f}")
            print(f"[Pipeline]    理由: {result.reason[:80]}...")
            return selected

        return candidates[0] if candidates else None

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

            # 打印实体数量预期（用于 Shot 1 验证）
            if parse_result.entity_counts:
                for ec in parse_result.entity_counts:
                    print(f"[Pipeline] 预期 {ec.entity_type}: {ec.expected_count} "
                          f"({'显式' if ec.is_explicit else '推断'})")

            reference_used: dict = {}
            all_ref_paths: List[str] = []

            # ── 检测镜头类型（close-up / medium / wide）───────────────────────
            shot_text_lower = shot.text.lower()
            is_closeup = any(kw in shot_text_lower for kw in ["close-up", "closeup", "close up", "tight shot", "focuses on"])
            is_wide = any(kw in shot_text_lower for kw in ["wide shot", "wide angle", "establishing shot", "full shot"])

            # ── 检测局部特写（手、脚、物体等）─────────────────────────────────
            # 对于局部特写，即使 LLM 关联到某个 character，也不应该用人脸参考图
            # 因为手部特写用人脸参考会导致奇怪的结果
            body_part_keywords = [
                "hand", "hands", "finger", "fingers", "palm",
                "foot", "feet", "leg", "legs",
                "arm", "arms", "shoulder",
                "back", "torso",
            ]
            is_body_part_closeup = is_closeup and any(kw in shot_text_lower for kw in body_part_keywords)
            if is_body_part_closeup:
                print(f"[Pipeline] 检测到局部特写镜头（身体部位），将跳过 character 人脸参考图")

            # ── 分离 location 和非 location 实体 ──────────────────────────────
            # 注意：过滤掉 style 类型（风格通过 Global Context 处理，不作为参考图）
            location_entities = [e for e in high_priority if e.type == "location"]
            non_location_entities = [e for e in high_priority if e.type not in ("location", "style")]

            # ── 处理非 location 实体（character, object）──────────────────────
            # 按 high → medium 优先级排序，每个实体只取 best 1 张
            priority_order = {"high": 0, "medium": 1}
            sorted_non_loc = sorted(
                non_location_entities,
                key=lambda e: priority_order.get(e.grounding_priority, 2)
            )

            # ── Agentic 镜头类型处理：智能决定是否传 location ──────────────────
            # 策略：
            #   - wide/medium shot: 始终传 location（场景一致性）
            #   - close-up shot: 由 LLM 分析光线复杂度，决定是否传 location
            #     - 简单光线：不传 location，通过 prompt 描述光线
            #     - 复杂光线：传 location，保持色调一致
            #
            # 注意：Phantom 最多支持 4 张参考图
            closeup_lighting_analysis = None  # 用于后续 prompt 构建

            if is_closeup:
                # Close-up 镜头：Agentic 决策是否需要 location
                print(f"[Pipeline] 检测到 close-up 镜头，启动 Agentic 光线分析...")

                # 检查是否有 location 参考图可用
                has_location_ref = False
                for loc_entity in location_entities:
                    refs = self.registry.query(
                        loc_entity.entity_id, top_k=1,
                        min_quality=0.3,
                        anchor_strategy="earliest_good"
                    )
                    if refs:
                        has_location_ref = True
                        break

                if has_location_ref and location_entities:
                    # 调用 LLM 分析光线复杂度
                    closeup_lighting_analysis = self.parser.analyze_closeup_lighting(location_entities)

                    if closeup_lighting_analysis and closeup_lighting_analysis.needs_location_ref:
                        # LLM 判断需要传 location
                        include_location = True
                        max_non_loc_refs = 3
                        print(f"[Pipeline] Agentic 决策: 传 location 参考图 "
                              f"(complexity={closeup_lighting_analysis.complexity_score}, "
                              f"reason='{closeup_lighting_analysis.reason}')")
                    else:
                        # LLM 判断不需要传 location，通过 prompt 描述光线
                        include_location = False
                        max_non_loc_refs = 4
                        print(f"[Pipeline] Agentic 决策: 不传 location，通过 prompt 描述光线")
                        if closeup_lighting_analysis:
                            print(f"[Pipeline] 光线描述: {closeup_lighting_analysis.lighting_description}")
                else:
                    # 没有 location 参考图可用，跳过分析
                    include_location = False
                    max_non_loc_refs = 4
                    print(f"[Pipeline] Close-up 镜头，无可用 location 参考图")

            elif is_wide:
                include_location = True
                max_non_loc_refs = 3 if location_entities else 4
                print(f"[Pipeline] 检测到 wide shot 镜头，传 location 参考图")
            else:
                # medium shot
                include_location = True
                max_non_loc_refs = 3 if location_entities else 4

            # ── 关键改动：使用锚点策略，防止误差累积 ──────────────────────────
            # 对于 character，优先使用"最早的大正脸"参考图（锚点）
            # 策略：高质量正脸 > 早期出现，但同等高质量选更早的
            #
            # 【v2.3 新增】Frontal-Aware 检查：
            # 如果 character 的锚点参考是背影/侧面（id_confidence < min_face_confidence），
            # 则不使用这个参考图，改为在 prompt 中注入外观描述（Appearance Context）
            #
            # 【v3.0 新增】Agentic 参考图选择：
            # 如果 ref_selection_mode != "traditional"，使用 VLM Agent 智能选择参考图
            # Agent 会根据当前 shot 的上下文（镜头类型、动作描述）选择最合适的参考图
            faceless_characters = []  # 记录无脸参考的 character，后续构建 Appearance Context

            # 检测镜头类型（用于 Agent 选择）
            shot_type_for_agent = "medium"
            if is_closeup:
                shot_type_for_agent = "close-up"
            elif is_wide:
                shot_type_for_agent = "wide"

            for entity in sorted_non_loc[:max_non_loc_refs]:
                # character 使用锚点查询：优先选择高质量正脸
                if entity.type == "character":
                    # 【v2.4】Body Part Closeup 检查：
                    # 如果是身体部位特写（手、脚等），即使 LLM 关联到 character，
                    # 也不应该传人脸参考图，否则会导致生成结果不合理
                    if is_body_part_closeup:
                        faceless_characters.append(entity)
                        print(f"[Pipeline] {entity.entity_id}: 身体部位特写镜头，"
                              f"跳过人脸参考图，将使用 Appearance Context 描述")
                        continue

                    # 【v3.0】根据 ref_selection_mode 选择参考图
                    if self.ref_selection_mode != "traditional":
                        # Agentic 模式：使用 VLM Agent 智能选择
                        candidates = self.registry.query(
                            entity.entity_id,
                            top_k=6,  # 给 Agent 更多候选
                            min_quality=self.min_ref_quality,
                            anchor_strategy="earliest_good",
                        )
                        if candidates:
                            anchor = self._agent_select_reference(
                                entity=entity,
                                candidates=candidates,
                                shot_text=shot.text,
                                shot_type=shot_type_for_agent,
                            )
                        else:
                            anchor = None
                    else:
                        # 传统模式：使用固定的锚点策略
                        anchor = self.registry.query_anchor(
                            entity.entity_id,
                            min_quality=self.min_ref_quality,
                            high_quality_threshold=self.high_quality_threshold,
                        )

                    if anchor:
                        # 【v2.3】检查 id_confidence：如果太低，说明是背影/侧面
                        if anchor.id_confidence < self.min_face_confidence:
                            # 无脸参考：不传参考图，后续通过 Appearance Context 描述外观
                            faceless_characters.append(entity)
                            print(f"[Pipeline] {entity.entity_id}: 锚点为无脸参考 "
                                  f"(id_conf={anchor.id_confidence:.2f} < {self.min_face_confidence})"
                                  f"，跳过参考图，将使用 Appearance Context")
                        else:
                            # 有脸参考：正常使用
                            reference_used[entity.entity_id] = [anchor.crop_path]
                            all_ref_paths.append(anchor.crop_path)
                            print(f"[Pipeline] {entity.entity_id}: 使用{'Agent选择的' if self.ref_selection_mode != 'traditional' else '锚点'}参考图 "
                                  f"(shot={anchor.shot_id}, score={anchor.quality_score:.2f}, "
                                  f"id_conf={anchor.id_confidence:.2f})")
                else:
                    # object 等其他类型
                    # 【v3.0】同样支持 Agentic 选择
                    if self.ref_selection_mode != "traditional":
                        # Agentic 模式
                        candidates = self.registry.query(
                            entity.entity_id,
                            top_k=6,
                            min_quality=self.min_ref_quality,
                            anchor_strategy="earliest_good",
                        )
                        if candidates:
                            selected = self._agent_select_reference(
                                entity=entity,
                                candidates=candidates,
                                shot_text=shot.text,
                                shot_type=shot_type_for_agent,
                            )
                            if selected:
                                reference_used[entity.entity_id] = [selected.crop_path]
                                all_ref_paths.append(selected.crop_path)
                    else:
                        # 传统模式
                        refs = self.registry.query(
                            entity.entity_id, top_k=1,
                            min_quality=self.min_ref_quality,
                            anchor_strategy="earliest_good"
                        )
                        if refs:
                            reference_used[entity.entity_id] = [refs[0].crop_path]
                            all_ref_paths.append(refs[0].crop_path)

            # ── 处理 location 实体（场景一致性 / 光线一致性）────────────────────
            # 根据上面的 Agentic 决策结果决定是否传 location
            # 【v3.0】location 也支持 Agent 选择
            if include_location:
                for loc_entity in location_entities:
                    if self.ref_selection_mode != "traditional":
                        # Agentic 模式
                        candidates = self.registry.query(
                            loc_entity.entity_id,
                            top_k=6,
                            min_quality=0.3,
                            anchor_strategy="earliest_good",
                        )
                        if candidates:
                            anchor = self._agent_select_reference(
                                entity=loc_entity,
                                candidates=candidates,
                                shot_text=shot.text,
                                shot_type=shot_type_for_agent,
                            )
                        else:
                            anchor = None
                    else:
                        # 传统模式
                        anchor = self.registry.query_anchor_location(
                            loc_entity.entity_id,
                            min_quality=0.3,
                            high_quality_threshold=0.7,  # 质量 >= 0.7 的直接用
                            quality_gap_ratio=1.5,       # 后续需要好 1.5 倍以上才切换
                        )

                    if anchor:
                        reference_used[loc_entity.entity_id] = [anchor.crop_path]
                        all_ref_paths.append(anchor.crop_path)
                        print(f"[Pipeline] Location '{loc_entity.entity_id}' 使用{'Agent选择的' if self.ref_selection_mode != 'traditional' else '锚点'}参考图 "
                              f"(shot={anchor.shot_id}, score={anchor.quality_score:.2f})")
                    else:
                        print(f"[Pipeline] Location '{loc_entity.entity_id}' 无参考图 (新场景，将在生成后 grounding)")

            # ── 关键改动：基于是否有 subject ref 决定生成模式 ──────────────────
            # 问题：如果只有 location ref（无 character/object ref），S2V 会失去
            #       人物外观锚定，可能生成动画风格视频
            # 解决：只有当有 subject（character/object）ref 时才用 S2V，
            #       否则回退 T2V + prompt 强化环境描述
            #
            # 【v2.3 改进】"有效的 subject ref" 定义：
            #   - character: 必须有脸（不在 faceless_characters 中）
            #   - object: 正常计入
            #   - 无脸 character 不计入有效 subject ref
            #
            # 【v3.1 修复】Character-Aware 路由：
            #   问题：当场景中有 character 实体但只有 object 参考图时，
            #         S2V 模式会导致人物风格漂移（如动画化）
            #   原因：object 参考图（如马）不能锚定人物外观，Phantom 会"脑补"人物
            #   修复：如果场景有 character 实体但无 frontal character 参考，
            #         即使有 object 参考也应该回退 T2V
            #
            # 分类统计
            character_entities_in_shot = [e for e in non_location_entities if e.type == "character"]
            object_entities_in_shot = [e for e in non_location_entities if e.type == "object"]

            # 有效的 character 参考（frontal，不在 faceless 中）
            has_frontal_character_ref = any(
                e.entity_id in reference_used
                for e in character_entities_in_shot
                if e not in faceless_characters
            )
            # 有效的 object 参考
            has_object_ref = any(
                e.entity_id in reference_used
                for e in object_entities_in_shot
            )

            num_character_refs = len([
                e for e in character_entities_in_shot
                if e.entity_id in reference_used and e not in faceless_characters
            ])
            num_object_refs = len([
                e for e in object_entities_in_shot
                if e.entity_id in reference_used
            ])
            num_location_refs = len([e for e in location_entities if e.entity_id in reference_used])

            # 【v3.1】Character-Aware 路由决策
            # 核心原则：如果场景有 character，必须有 frontal character 参考才能用 S2V
            if self.mock_mode:
                generation_mode = "mock"
            elif has_frontal_character_ref:
                # 有 frontal character 参考 → S2V 安全，人物外观有锚定
                generation_mode = "phantom"
            elif character_entities_in_shot and not has_frontal_character_ref:
                # 【v3.1 关键修复】场景有 character 但无 frontal 参考
                # 即使有 object 参考（如马），也不能锚定人物外观
                # 必须回退 T2V，否则会导致动画化
                generation_mode = "t2v"
                reason_parts = []
                if faceless_characters:
                    reason_parts.append(f"{len(faceless_characters)} 个无脸 character")
                else:
                    reason_parts.append("character 无参考图")
                if has_object_ref:
                    reason_parts.append(f"有 {num_object_refs} 个 object 参考但不能锚定人物")
                print(f"[Pipeline] ⚠️  {', '.join(reason_parts)}，回退 T2V 避免人物风格漂移")
                # 清空 all_ref_paths，T2V 不传参考图
                all_ref_paths = []
            elif has_object_ref:
                # 场景无 character 实体，但有 object 参考 → S2V 可以
                generation_mode = "phantom"
            else:
                # 无任何 subject 参考 → T2V
                generation_mode = "t2v"
                if num_location_refs > 0:
                    print(f"[Pipeline] ⚠️  只有 location 参考，回退 T2V")
                    all_ref_paths = []

            print(f"[Pipeline] 模式: {generation_mode} | 参考图: {len(all_ref_paths)} 张 "
                  f"(frontal character: {num_character_refs}, object: {num_object_refs}, "
                  f"location: {num_location_refs}, 无脸 character: {len(faceless_characters)})")

            # ── 构建生成 prompt ────────────────────────────────────────────────
            # Prompt 分层：
            #   1. global_context: 全局风格/氛围/叙事背景
            #   2. lighting_context: Close-up 光线引导（仅当不传 location 时）
            #   3. shot_context:   本 shot 出现的实体描述
            #   4. shot.text:      本 shot 的具体动作描述
            #
            # 不直接使用原始 global_caption 全文，因为它描述整个视频叙事，
            # 会把当前 shot 不应出现的实体（如后续 shot 才登场的角色）注入 prompt，
            # 导致视频模型生成出不属于本 shot 的内容。
            prompt_parts = []

            # Layer 1: 全局语义上下文（风格、氛围、叙事背景）
            global_context = self.parser.build_global_context_prompt()
            if global_context:
                prompt_parts.append(global_context)
                print(f"[Pipeline] Global context:\n{global_context}")

            # Layer 2: Close-up 光线引导（仅当不传 location 参考图时）
            if is_closeup and not include_location and closeup_lighting_analysis:
                lighting_prompt = self.parser.build_closeup_lighting_prompt(closeup_lighting_analysis)
                if lighting_prompt:
                    prompt_parts.append(lighting_prompt)
                    print(f"[Pipeline] Lighting context:\n{lighting_prompt}")

            # Layer 2.5: T2V 回退时的环境描述强化
            # 当无 subject ref 回退 T2V 时，通过 prompt 描述 location 环境以保持一致性
            if generation_mode == "t2v" and location_entities:
                env_desc_parts = []
                for loc_entity in location_entities:
                    # 从 parser 的已知实体库中获取完整属性
                    known = self.parser._get_entity(loc_entity.entity_id)
                    source = known if known else loc_entity
                    desc = source.text_description.strip()
                    if source.attributes:
                        attr_items = [f"{k}: {v}" for k, v in source.attributes.items() if v]
                        if attr_items:
                            desc = f"{desc} ({', '.join(attr_items)})"
                    env_desc_parts.append(desc)
                if env_desc_parts:
                    env_context = "[Environment Context]\n" + "\n".join(f"Scene: {d}." for d in env_desc_parts)
                    prompt_parts.append(env_context)
                    print(f"[Pipeline] Environment context (T2V fallback):\n{env_context}")

            # Layer 2.6【v2.3 新增】: 无脸 character 的外观描述（Appearance Context）
            # 当 character 只有背影参考（无脸）时，通过 prompt 详细描述其外观
            # 这样 T2V 生成时可以保持衣服、发型等外观一致性
            if faceless_characters:
                appearance_parts = []
                for fc_entity in faceless_characters:
                    known = self.parser._get_entity(fc_entity.entity_id)
                    source = known if known else fc_entity
                    desc = source.text_description.strip()
                    if source.attributes:
                        # 筛选外观相关属性
                        appearance_attrs = {}
                        appearance_keys = [
                            "hair_color", "hair_style", "clothing", "gender",
                            "age", "build", "skin_tone", "distinctive_features",
                            "outfit", "accessories", "appearance"
                        ]
                        for k, v in source.attributes.items():
                            if v and any(ak in k.lower() for ak in appearance_keys):
                                appearance_attrs[k] = v
                        if appearance_attrs:
                            attr_items = [f"{k}: {v}" for k, v in appearance_attrs.items()]
                            desc = f"{desc} ({', '.join(attr_items)})"
                    appearance_parts.append(f"Character: {desc}.")
                if appearance_parts:
                    appearance_context = (
                        "[Appearance Context - No Frontal Reference Available]\n"
                        + "\n".join(appearance_parts)
                        + "\nNote: Maintain visual consistency with the described appearance."
                    )
                    prompt_parts.append(appearance_context)
                    print(f"[Pipeline] Appearance context (无脸 character):\n{appearance_context}")

            # Layer 3: 当前 shot 的实体描述
            shot_context = self.parser.build_shot_context(parse_result)
            if shot_context:
                prompt_parts.append(f"[Shot Entities]\n{shot_context}")
                print(f"[Pipeline] Shot context:\n{shot_context}")

            # Layer 4: 当前 shot 的动作描述
            prompt_parts.append(shot.text)

            gen_prompt = "\n\n".join(prompt_parts)

            # ── 保存实际使用的参考图到 used_refs/ 目录 ──────────────────────────
            # 这里记录的是每个 shot 真正输入模型的参考图，而非 selected_refs 中的候选池
            used_refs_shot_dir = os.path.join(
                self.output_dir, "used_refs", f"shot_{shot.shot_id:03d}"
            )
            os.makedirs(used_refs_shot_dir, exist_ok=True)
            for entity_id, ref_paths in reference_used.items():
                for idx, ref_path in enumerate(ref_paths):
                    try:
                        ext = os.path.splitext(ref_path)[1] or ".jpg"
                        dst_name = f"{entity_id}_{idx:02d}{ext}"
                        shutil.copy2(ref_path, os.path.join(used_refs_shot_dir, dst_name))
                    except Exception as e:
                        print(f"[Pipeline] 复制参考图失败 {ref_path}: {e}")
            print(f"[Pipeline] 实际使用的参考图已保存到: {used_refs_shot_dir} ({len(all_ref_paths)} 张)")

            # 保存 prompt 到文件，方便 debug
            # 文件名包含生成模式：shot_001_t2v_prompt.txt 或 shot_001_s2v_prompt.txt
            prompt_path = os.path.join(
                self.output_dir, "prompts", f"shot_{shot.shot_id:03d}_{generation_mode}_prompt.txt"
            )
            shot_seed = self.base_seed + shot.shot_id  # 预计算用于记录
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(f"=== Shot {shot.shot_id} Generation Prompt ===\n")
                f.write(f"Mode: {generation_mode}\n")
                f.write(f"Seed: {shot_seed} (base={self.base_seed})\n")
                f.write(f"Shot type: {'close-up' if is_closeup else ('wide' if is_wide else 'medium')}\n")
                f.write(f"Reference images: {len(all_ref_paths)}\n")
                if all_ref_paths:
                    f.write(f"Reference paths:\n")
                    for rp in all_ref_paths:
                        f.write(f"  - {rp}\n")
                f.write(f"\n{'='*50}\n")
                f.write(f"FINAL PROMPT TO MODEL:\n")
                f.write(f"{'='*50}\n\n")
                f.write(gen_prompt)
            print(f"[Pipeline] Prompt 已保存: {prompt_path}")

            # ── Shot 1 验证：检查是否需要验证循环 ──────────────────────────────
            # 条件：T2V 模式 + 开启验证 + 有人数预期
            needs_verification = (
                generation_mode == "t2v" and
                self.enable_shot1_verification and
                self.shot1_verify_person_count and
                not self.mock_mode and
                parse_result.entity_counts
            )
            # 提取人数预期
            person_count_expectation = None
            if needs_verification:
                for ec in parse_result.entity_counts:
                    if ec.entity_type == "character":
                        person_count_expectation = ec.expected_count
                        break

            # 广播给其他 rank
            broadcast_data = [
                gen_prompt, all_ref_paths, generation_mode,
                needs_verification, person_count_expectation, parse_result
            ]
        else:
            broadcast_data = [None, None, None, False, None, None]

        if self.world_size > 1 and dist.is_initialized():
            dist.broadcast_object_list(broadcast_data, src=0)

        prompt, all_ref_paths, generation_mode, needs_verification, person_count_expectation, parse_result = broadcast_data

        # ── Step 3: 生成视频（可能包含验证循环）───────────────────────────────
        all_ref_images: List[Image.Image] = []
        for p in (all_ref_paths or []):
            try:
                all_ref_images.append(Image.open(p).convert("RGB"))
            except Exception:
                pass

        video_path = os.path.join(
            self.output_dir, "videos", f"shot_{shot.shot_id:03d}.mp4"
        )

        # Shot 1 验证循环
        if needs_verification and person_count_expectation is not None:
            video_path, verification_metadata = self._generate_with_verification(
                shot=shot,
                prompt=prompt,
                all_ref_images=all_ref_images,
                video_path=video_path,
                expected_person_count=person_count_expectation,
            )
        else:
            # 普通生成（无验证）
            shot_seed = self.base_seed + shot.shot_id
            self.generator.config.seed = shot_seed
            if self.is_rank0:
                print(f"[Pipeline] 使用 seed: {shot_seed} (base={self.base_seed}, shot_id={shot.shot_id})")

            self.generator.generate(
                text_prompt=prompt,
                references=all_ref_images,
                output_path=video_path,
            )
            verification_metadata = {"verified": False}

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
            metadata={
                "gen_prompt": prompt,
                "verification": verification_metadata,
            },
        )

    def _generate_with_verification(
        self,
        shot: ShotConfig,
        prompt: str,
        all_ref_images: List[Image.Image],
        video_path: str,
        expected_person_count: int,
    ) -> tuple:
        """
        带验证的生成循环（仅用于 Shot 1 T2V 模式）

        流程:
          1. 生成视频
          2. 检测人数
          3. 如果人数不匹配，换 seed 重试
          4. 如果多次重试仍不匹配，增强 prompt 再试
          5. 最多 max_retries 次，超出则返回最后一次结果并警告

        Returns:
            (video_path, metadata_dict)
        """
        import torch.distributed as dist

        max_retries = self.shot1_max_retries
        current_prompt = prompt
        retry_history = []

        for attempt in range(max_retries + 1):
            # 计算当前 seed
            shot_seed = self.base_seed + shot.shot_id + attempt * 1000
            self.generator.config.seed = shot_seed

            # 第 2 轮及以后使用增强 prompt
            if attempt >= 2 and attempt == 2:
                current_prompt = build_count_enhanced_prompt(prompt, expected_person_count)
                if self.is_rank0:
                    print(f"[Verify] 使用增强 prompt (强调人数={expected_person_count})")

            if self.is_rank0:
                attempt_str = f"[Verify] 尝试 {attempt + 1}/{max_retries + 1}"
                print(f"{attempt_str}: seed={shot_seed}")

            # 生成
            self.generator.generate(
                text_prompt=current_prompt,
                references=all_ref_images,
                output_path=video_path,
            )

            # 只有 rank 0 做验证
            if not self.is_rank0:
                # 非 rank 0 等待验证结果广播
                if self.world_size > 1 and dist.is_initialized():
                    verify_result = [None]
                    dist.broadcast_object_list(verify_result, src=0)
                    passed = verify_result[0]
                    if passed:
                        break
                continue

            # rank 0 执行验证
            expectations = [
                EntityCountExpectation(
                    entity_type="person",
                    expected_count=expected_person_count,
                    description=f"{expected_person_count} person(s)",
                    tolerance=0,
                    is_critical=True,
                )
            ]

            verify_result = self.verifier.verify(
                video_path=video_path,
                expectations=expectations,
                sample_frames=3,
                temp_dir=os.path.join(self.output_dir, "verify_frames", f"shot_{shot.shot_id:03d}"),
            )

            retry_history.append({
                "attempt": attempt + 1,
                "seed": shot_seed,
                "status": verify_result.status.value,
                "expected": verify_result.expected_counts,
                "actual": verify_result.actual_counts,
                "enhanced_prompt": attempt >= 2,
            })

            print(f"[Verify] 结果: {verify_result.status.value} | "
                  f"预期={verify_result.expected_counts} | "
                  f"实际={verify_result.actual_counts}")

            # 广播验证结果
            if self.world_size > 1 and dist.is_initialized():
                dist.broadcast_object_list([verify_result.passed], src=0)

            if verify_result.passed:
                print(f"[Verify] ✅ 验证通过！人数匹配")
                break
            elif attempt < max_retries:
                print(f"[Verify] ❌ 人数不匹配，准备重试...")
            else:
                print(f"[Verify] ⚠️ 达到最大重试次数，使用最后结果")
                print(f"[Verify] 警告: 生成的人数 ({verify_result.actual_counts.get('person', '?')}) "
                      f"与预期 ({expected_person_count}) 不符！")

        metadata = {
            "verified": True,
            "final_attempt": len(retry_history),
            "final_passed": retry_history[-1]["status"] == "passed" if retry_history else False,
            "expected_person_count": expected_person_count,
            "actual_person_count": retry_history[-1]["actual"].get("person", -1) if retry_history else -1,
            "retry_history": retry_history,
        }

        return video_path, metadata

    # ── 生成后 Grounding & 入库 ───────────────────────────────────────────────

    @staticmethod
    def _compute_iou(box1, box2):
        """计算两个 bbox 的 IoU，box 格式: (x1, y1, x2, y2)"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def _cross_entity_dedup(
        self,
        all_ground_results: dict,
        iou_threshold: float = 0.5,
    ) -> dict:
        """
        跨实体 IoU 去重：同一帧中，如果多个实体的检测框高度重叠，
        只保留置信度最高的那个实体的检测结果。

        Args:
            all_ground_results: {entity_id: [GroundingResult, ...]}
            iou_threshold: IoU 阈值，超过则认为是同一个目标

        Returns:
            去重后的 {entity_id: [GroundingResult, ...]}
        """
        # 按帧分组所有检测结果
        frame_to_results = {}  # frame_path -> [(entity_id, GroundingResult), ...]
        for entity_id, results in all_ground_results.items():
            for r in results:
                if r.frame_path not in frame_to_results:
                    frame_to_results[r.frame_path] = []
                frame_to_results[r.frame_path].append((entity_id, r))

        # 对每帧做去重
        kept_results = {eid: [] for eid in all_ground_results}
        removed_count = 0

        for frame_path, detections in frame_to_results.items():
            # 按置信度排序（高→低）
            detections.sort(key=lambda x: x[1].score, reverse=True)

            suppressed = set()  # 被抑制的检测索引
            for i, (eid_i, det_i) in enumerate(detections):
                if i in suppressed:
                    continue
                # 保留这个检测
                kept_results[eid_i].append(det_i)
                # 抑制与它高度重叠的其他实体检测
                for j in range(i + 1, len(detections)):
                    if j in suppressed:
                        continue
                    eid_j, det_j = detections[j]
                    # 只抑制不同实体的重叠框（同一实体的多个框保留，由后续评分筛选）
                    if eid_i != eid_j:
                        iou = self._compute_iou(det_i.bbox, det_j.bbox)
                        if iou > iou_threshold:
                            suppressed.add(j)
                            removed_count += 1
                            print(f"[Pipeline] IoU去重: {eid_j} 的检测被 {eid_i} 抑制 "
                                  f"(IoU={iou:.2f}, frame={os.path.basename(frame_path)})")

        if removed_count > 0:
            print(f"[Pipeline] 跨实体IoU去重: 共移除 {removed_count} 个重叠检测")

        return kept_results

    def _ground_and_register(
        self,
        video_path: str,
        parse_result: ParseResult,
        shot_id: int,
    ) -> dict:
        """
        对已生成的视频做 Grounding，提取每个实体的 crop，
        质量评分后存入 Entity Registry，供后续镜头使用。

        使用 ReferDINO 进行多实体联合检测，利用对比消歧效果更好。
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

        # ── Step 1: 收集需要 grounding 的实体 ──────────────────────────────────
        entities_to_ground = []
        for entity in parse_result.entities:
            # 跳过低优先级实体
            if entity.grounding_priority == "low":
                continue
            # style 类型不做 grounding（风格是抽象概念，无法在帧中定位）
            if entity.type == "style":
                print(f"[Pipeline] 跳过 style 实体 {entity.entity_id}（风格通过 Global Context 处理）")
                continue
            entities_to_ground.append({
                "entity_id": entity.entity_id,
                "text": entity.text_description,
                "type": entity.type,
            })

        if not entities_to_ground:
            print(f"[Pipeline] 无需要 grounding 的实体")
            return {}

        # ── Step 2: 使用 ReferDINO 多实体联合检测 ─────────────────────────────
        # ReferDINO 的优势：多实体同时输入时，利用对比关系消歧，效果更好
        # 例如 "woman in white bee suit" 和 "boy in white bee suit" 同时检测
        print(f"[Pipeline] ReferDINO 多实体联合检测: {[e['entity_id'] for e in entities_to_ground]}")

        multi_result = self.grounder.ground_with_joint_caption(
            frame_paths=frames,
            entities=entities_to_ground,
            output_dir=crops_dir,
            max_results_per_entity=5,
        )

        # 转换为原有格式
        all_ground_results = {}
        for entity_id, results in multi_result.results_by_entity.items():
            if results:
                all_ground_results[entity_id] = results
            else:
                print(f"[Pipeline] {entity_id}: grounding 未找到")

        # ── Step 3: 跨实体 IoU 去重 ───────────────────────────────────────────
        # ReferDINO 的联合检测已经有较好的区分，但仍做一次去重保险
        character_entities = {
            e.entity_id for e in parse_result.entities if e.type == "character"
        }
        char_results = {
            eid: results for eid, results in all_ground_results.items()
            if eid in character_entities
        }
        if len(char_results) > 1:
            # 有多个 character，需要去重
            deduped_char = self._cross_entity_dedup(char_results, iou_threshold=0.5)
            all_ground_results.update(deduped_char)

        # ── Step 4: 质量评分 + 入库 ───────────────────────────────────────────
        grounded: dict = {}

        for entity in parse_result.entities:
            if entity.entity_id not in all_ground_results:
                continue

            ground_results = all_ground_results[entity.entity_id]
            if not ground_results:
                print(f"[Pipeline] {entity.entity_id}: 去重后无有效检测")
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
                # 获取 id_confidence（仅 character 类型有意义，其他类型默认 1.0）
                id_conf = s.id_confidence if entity.type == "character" else 1.0
                entry = ReferenceEntry(
                    entity_id=entity.entity_id,
                    shot_id=shot_id,
                    frame_path="",
                    crop_path=s.crop_path,
                    quality_score=float(s.final_score),
                    source="grounding",
                    id_confidence=float(id_conf),
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
