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
import gc
import torch
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from PIL import Image
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entity_parser.parser import EntityParser, ParseResult, Entity, EntityCountInfo, CloseupLightingAnalysis
from visual_grounding.referdino_grounder import ReferDINOGrounder, extract_frames, MultiEntityGroundingResult
from visual_grounding.reid import ReferenceQualityScorer
from reference_manager.registry import EntityRegistry, ReferenceEntry
from reference_manager.smart_registry import SmartEntityRegistry, RegistryConfig, ReferenceEntry as SmartReferenceEntry
from generator.ref2video import Reference2VideoGenerator, GenerationConfig
from verification.entity_count_verifier import (
    EntityCountVerifier,
    EntityCountExpectation,
    VerificationResult,
    VerificationStatus,
    build_count_enhanced_prompt,
)
from orchestrator.agentic_scheduler import (
    AgenticScheduler,
    ScheduleResult,
    ShotInfo,
    EntityInfo,
    ShotType,
    visualize_schedule,
)
from verification.video_critic import (
    VideoQualityCritic,
    CritiqueResult,
    RepairStrategyGenerator,
    IssueSeverity,
)
# ── 根因分析 & 智能重试 ──
from retry.root_cause_analyzer import (
    RootCauseAnalyzer,
    IssueCategory,
    DiagnosisResult,
)
from retry.smart_retry import (
    SmartRetryExecutor,
    RetryStrategy,
)
# ── 经验记忆系统 ──
from experience.advisor import (
    ExperienceAdvisor,
    ExperienceAdvice,
)
from experience.database import (
    SceneFingerprint,
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
        # ── SmartRegistry 参数 ──
        use_smart_registry: bool = True,          # 是否使用智能 Registry（推荐开启）
        registry_similarity_threshold: float = 0.92,  # CLIP 相似度去重阈值
        registry_max_refs_per_shot: int = 2,      # 每 shot 每实体最多注册几张
        # ── Agentic Scheduling 参数 ──
        enable_dag_scheduling: bool = True,       # 是否启用 DAG 调度优化（推荐开启）
        dag_scheduling_model: str = "claude-sonnet-4-6",  # DAG 调度使用的 LLM
        dag_min_benefit_threshold: float = 0.15,  # DAG 优化最小收益阈值
        # ── Self-Critique 参数 ──
        enable_self_critique: bool = True,        # 是否启用 Self-Critique（默认开启）
        critique_model: str = "claude-sonnet-4-6",  # Critique 使用的 VLM
        critique_pass_threshold: float = 0.7,     # Critique 通过阈值
        critique_max_retries: int = 2,            # Critique 失败后最大重试次数
        critique_sample_frames: int = 5,          # Critique 采样帧数
        # ── 根因分析 & 智能重试参数 ──
        enable_smart_retry: bool = True,          # 是否启用根因分析式重试（默认开启）
        # ── 经验记忆系统参数 ──
        enable_experience_memory: bool = True,    # 是否启用经验记忆（默认开启）
        experience_db_path: str = "",             # 经验数据库路径（空则使用 output_dir/experience.db）
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

        # SmartRegistry 参数
        self.use_smart_registry = use_smart_registry
        self.registry_similarity_threshold = registry_similarity_threshold
        self.registry_max_refs_per_shot = registry_max_refs_per_shot
        self._clip_model = None  # 延迟加载，用于相似度检测

        # ── Agentic Scheduling 参数 ──
        self.enable_dag_scheduling = enable_dag_scheduling
        self.dag_scheduling_model = dag_scheduling_model
        self.dag_min_benefit_threshold = dag_min_benefit_threshold
        self._scheduler = None  # 延迟初始化

        # ── Self-Critique 参数 ──
        self.enable_self_critique = enable_self_critique
        self.critique_model = critique_model
        self.critique_pass_threshold = critique_pass_threshold
        self.critique_max_retries = critique_max_retries
        self.critique_sample_frames = critique_sample_frames
        self._video_critic = None  # 延迟初始化

        # ── 根因分析 & 智能重试参数 ──
        self.enable_smart_retry = enable_smart_retry
        self._root_cause_analyzer = None  # 延迟初始化
        self._smart_retry_executor = None  # 延迟初始化

        # ── 经验记忆系统参数 ──
        self.enable_experience_memory = enable_experience_memory
        self.experience_db_path = experience_db_path or os.path.join(output_dir, "experience.db")
        self._experience_advisor = None  # 延迟初始化

        if self.is_rank0:
            for sub in ["frames", "crops", "videos", "registry", "selected_refs", "prompts", "used_refs"]:
                os.makedirs(os.path.join(output_dir, sub), exist_ok=True)
            # rank 0 等目录创建完再让其他 rank 继续
        if self.world_size > 1 and dist.is_initialized():
            dist.barrier()

        # LLM / Grounding / Registry 只在 rank 0 初始化
        if self.is_rank0:
            self.parser = EntityParser(model=llm_model)

            # ── Registry 初始化（SmartRegistry vs 旧版）──
            registry_db_path = os.path.join(output_dir, "registry", "entities.db")
            if self.use_smart_registry:
                # 使用智能 Registry：自动淘汰低质量/冗余参考图
                smart_config = RegistryConfig(
                    min_quality_to_register=self.min_ref_quality,
                    min_id_confidence=self.min_face_confidence,
                    max_refs_per_entity=self.max_refs_per_entity * 3,  # 允许更多，靠淘汰控制
                    max_refs_per_shot=self.registry_max_refs_per_shot,
                    similarity_threshold=self.registry_similarity_threshold,
                    eviction_quality_threshold=self.min_ref_quality + 0.1,
                    protected_anchor_count=2,
                )
                # 尝试加载 CLIP 模型用于相似度去重
                clip_model = self._load_clip_model_if_available()
                self.registry = SmartEntityRegistry(
                    db_path=registry_db_path,
                    config=smart_config,
                    clip_model=clip_model,
                )
                print(f"[Pipeline] 🧠 SmartRegistry 已启用 | "
                      f"max_per_shot={self.registry_max_refs_per_shot} | "
                      f"similarity_threshold={self.registry_similarity_threshold}")
            else:
                # 使用旧版 Registry（只注册不淘汰）
                self.registry = EntityRegistry(registry_db_path)
                print(f"[Pipeline] 使用旧版 EntityRegistry（无淘汰机制）")
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
            if self.enable_dag_scheduling:
                print(f"[Pipeline] 📊 DAG 调度优化已启用 (model={self.dag_scheduling_model})")
            if self.enable_self_critique:
                print(f"[Pipeline] 🔍 Self-Critique 已启用 (model={self.critique_model}, threshold={self.critique_pass_threshold})")
            if self.enable_smart_retry:
                print(f"[Pipeline] 🎯 根因分析式重试已启用 (max_retries={self.critique_max_retries})")
            if self.enable_experience_memory:
                print(f"[Pipeline] 📚 经验记忆系统已启用 (db={self.experience_db_path})")

    # ── CLIP 模型加载（用于 SmartRegistry 相似度去重）───────────────────────────

    def _load_clip_model_if_available(self):
        """
        尝试加载 CLIP 模型用于参考图相似度检测

        Returns:
            (model, preprocess) tuple 或 None
        """
        try:
            import clip
            import torch
            model, preprocess = clip.load("ViT-B/32", device=self.device)
            model.eval()
            print(f"[Pipeline] CLIP 模型已加载 (用于参考图去重)")
            return (model, preprocess)
        except ImportError:
            print(f"[Pipeline] ⚠️ CLIP 未安装，参考图去重功能禁用 (pip install git+https://github.com/openai/CLIP.git)")
            return None
        except Exception as e:
            print(f"[Pipeline] ⚠️ CLIP 加载失败: {e}，参考图去重功能禁用")
            return None

    # ── VRAM 管理 ─────────────────────────────────────────────────────────────

    def _cleanup_vram_before_retry(self):
        """
        在重试生成前清理 VRAM，防止 OOM

        问题背景：
          - Critique 重试时，上一次 generate() 的中间张量可能仍在 GPU 内存
          - PyTorch 的缓存分配器会保留已分配的内存块
          - 多次重试后累积的碎片可能导致 OOM

        解决方案：
          1. 强制 Python GC 回收不再引用的对象
          2. 清空 PyTorch CUDA 缓存
          3. 同步 CUDA 流，确保所有操作完成
          4. 打印当前 VRAM 使用量（便于调试）
        """
        # Step 1: Python GC
        gc.collect()

        # Step 2: PyTorch CUDA 缓存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

            # Step 3: 同步所有 CUDA 流
            torch.cuda.synchronize()

            # Step 4: 打印 VRAM 状态
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[Pipeline] VRAM 清理完成: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

    # ── Agentic 参考图选择 ─────────────────────────────────────────────────────

    @property
    def scheduler(self):
        """获取 DAG 调度器（延迟初始化）"""
        if self._scheduler is None and self.enable_dag_scheduling:
            try:
                self._scheduler = AgenticScheduler(
                    model=self.dag_scheduling_model,
                    min_benefit_threshold=self.dag_min_benefit_threshold,
                    enable_heuristic_fallback=True,
                    verbose=True,
                )
                print(f"[Pipeline] AgenticScheduler 初始化成功")
            except Exception as e:
                print(f"[Pipeline] ⚠️ AgenticScheduler 初始化失败: {e}")
                print(f"[Pipeline] 回退到线性执行顺序")
                self.enable_dag_scheduling = False
        return self._scheduler

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

    @property
    def video_critic(self):
        """获取视频质量评审专家（延迟初始化）"""
        if self._video_critic is None and self.enable_self_critique:
            try:
                self._video_critic = VideoQualityCritic(
                    model=self.critique_model,
                    pass_threshold=self.critique_pass_threshold,
                    sample_frames=self.critique_sample_frames,
                    verbose=True,
                )
                print(f"[Pipeline] VideoQualityCritic 初始化成功")
            except Exception as e:
                print(f"[Pipeline] ⚠️ VideoQualityCritic 初始化失败: {e}")
                print(f"[Pipeline] Self-Critique 功能已禁用")
                self.enable_self_critique = False
        return self._video_critic

    @property
    def root_cause_analyzer(self):
        """获取根因分析器（延迟初始化）"""
        if self._root_cause_analyzer is None and self.enable_smart_retry:
            try:
                self._root_cause_analyzer = RootCauseAnalyzer(verbose=True)
                print(f"[Pipeline] RootCauseAnalyzer 初始化成功")
            except Exception as e:
                print(f"[Pipeline] ⚠️ RootCauseAnalyzer 初始化失败: {e}")
                self.enable_smart_retry = False
        return self._root_cause_analyzer

    @property
    def smart_retry_executor(self):
        """获取智能重试执行器（延迟初始化）"""
        if self._smart_retry_executor is None and self.enable_smart_retry:
            try:
                self._smart_retry_executor = SmartRetryExecutor(
                    max_retries=self.critique_max_retries,
                    base_seed=self.base_seed,
                    verbose=True,
                )
                print(f"[Pipeline] SmartRetryExecutor 初始化成功")
            except Exception as e:
                print(f"[Pipeline] ⚠️ SmartRetryExecutor 初始化失败: {e}")
                self.enable_smart_retry = False
        return self._smart_retry_executor

    @property
    def experience_advisor(self):
        """获取经验顾问（延迟初始化）"""
        if self._experience_advisor is None and self.enable_experience_memory:
            try:
                self._experience_advisor = ExperienceAdvisor(
                    db_path=self.experience_db_path,
                    min_experiences_for_advice=3,
                    verbose=True,
                )
                print(f"[Pipeline] ExperienceAdvisor 初始化成功")
            except Exception as e:
                print(f"[Pipeline] ⚠️ ExperienceAdvisor 初始化失败: {e}")
                self.enable_experience_memory = False
        return self._experience_advisor

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

        # ── DAG 调度：确定最优执行顺序 ──────────────────────────────────────────
        schedule_result = None
        execution_order = [s.shot_id for s in shots]  # 默认线性顺序

        if self.is_rank0 and self.enable_dag_scheduling and len(shots) > 1:
            schedule_result = self._compute_dag_schedule(shots, global_caption)
            if schedule_result and schedule_result.dag_optimized:
                execution_order = schedule_result.execution_order
                print(f"\n[Pipeline] 📊 DAG 调度优化生效:")
                print(f"           原始顺序: {[s.shot_id for s in shots]}")
                print(f"           执行顺序: {execution_order}")
                print(f"           预期质量提升: +{schedule_result.expected_benefit.get('anchor_quality_improvement', 0):.2f}")

        # 广播执行顺序给其他 rank
        if self.world_size > 1 and dist.is_initialized():
            order_data = [execution_order]
            dist.broadcast_object_list(order_data, src=0)
            execution_order = order_data[0]

        # 建立 shot_id -> ShotConfig 的映射
        shot_map = {s.shot_id: s for s in shots}

        # ── 按 DAG 优化后的顺序执行 ─────────────────────────────────────────────
        results = []
        results_by_id = {}  # 按 shot_id 存储结果，最后按叙事顺序输出

        for exec_idx, shot_id in enumerate(execution_order):
            shot = shot_map.get(shot_id)
            if shot is None:
                print(f"[Pipeline] ⚠️ Shot {shot_id} 不存在，跳过")
                continue

            if self.is_rank0:
                print(f"\n{'='*60}")
                dag_info = ""
                if schedule_result and schedule_result.dag_optimized:
                    narrative_pos = [s.shot_id for s in shots].index(shot_id) + 1
                    dag_info = f" (DAG执行: {exec_idx+1}/{len(execution_order)}, 叙事: {narrative_pos}/{len(shots)})"
                print(f"[Pipeline] ── Shot {shot.shot_id}{dag_info} ──")

            result = self._process_shot(shot)
            if self.is_rank0 and result:
                results_by_id[shot_id] = result

        # ── 按叙事顺序整理结果 ────────────────────────────────────────────────
        if self.is_rank0:
            for shot in shots:
                if shot.shot_id in results_by_id:
                    results.append(results_by_id[shot.shot_id])

            # 保存调度报告
            if schedule_result:
                self._save_schedule_report(schedule_result)

            self._save_report(results)

            # ── 打印经验系统总结 ──────────────────────────────────────────────
            if self.enable_experience_memory and self._experience_advisor:
                try:
                    summary = self._experience_advisor.get_session_summary()
                    if summary["total"] > 0:
                        print(f"\n[Pipeline] 📚 本次 Session 经验总结:")
                        print(f"           生成次数: {summary['total']}")
                        print(f"           成功率: {summary['success_rate']:.1%}")
                        print(f"           平均尝试: {summary['avg_attempts']:.1f}")
                        if summary.get("common_issues"):
                            print(f"           常见问题: {summary['common_issues']}")
                except Exception as e:
                    print(f"[Pipeline] ⚠️ 获取经验总结失败: {e}")

        return results

    def _compute_dag_schedule(
        self,
        shots: List[ShotConfig],
        global_caption: str = "",
    ) -> Optional[ScheduleResult]:
        """
        计算 DAG 调度顺序

        分析 script，预测每个 shot 的 grounding 质量，
        构建执行依赖图，返回优化后的执行顺序。

        Returns:
            ScheduleResult 或 None（调度失败时）
        """
        if self.scheduler is None:
            return None

        print(f"\n[Pipeline] 📊 开始 DAG 调度分析...")

        # Step 1: 预解析所有 shot，获取实体信息
        # 使用一个临时 parser 来避免影响主 parser 的状态
        from entity_parser.parser import EntityParser
        temp_parser = EntityParser(model=self.parser.llm.model)
        if global_caption:
            temp_parser.set_global_caption(global_caption)

        shot_infos = []
        all_entities = {}  # entity_id -> EntityInfo

        for shot in shots:
            parse_result = temp_parser.parse(shot.text, shot.shot_id)

            # 分析镜头类型
            shot_text_lower = shot.text.lower()
            if any(kw in shot_text_lower for kw in ["close-up", "closeup", "close up", "tight shot"]):
                shot_type = ShotType.CLOSEUP
            elif any(kw in shot_text_lower for kw in ["wide shot", "wide angle", "full shot"]):
                shot_type = ShotType.WIDE
            elif any(kw in shot_text_lower for kw in ["establishing", "exterior of"]):
                shot_type = ShotType.ESTABLISHING
            else:
                shot_type = ShotType.MEDIUM

            entity_ids = []
            for entity in parse_result.entities:
                if entity.grounding_priority in ("high", "medium"):
                    entity_ids.append(entity.entity_id)

                    # 记录实体信息
                    if entity.entity_id not in all_entities:
                        all_entities[entity.entity_id] = EntityInfo(
                            entity_id=entity.entity_id,
                            entity_type=entity.type,
                            text_description=entity.text_description,
                            first_appearance_shot=shot.shot_id,
                        )

            shot_infos.append(ShotInfo(
                shot_id=shot.shot_id,
                text=shot.text,
                shot_type=shot_type,
                entities=entity_ids,
            ))

        entity_list = list(all_entities.values())

        print(f"[Pipeline] 预解析完成: {len(shot_infos)} shots, {len(entity_list)} entities")
        for si in shot_infos:
            print(f"           Shot {si.shot_id} [{si.shot_type.value}]: {si.entities}")

        # Step 2: 调用调度器分析
        try:
            schedule = self.scheduler.analyze(shot_infos, entity_list)
            return schedule
        except Exception as e:
            print(f"[Pipeline] ⚠️ DAG 调度分析失败: {e}")
            return None

    def _save_schedule_report(self, schedule: ScheduleResult):
        """保存调度分析报告"""
        report_path = os.path.join(self.output_dir, "schedule_report.json")

        report = {
            "dag_optimized": schedule.dag_optimized,
            "narrative_order": schedule.narrative_order,
            "execution_order": schedule.execution_order,
            "reference_sources": schedule.reference_sources,
            "expected_benefit": schedule.expected_benefit,
            "dependencies": [
                {"from": d[0], "to": d[1], "reason": d[2]}
                for d in schedule.dependencies
            ],
            "quality_matrix": schedule.quality_matrix,
            "reasoning": schedule.reasoning,
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"[Pipeline] 调度报告已保存: {report_path}")

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
                needs_verification, person_count_expectation, parse_result,
                reference_used  # 添加 reference_used 到广播数据
            ]
        else:
            broadcast_data = [None, None, None, False, None, None, {}]

        if self.world_size > 1 and dist.is_initialized():
            dist.broadcast_object_list(broadcast_data, src=0)

        prompt, all_ref_paths, generation_mode, needs_verification, person_count_expectation, parse_result, reference_used = broadcast_data

        # ── Step 2.5: 获取经验建议（生成前）─────────────────────────────────────
        experience_advice = None
        if self.is_rank0 and self.enable_experience_memory and self.experience_advisor and generation_mode == "phantom":
            try:
                # 检测镜头类型
                shot_text_lower = shot.text.lower()
                if any(kw in shot_text_lower for kw in ["close-up", "closeup", "close up"]):
                    shot_type = "closeup"
                elif any(kw in shot_text_lower for kw in ["wide shot", "wide angle", "full shot"]):
                    shot_type = "wide"
                else:
                    shot_type = "medium"

                # 创建场景指纹
                expected_entities = [
                    {"entity_id": e.entity_id, "type": e.type}
                    for e in parse_result.entities
                    if e.grounding_priority in ("high", "medium")
                ]

                fingerprint = self.experience_advisor.create_fingerprint(
                    entities=expected_entities,
                    shot_text=shot.text,
                    shot_type=shot_type,
                )

                # 获取建议
                experience_advice = self.experience_advisor.get_advice(fingerprint)

                if experience_advice and experience_advice.has_suggestions():
                    # 应用建议的参数
                    if experience_advice.suggested_ip_adapter_scale:
                        old_scale = self.generator.config.guide_scale_img
                        self.generator.config.guide_scale_img = experience_advice.suggested_ip_adapter_scale
                        print(f"[Pipeline] 📚 经验建议: ip_adapter_scale {old_scale:.2f} → {experience_advice.suggested_ip_adapter_scale:.2f}")

                    if experience_advice.suggested_steps:
                        old_steps = self.generator.config.num_inference_steps
                        if experience_advice.suggested_steps != old_steps:
                            self.generator.config.num_inference_steps = experience_advice.suggested_steps
                            print(f"[Pipeline] 📚 经验建议: steps {old_steps} → {experience_advice.suggested_steps}")

                    # 打印建议的策略
                    if experience_advice.recommended_strategies:
                        print(f"[Pipeline] 📚 如遇问题，推荐策略: {experience_advice.recommended_strategies}")

                    # 打印预期
                    print(f"[Pipeline] 📚 预期: {experience_advice.expected_attempts:.1f} 次尝试, "
                          f"{experience_advice.expected_success_rate:.0%} 成功率")
            except Exception as e:
                print(f"[Pipeline] ⚠️ 获取经验建议失败: {e}")

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

        # ── Step 3.5: Self-Critique 循环（生成后、grounding 前）─────────────────
        # 【重要】所有 rank 都必须参与 Critique 循环，因为重试时需要分布式推理
        # VLM 分析只在 rank 0 上做，结果通过 broadcast 同步给其他 rank
        critique_metadata = {"critique_enabled": False}
        should_critique = self.enable_self_critique and not self.mock_mode and all_ref_paths and generation_mode == "phantom"

        if should_critique:
            video_path, critique_metadata = self._generate_with_critique_distributed(
                shot=shot,
                prompt=prompt,
                all_ref_images=all_ref_images,
                all_ref_paths=all_ref_paths,
                video_path=video_path,
                parse_result=parse_result,
                reference_used=reference_used,
            )
        elif self.is_rank0 and self.enable_self_critique and not self.mock_mode:
            print(f"[Pipeline] Self-Critique 跳过: {'无参考图' if not all_ref_paths else 'T2V模式'}")

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
                "critique": critique_metadata,
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

            # 【关键修复】重试前清理 VRAM（attempt > 0 时）
            if attempt > 0:
                self._cleanup_vram_before_retry()

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

    # ── Self-Critique 循环（分布式版本）─────────────────────────────────────────

    def _generate_with_critique_distributed(
        self,
        shot: ShotConfig,
        prompt: str,
        all_ref_images: List[Image.Image],
        all_ref_paths: List[str],
        video_path: str,
        parse_result: ParseResult,
        reference_used: dict,
    ) -> tuple:
        """
        分布式 Self-Critique 循环 - 所有 rank 都参与

        关键修复：
          - VLM 分析只在 rank 0 上执行（避免重复 API 调用）
          - 重试决策通过 broadcast 同步给所有 rank
          - generator.generate() 由所有 rank 一起执行（NCCL 同步）

        这样可以避免 NCCL 超时死锁。
        """
        import torch.distributed as dist
        import json
        import shutil
        from datetime import datetime

        if self.is_rank0:
            print(f"\n[Critique] ── 开始 Self-Critique 循环 (Shot {shot.shot_id}) ──")

        critique_metadata = {"critique_enabled": True}
        final_video_path = video_path
        best_score = 0.0
        best_video_path = video_path
        critique_history = []

        # ── 创建 Critique 调试目录 ──────────────────────────────────────────────
        # 目录结构: output_dir/critique_debug/shot_XXX/
        #   ├── attempt_1/
        #   │   ├── video.mp4              # 该次尝试的视频
        #   │   ├── critique_result.json   # VLM 分析结果
        #   │   ├── issues.txt             # 问题列表（人类可读）
        #   │   └── params.json            # 生成参数
        #   ├── attempt_2/
        #   │   └── ...
        #   ├── summary.json               # 整体汇总
        #   └── critique.log               # 完整日志
        critique_debug_dir = None
        critique_log_file = None
        if self.is_rank0:
            critique_debug_dir = os.path.join(
                self.output_dir, "critique_debug", f"shot_{shot.shot_id:03d}"
            )
            os.makedirs(critique_debug_dir, exist_ok=True)

            # ── 保存参考图到 ref_images/ 子目录 ────────────────────────────────
            if reference_used:
                ref_images_dir = os.path.join(critique_debug_dir, "ref_images")
                os.makedirs(ref_images_dir, exist_ok=True)
                for entity_id, ref_paths in reference_used.items():
                    for idx, ref_path in enumerate(ref_paths):
                        try:
                            ext = os.path.splitext(ref_path)[1] or ".jpg"
                            dst_name = f"{entity_id}_{idx:02d}{ext}"
                            shutil.copy2(ref_path, os.path.join(ref_images_dir, dst_name))
                        except Exception as e:
                            print(f"[Critique] 复制参考图失败 {ref_path}: {e}")

            critique_log_file = open(
                os.path.join(critique_debug_dir, "critique.log"), "w", encoding="utf-8"
            )
            critique_log_file.write(f"=== Self-Critique Debug Log ===\n")
            critique_log_file.write(f"Shot: {shot.shot_id}\n")
            critique_log_file.write(f"Start Time: {datetime.now().isoformat()}\n")
            critique_log_file.write(f"Shot Text: {shot.text}\n")
            critique_log_file.write(f"Reference Images: {list(reference_used.keys())}\n")
            for entity_id, ref_paths in reference_used.items():
                critique_log_file.write(f"  - {entity_id}: {ref_paths}\n")
            critique_log_file.write(f"{'='*60}\n\n")

        # 准备实体信息（所有 rank 都需要）
        expected_entities = [
            {
                "entity_id": e.entity_id,
                "type": e.type,
                "text_description": e.text_description,
            }
            for e in parse_result.entities
            if e.grounding_priority in ("high", "medium")
        ]
        entity_counts = [
            {"entity_type": ec.entity_type, "expected_count": ec.expected_count}
            for ec in parse_result.entity_counts
        ] if parse_result.entity_counts else None

        current_params = {
            "ip_adapter_scale": self.generator.config.guide_scale_img,
            "guide_scale_text": self.generator.config.guide_scale_text,
            "num_inference_steps": self.generator.config.num_inference_steps,
            "seed": self.generator.config.seed,
        }

        for attempt in range(self.critique_max_retries + 1):
            # ── Step 1: Rank 0 执行 VLM Critique ──
            if self.is_rank0:
                print(f"[Critique] 尝试 {attempt + 1}/{self.critique_max_retries + 1}: 分析视频...")
                critique_result = self.video_critic.critique(
                    video_path=final_video_path,
                    reference_images=reference_used,
                    expected_entities=expected_entities,
                    shot_text=shot.text,
                    entity_counts=entity_counts,
                )
                score = critique_result.overall_score
                passed = critique_result.passed

                # ── 保存该次尝试的调试信息 ──────────────────────────────────────
                attempt_dir = os.path.join(critique_debug_dir, f"attempt_{attempt + 1}")
                os.makedirs(attempt_dir, exist_ok=True)

                # 1. 复制当前视频到调试目录
                try:
                    video_ext = os.path.splitext(final_video_path)[1] or ".mp4"
                    shutil.copy2(final_video_path, os.path.join(attempt_dir, f"video{video_ext}"))
                except Exception as e:
                    critique_log_file.write(f"[WARN] 复制视频失败: {e}\n")

                # 2. 保存 Critique 结果为 JSON
                critique_result_dict = {
                    "overall_score": score,
                    "passed": passed,
                    "threshold": critique_result.threshold if hasattr(critique_result, 'threshold') else 0.7,
                    "num_issues": len(critique_result.issues),
                    "issues": [
                        {
                            "severity": issue.severity.value if hasattr(issue.severity, 'value') else str(issue.severity),
                            "category": issue.category if hasattr(issue, 'category') else "unknown",
                            "description": issue.description,
                            "entity_id": issue.entity_id if hasattr(issue, 'entity_id') else None,
                        }
                        for issue in critique_result.issues
                    ],
                    "repair_suggestions": [
                        {
                            "action": s.action if hasattr(s, 'action') else str(s),
                            "details": s.details if hasattr(s, 'details') else "",
                        }
                        for s in (critique_result.repair_suggestions if hasattr(critique_result, 'repair_suggestions') else [])
                    ],
                }
                with open(os.path.join(attempt_dir, "critique_result.json"), "w", encoding="utf-8") as f:
                    json.dump(critique_result_dict, f, ensure_ascii=False, indent=2)

                # 3. 保存人类可读的问题列表
                with open(os.path.join(attempt_dir, "issues.txt"), "w", encoding="utf-8") as f:
                    f.write(f"=== Critique Issues (Attempt {attempt + 1}) ===\n")
                    f.write(f"Score: {score:.2f} | Passed: {passed}\n")
                    f.write(f"{'='*50}\n\n")
                    severity_icons = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
                    for i, issue in enumerate(critique_result.issues, 1):
                        sev = issue.severity.value if hasattr(issue.severity, 'value') else str(issue.severity)
                        icon = severity_icons.get(sev, "⚪")
                        f.write(f"{i}. {icon} [{sev.upper()}] {issue.description}\n\n")
                    if critique_result.repair_suggestions if hasattr(critique_result, 'repair_suggestions') else []:
                        f.write(f"\n{'='*50}\n")
                        f.write("=== Repair Suggestions ===\n")
                        for s in critique_result.repair_suggestions:
                            action = s.action if hasattr(s, 'action') else str(s)
                            details = s.details if hasattr(s, 'details') else ""
                            f.write(f"- {action}: {details}\n")

                # 4. 保存当前参数
                with open(os.path.join(attempt_dir, "params.json"), "w", encoding="utf-8") as f:
                    json.dump(current_params, f, ensure_ascii=False, indent=2)

                # 5. 写入日志
                critique_log_file.write(f"\n--- Attempt {attempt + 1} ---\n")
                critique_log_file.write(f"Time: {datetime.now().isoformat()}\n")
                critique_log_file.write(f"Video: {final_video_path}\n")
                critique_log_file.write(f"Score: {score:.2f} | Passed: {passed}\n")
                critique_log_file.write(f"Params: {current_params}\n")
                critique_log_file.write(f"Issues ({len(critique_result.issues)}):\n")
                for issue in critique_result.issues:
                    sev = issue.severity.value if hasattr(issue.severity, 'value') else str(issue.severity)
                    critique_log_file.write(f"  [{sev}] {issue.description}\n")
                critique_log_file.flush()

                # 记录历史
                critique_history.append({
                    "attempt": attempt + 1,
                    "score": score,
                    "passed": passed,
                    "num_issues": len(critique_result.issues),
                    "critical_issues": len(critique_result.critical_issues),
                    "high_issues": len(critique_result.high_issues),
                    "params": current_params.copy(),
                    "debug_dir": attempt_dir,  # 记录调试目录路径
                })

                # 更新最佳结果
                if score > best_score:
                    best_score = score
                    best_video_path = final_video_path

                # 决定是否需要重试
                need_retry = not passed and attempt < self.critique_max_retries
                retry_params = None

                if passed:
                    print(f"[Critique] ✅ 通过！分数: {score:.2f}")
                    critique_log_file.write(f"Result: ✅ PASSED\n")
                elif not need_retry:
                    print(f"[Critique] ⚠️ 达到最大重试次数，使用最佳结果 (score={best_score:.2f})")
                    critique_log_file.write(f"Result: ⚠️ MAX RETRIES REACHED, using best (score={best_score:.2f})\n")
                    final_video_path = best_video_path
                else:
                    # 生成修复参数
                    print(f"[Critique] ❌ 未通过 (score={score:.2f})，准备修复...")
                    critique_log_file.write(f"Result: ❌ FAILED, preparing retry...\n")
                    for issue in critique_result.issues[:3]:
                        icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(
                            issue.severity.value, "⚪"
                        )
                        print(f"[Critique]    {icon} {issue.description}")

                    # ── 根因分析 & 智能重试（新增）──────────────────────────────────
                    if self.enable_smart_retry and self.root_cause_analyzer:
                        # Step 1: 诊断问题根因
                        diagnosis = self.root_cause_analyzer.diagnose(critique_result)

                        # Step 2: 基于根因选择重试策略
                        if self.smart_retry_executor:
                            # 记录上一次分数
                            if attempt == 0:
                                self.smart_retry_executor.reset()  # 新 shot 开始时重置
                            self.smart_retry_executor.record_result(score, passed)

                            # 获取当前问题类别用于策略选择
                            current_issues = [c.category.value for c in diagnosis.all_causes] if diagnosis.all_causes else []

                            # 选择策略
                            strategy = self.smart_retry_executor.select_strategy(
                                diagnosis=diagnosis,
                                attempt=attempt + 1,
                                current_score=score,
                            )

                            if strategy:
                                # 准备实体数量信息
                                entity_count_dict = {}
                                if entity_counts:
                                    for ec in entity_counts:
                                        entity_count_dict[ec["entity_type"]] = ec["expected_count"]

                                # 应用策略
                                retry_params, new_prompt = self.smart_retry_executor.apply_strategy(
                                    strategy=strategy,
                                    current_params=current_params,
                                    current_prompt=prompt,
                                    shot_id=shot.shot_id,
                                    entity_counts=entity_count_dict,
                                )

                                # 保存策略信息到调试目录
                                strategy_info = {
                                    "target_category": strategy.target_category.value,
                                    "actions": [a.value for a in strategy.actions],
                                    "reasoning": strategy.reasoning,
                                    "diagnosis_summary": diagnosis.diagnosis_summary,
                                }
                                with open(os.path.join(attempt_dir, "strategy.json"), "w", encoding="utf-8") as f:
                                    json.dump(strategy_info, f, ensure_ascii=False, indent=2)
                                critique_log_file.write(f"Smart Retry Strategy: {strategy.target_category.value}\n")
                                critique_log_file.write(f"Actions: {[a.value for a in strategy.actions]}\n")
                            else:
                                # 无可用策略，回退到原有逻辑
                                repair_generator = RepairStrategyGenerator()
                                retry_params = repair_generator.generate_repair_params(
                                    critique_result=critique_result,
                                    current_params=current_params,
                                )
                        else:
                            # SmartRetryExecutor 不可用，回退
                            repair_generator = RepairStrategyGenerator()
                            retry_params = repair_generator.generate_repair_params(
                                critique_result=critique_result,
                                current_params=current_params,
                            )
                    else:
                        # 原有逻辑：使用 RepairStrategyGenerator
                        repair_generator = RepairStrategyGenerator()
                        retry_params = repair_generator.generate_repair_params(
                            critique_result=critique_result,
                            current_params=current_params,
                        )

                    if retry_params.get("force_t2v"):
                        print(f"[Critique] 建议切换到 T2V 模式，但本次保留 S2V，仅调整参数")
                        critique_log_file.write(f"Note: T2V suggested but keeping S2V\n")
                        del retry_params["force_t2v"]

                    # 确保有新 seed
                    if retry_params.get("seed") == current_params.get("seed"):
                        import random
                        retry_params["seed"] = self.base_seed + shot.shot_id + (attempt + 1) * 1000 + random.randint(0, 999)

                    # 保存修复参数
                    with open(os.path.join(attempt_dir, "retry_params.json"), "w", encoding="utf-8") as f:
                        json.dump(retry_params, f, ensure_ascii=False, indent=2)
                    critique_log_file.write(f"Retry params: {retry_params}\n")

                # 准备广播数据: [need_retry, retry_params, final_video_path, best_score]
                broadcast_data = [need_retry, retry_params, final_video_path, best_score]
            else:
                broadcast_data = [None, None, None, None]

            # ── Step 2: 广播重试决策给所有 rank ──
            if self.world_size > 1 and dist.is_initialized():
                dist.broadcast_object_list(broadcast_data, src=0)

            need_retry, retry_params, final_video_path, best_score = broadcast_data

            # 不需要重试，退出循环
            if not need_retry:
                break

            # ── Step 3: 所有 rank 一起重新生成视频 ──
            if retry_params:
                # 更新生成参数（所有 rank 都需要更新）
                params_changed = []
                if retry_params.get("ip_adapter_scale") != current_params.get("ip_adapter_scale"):
                    old_val = current_params.get("ip_adapter_scale", 0)
                    new_val = retry_params["ip_adapter_scale"]
                    self.generator.config.guide_scale_img = new_val
                    params_changed.append(f"ip_adapter_scale: {old_val:.2f} → {new_val:.2f}")

                # 步数修改：设置上限为 50（Phantom/Wan 使用 UniPC solver，超过 50 步边际收益递减）
                if retry_params.get("num_inference_steps") != current_params.get("num_inference_steps"):
                    old_val = current_params.get("num_inference_steps", 50)
                    new_val = retry_params["num_inference_steps"]
                    max_steps = 50  # 基于 Phantom 官方默认值，UniPC solver 下 50 步足够
                    if new_val > max_steps:
                        if self.is_rank0:
                            print(f"[Critique] 步数建议 {new_val} 超过上限，限制为 {max_steps}")
                        new_val = max_steps
                    if new_val != old_val:
                        self.generator.config.num_inference_steps = new_val
                        params_changed.append(f"steps: {old_val} → {new_val}")

                if retry_params.get("seed") != current_params.get("seed"):
                    new_seed = retry_params["seed"]
                    self.generator.config.seed = new_seed
                    params_changed.append(f"seed → {new_seed}")

                current_params = retry_params

                if self.is_rank0 and params_changed:
                    print(f"[Critique] 参数调整: {', '.join(params_changed)}")
                    critique_log_file.write(f"Params changed: {', '.join(params_changed)}\n")

            retry_video_path = os.path.join(
                self.output_dir, "videos",
                f"shot_{shot.shot_id:03d}_critique_retry{attempt + 1}.mp4"
            )

            # 清理 VRAM（所有 rank 都执行）
            self._cleanup_vram_before_retry()

            if self.is_rank0:
                print(f"[Critique] 重新生成视频...")
                critique_log_file.write(f"Regenerating video: {retry_video_path}\n")
                critique_log_file.flush()

            try:
                # 【关键】所有 rank 一起调用 generate()，保持 NCCL 同步
                self.generator.generate(
                    text_prompt=prompt,
                    references=all_ref_images,
                    output_path=retry_video_path,
                )
                final_video_path = retry_video_path
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                if self.is_rank0:
                    print(f"[Critique] ❌ 生成失败: {e}")
                    print(f"[Critique] 终止重试，使用最佳结果 (score={best_score:.2f})")
                    critique_log_file.write(f"ERROR: Generation failed - {e}\n")
                    critique_history.append({
                        "attempt": attempt + 1,
                        "score": 0,
                        "passed": False,
                        "error": str(e),
                    })
                self._cleanup_vram_before_retry()
                final_video_path = best_video_path
                break

        # 构建元数据
        critique_metadata = {
            "critique_enabled": True,
            "final_attempt": len(critique_history),
            "final_score": critique_history[-1]["score"] if critique_history else 0,
            "final_passed": critique_history[-1].get("passed", False) if critique_history else False,
            "best_score": best_score,
            "critique_history": critique_history,
            "debug_dir": critique_debug_dir,  # 添加调试目录到元数据
        }

        # ── 保存汇总信息并关闭日志 ────────────────────────────────────────────
        if self.is_rank0 and critique_debug_dir:
            # 保存汇总 JSON
            summary = {
                "shot_id": shot.shot_id,
                "shot_text": shot.text,
                "total_attempts": len(critique_history),
                "final_score": best_score,
                "final_passed": critique_history[-1].get("passed", False) if critique_history else False,
                "final_video": final_video_path,
                "best_video": best_video_path,
                "reference_images": {k: v for k, v in reference_used.items()},
                "critique_history": critique_history,
            }
            with open(os.path.join(critique_debug_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            # 关闭日志文件
            critique_log_file.write(f"\n{'='*60}\n")
            critique_log_file.write(f"=== Critique Complete ===\n")
            critique_log_file.write(f"End Time: {datetime.now().isoformat()}\n")
            critique_log_file.write(f"Total Attempts: {len(critique_history)}\n")
            critique_log_file.write(f"Final Score: {best_score:.2f}\n")
            critique_log_file.write(f"Final Video: {final_video_path}\n")
            critique_log_file.close()

            print(f"[Critique] 调试信息已保存到: {critique_debug_dir}")

        # ── 记录生成经验（新增）─────────────────────────────────────────────────
        if self.is_rank0 and self.enable_experience_memory and self.experience_advisor:
            try:
                # 检测镜头类型
                shot_text_lower = shot.text.lower()
                if any(kw in shot_text_lower for kw in ["close-up", "closeup", "close up"]):
                    shot_type = "closeup"
                elif any(kw in shot_text_lower for kw in ["wide shot", "wide angle", "full shot"]):
                    shot_type = "wide"
                else:
                    shot_type = "medium"

                # 创建场景指纹
                fingerprint = self.experience_advisor.create_fingerprint(
                    entities=expected_entities,
                    shot_text=shot.text,
                    shot_type=shot_type,
                )

                # 收集遇到的问题类别
                encountered_issues = []
                if self.enable_smart_retry and self._root_cause_analyzer and critique_history:
                    # 从历史中提取问题类别
                    for hist in critique_history:
                        if hist.get("num_issues", 0) > 0:
                            # 尝试从 diagnosis 获取类别（如果保存了的话）
                            pass
                    # 简化：直接记录是否有 critical/high 问题
                    for hist in critique_history:
                        if hist.get("critical_issues", 0) > 0:
                            encountered_issues.append("identity")  # 严重问题通常是身份相关
                        if hist.get("high_issues", 0) > 0:
                            encountered_issues.append("quality")
                    encountered_issues = list(set(encountered_issues))

                # 收集使用的策略
                strategies_used = []
                if self.enable_smart_retry and self._smart_retry_executor:
                    for strategy in self._smart_retry_executor.strategy_history:
                        # 判断策略是否成功（下一次分数是否提升）
                        idx = self._smart_retry_executor.strategy_history.index(strategy)
                        if idx + 1 < len(self._smart_retry_executor.score_history):
                            prev_score = self._smart_retry_executor.score_history[idx]
                            next_score = self._smart_retry_executor.score_history[idx + 1]
                            succeeded = next_score > prev_score
                        else:
                            succeeded = critique_history[-1].get("passed", False) if critique_history else False
                        strategies_used.append((strategy.target_category.value, succeeded))

                # 记录经验
                final_passed = critique_history[-1].get("passed", False) if critique_history else False
                self.experience_advisor.record_generation(
                    fingerprint=fingerprint,
                    generation_mode="phantom",  # 只有 S2V 模式才会进入 critique 循环
                    params=current_params,
                    attempts=len(critique_history),
                    issues=encountered_issues,
                    strategies_used=strategies_used,
                    final_score=best_score,
                    success=final_passed,
                )
                print(f"[Critique] 📚 经验已记录 (fingerprint={fingerprint.to_hash()})")
            except Exception as e:
                print(f"[Critique] ⚠️ 记录经验失败: {e}")

        if self.is_rank0:
            print(f"[Critique] ── Self-Critique 完成 (最终分数: {best_score:.2f}) ──\n")

        return final_video_path, critique_metadata

    # ── Self-Critique 循环（原版，保留兼容性）────────────────────────────────────

    def _generate_with_critique(
        self,
        shot: ShotConfig,
        prompt: str,
        all_ref_images: List[Image.Image],
        all_ref_paths: List[str],
        video_path: str,
        parse_result: ParseResult,
        reference_used: dict,
    ) -> tuple:
        """
        带 Self-Critique 的生成循环

        流程:
          1. 视频已生成（首次生成在调用此函数前完成）
          2. VLM Critique 分析视频与参考图的一致性
          3. 如果发现严重问题，根据建议调整参数重新生成
          4. 最多重试 critique_max_retries 次

        Args:
            shot: 当前镜头配置
            prompt: 生成 prompt
            all_ref_images: 参考图 PIL 列表
            all_ref_paths: 参考图路径列表
            video_path: 已生成的视频路径
            parse_result: 实体解析结果
            reference_used: 使用的参考图 {entity_id: [paths]}

        Returns:
            (final_video_path, critique_metadata)
        """
        print(f"\n[Critique] ── 开始 Self-Critique 循环 (Shot {shot.shot_id}) ──")

        critique_history = []
        repair_generator = RepairStrategyGenerator()

        # 准备实体信息
        expected_entities = [
            {
                "entity_id": e.entity_id,
                "type": e.type,
                "text_description": e.text_description,
            }
            for e in parse_result.entities
            if e.grounding_priority in ("high", "medium")
        ]

        # 准备实体数量预期
        entity_counts = [
            {
                "entity_type": ec.entity_type,
                "expected_count": ec.expected_count,
            }
            for ec in parse_result.entity_counts
        ] if parse_result.entity_counts else None

        # 当前生成参数（用于修复策略）
        current_params = {
            "ip_adapter_scale": self.generator.config.guide_scale_img,
            "guide_scale_text": self.generator.config.guide_scale_text,
            "num_inference_steps": self.generator.config.num_inference_steps,
            "seed": self.generator.config.seed,
        }

        final_video_path = video_path
        best_score = 0.0
        best_video_path = video_path

        for attempt in range(self.critique_max_retries + 1):
            # Step 1: 调用 VLM Critique
            print(f"[Critique] 尝试 {attempt + 1}/{self.critique_max_retries + 1}: 分析视频...")

            critique_result = self.video_critic.critique(
                video_path=final_video_path,
                reference_images=reference_used,
                expected_entities=expected_entities,
                shot_text=shot.text,
                entity_counts=entity_counts,
            )

            # 记录历史
            critique_history.append({
                "attempt": attempt + 1,
                "score": critique_result.overall_score,
                "passed": critique_result.passed,
                "num_issues": len(critique_result.issues),
                "critical_issues": len(critique_result.critical_issues),
                "high_issues": len(critique_result.high_issues),
                "params": current_params.copy(),
                "suggestions": [
                    {"action": s.action, "target": s.target, "detail": s.detail}
                    for s in critique_result.suggestions[:3]
                ],
            })

            # 记录最佳结果
            if critique_result.overall_score > best_score:
                best_score = critique_result.overall_score
                best_video_path = final_video_path

            # Step 2: 判断是否通过
            if critique_result.passed:
                print(f"[Critique] ✅ 通过！分数: {critique_result.overall_score:.2f}")
                break

            # Step 3: 未通过，检查是否还有重试机会
            if attempt >= self.critique_max_retries:
                print(f"[Critique] ⚠️ 达到最大重试次数，使用最佳结果 (score={best_score:.2f})")
                final_video_path = best_video_path
                break

            # Step 4: 根据 Critique 建议调整参数
            print(f"[Critique] ❌ 未通过 (score={critique_result.overall_score:.2f})，准备修复...")

            # 打印主要问题
            for issue in critique_result.issues[:3]:
                icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(
                    issue.severity.value, "⚪"
                )
                print(f"[Critique]    {icon} {issue.description}")

            # 生成修复参数
            new_params = repair_generator.generate_repair_params(
                critique_result=critique_result,
                current_params=current_params,
            )

            # 检查是否需要强制切换 T2V
            if new_params.get("force_t2v"):
                print(f"[Critique] 建议切换到 T2V 模式，但本次保留 S2V，仅调整参数")
                del new_params["force_t2v"]

            # 更新参数
            params_changed = []
            if new_params.get("ip_adapter_scale") != current_params.get("ip_adapter_scale"):
                old_val = current_params.get("ip_adapter_scale", 0)
                new_val = new_params["ip_adapter_scale"]
                self.generator.config.guide_scale_img = new_val
                params_changed.append(f"ip_adapter_scale: {old_val:.2f} → {new_val:.2f}")

            if new_params.get("num_inference_steps") != current_params.get("num_inference_steps"):
                old_val = current_params.get("num_inference_steps", 50)
                new_val = new_params["num_inference_steps"]
                self.generator.config.num_inference_steps = new_val
                params_changed.append(f"steps: {old_val} → {new_val}")

            # 换 seed
            if new_params.get("seed") != current_params.get("seed"):
                new_seed = new_params["seed"]
                self.generator.config.seed = new_seed
                params_changed.append(f"seed → {new_seed}")
            else:
                # 即使没有建议换 seed，也自动换一个以增加多样性
                import random
                new_seed = self.base_seed + shot.shot_id + (attempt + 1) * 1000 + random.randint(0, 999)
                self.generator.config.seed = new_seed
                new_params["seed"] = new_seed
                params_changed.append(f"seed → {new_seed} (auto)")

            current_params = new_params

            if params_changed:
                print(f"[Critique] 参数调整: {', '.join(params_changed)}")

            # Step 5: 重新生成视频
            retry_video_path = os.path.join(
                self.output_dir, "videos",
                f"shot_{shot.shot_id:03d}_critique_retry{attempt + 1}.mp4"
            )

            # 【关键修复】重新生成前清理 VRAM，防止 OOM
            # 问题：上一次生成的中间张量可能仍在 GPU 内存中
            # 解决：强制 GC + empty_cache，确保 VRAM 可用
            self._cleanup_vram_before_retry()

            print(f"[Critique] 重新生成视频...")
            try:
                self.generator.generate(
                    text_prompt=prompt,
                    references=all_ref_images,
                    output_path=retry_video_path,
                )
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                # OOM 或 CUDA 错误时，优雅处理而非死锁
                print(f"[Critique] ❌ 生成失败 (可能是 OOM): {e}")
                print(f"[Critique] 终止重试，使用最佳结果 (score={best_score:.2f})")
                # 尝试再次清理 VRAM
                self._cleanup_vram_before_retry()
                final_video_path = best_video_path
                critique_history.append({
                    "attempt": attempt + 1,
                    "score": 0,
                    "passed": False,
                    "num_issues": -1,
                    "critical_issues": -1,
                    "high_issues": -1,
                    "params": current_params.copy(),
                    "error": str(e),
                })
                break

            final_video_path = retry_video_path

        # 构建元数据
        critique_metadata = {
            "critique_enabled": True,
            "final_attempt": len(critique_history),
            "final_score": critique_history[-1]["score"] if critique_history else 0,
            "final_passed": critique_history[-1]["passed"] if critique_history else False,
            "best_score": best_score,
            "critique_history": critique_history,
        }

        # 如果最终使用的不是最佳视频，复制最佳视频为最终输出
        if final_video_path != best_video_path and best_score > critique_history[-1]["score"]:
            import shutil
            final_output = os.path.join(
                self.output_dir, "videos", f"shot_{shot.shot_id:03d}.mp4"
            )
            shutil.copy2(best_video_path, final_output)
            final_video_path = final_output
            print(f"[Critique] 使用最佳结果 (score={best_score:.2f})")

        print(f"[Critique] ── Self-Critique 完成 (最终分数: {best_score:.2f}) ──\n")

        return final_video_path, critique_metadata

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

                # 根据 Registry 类型选择正确的 ReferenceEntry 类
                if self.use_smart_registry:
                    entry = SmartReferenceEntry(
                        entity_id=entity.entity_id,
                        shot_id=shot_id,
                        frame_path="",
                        crop_path=s.crop_path,
                        quality_score=float(s.final_score),
                        source="grounding",
                        id_confidence=float(id_conf),
                    )
                else:
                    entry = ReferenceEntry(
                        entity_id=entity.entity_id,
                        shot_id=shot_id,
                        frame_path="",
                        crop_path=s.crop_path,
                        quality_score=float(s.final_score),
                        source="grounding",
                        id_confidence=float(id_conf),
                    )

                # SmartRegistry 返回 (success, reason)，旧版 Registry 返回 None
                result = self.registry.register(entity.entity_id, entry)
                if isinstance(result, tuple):
                    success, reason = result
                    if not success:
                        print(f"[Pipeline] {entity.entity_id}: 注册被拒绝 - {reason}")
                        continue
                registered_paths.append(s.crop_path)

                # 复制到 selected_refs/{entity_id}/shot{N}_score{X:.2f}.jpg
                ext = os.path.splitext(s.crop_path)[1] or ".jpg"
                dst_name = f"shot{shot_id:03d}_score{s.final_score:.2f}{ext}"
                shutil.copy2(s.crop_path, os.path.join(selected_dir, dst_name))

            grounded[entity.entity_id] = registered_paths
            if registered_paths:
                print(f"[Pipeline] {entity.entity_id}: "
                      f"入库 {len(registered_paths)} 张 "
                      f"(best score={good[0].final_score:.3f})")

        # ── Step 5: 定期淘汰审计（SmartRegistry 专属）──────────────────────────
        if self.use_smart_registry and hasattr(self.registry, 'run_eviction_audit'):
            # 每 5 个 shot 运行一次淘汰审计
            if shot_id % 5 == 0 and shot_id > 0:
                eviction_results = self.registry.run_eviction_audit()
                if eviction_results:
                    total_evicted = sum(eviction_results.values())
                    print(f"[Pipeline] 🗑️ 淘汰审计完成: 共淘汰 {total_evicted} 张冗余参考图")
                    for eid, count in eviction_results.items():
                        print(f"           {eid}: {count} 张")

        return grounded

    # ── 报告 ──────────────────────────────────────────────────────────────────

    def _save_report(self, results: List[ShotResult]):
        report_path = os.path.join(self.output_dir, "pipeline_report.json")

        # 基础统计
        registry_stats = self.registry.stats()

        # SmartRegistry 额外统计：淘汰日志
        eviction_summary = None
        if self.use_smart_registry and hasattr(self.registry, 'get_eviction_log'):
            eviction_log = self.registry.get_eviction_log(limit=50)
            eviction_summary = {
                "total_evicted": registry_stats.get("evicted_references", 0),
                "by_reason": registry_stats.get("eviction_by_reason", {}),
                "recent_evictions": [
                    {
                        "entity_id": e.entity_id,
                        "crop_path": e.crop_path,
                        "reason": e.reason,
                        "quality_score": e.quality_score,
                    }
                    for e in eviction_log[:10]  # 最近 10 条
                ],
            }

        report = {
            "total_shots": len(results),
            "registry_stats": registry_stats,
            "eviction_summary": eviction_summary,
            "smart_registry_enabled": self.use_smart_registry,
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

        # 打印 SmartRegistry 摘要
        if eviction_summary and eviction_summary["total_evicted"] > 0:
            print(f"[Pipeline] 📊 SmartRegistry 统计:")
            print(f"           活跃参考图: {registry_stats.get('active_references', 'N/A')}")
            print(f"           已淘汰: {eviction_summary['total_evicted']}")
            print(f"           淘汰原因分布: {eviction_summary['by_reason']}")
