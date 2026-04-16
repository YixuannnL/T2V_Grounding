"""
generator/ref2video.py
职责: 封装 Reference2Video 生成调用

后端选择:
  - "phantom"  : Phantom-Wan（推荐）
                 VAE latent 时序拼接 conditioning，参考图只提供外观，不锁定姿态
                 支持最多 4 张参考图，真正的 ref2video
  - "cogvideo" : CogVideoX-5B-I2V（备用，首帧锁定姿态，慎用）
  - "mock"     : 不调用真实模型，输出彩色占位视频，用于调试 pipeline
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Optional
from PIL import Image


@dataclass
class GenerationConfig:
    num_frames: int = 81          # Wan2.1 要求 4n+1；81帧 ≈ 3.4秒 @24fps
    fps: int = 24
    width: int = 832
    height: int = 480
    num_inference_steps: int = 50
    guide_scale_text: float = 7.5  # 文本对齐强度
    guide_scale_img: float = 5.0   # 参考图外观强度
    seed: int = 42


class Reference2VideoGenerator:
    def __init__(
        self,
        backend: str = "phantom",
        device: str = "cuda",
        config: Optional[GenerationConfig] = None,
        # Phantom 参数
        wan2_t2v_dir: str = "weights/Wan2.1-T2V-14B",
        phantom_ckpt: str = "weights/Phantom-Wan-14B",
        phantom_task: str = "s2v-14B",
        # 多卡参数（torchrun 下自动读取 LOCAL_RANK）
        use_usp: bool = False,
        dit_fsdp: bool = False,
        t5_fsdp: bool = False,
        # CogVideoX 参数（备用）
        cogvideo_model_id: str = "THUDM/CogVideoX1.5-5B-I2V",
    ):
        import torch
        import torch.distributed as dist

        self.backend = backend
        self.config = config or GenerationConfig()
        self.wan2_t2v_dir = wan2_t2v_dir
        self.phantom_ckpt = phantom_ckpt
        self.phantom_task = phantom_task
        self.use_usp = use_usp
        self.dit_fsdp = dit_fsdp
        self.t5_fsdp = t5_fsdp
        self.cogvideo_model_id = cogvideo_model_id
        self._pipeline = None

        # 自动检测分布式环境（torchrun 会设置 LOCAL_RANK）
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.device = torch.device(f"cuda:{local_rank}")

        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(local_rank)
            print(f"[Generator] 分布式初始化完成: rank={self.rank}/{self.world_size}")

        # USP (Ulysses Sequence Parallel) 初始化 — 必须在 dist.init_process_group 之后
        # num_heads=40 可被 2/4/8 整除，8 卡全部做序列并行
        if use_usp and self.world_size > 1:
            from xfuser.core.distributed import (initialize_model_parallel,
                                                 init_distributed_environment)
            init_distributed_environment(rank=self.rank, world_size=self.world_size)
            initialize_model_parallel(
                sequence_parallel_degree=self.world_size,
                ring_degree=1,
                ulysses_degree=self.world_size,
            )
            if self.rank == 0:
                print(f"[Generator] USP 序列并行初始化完成: ulysses={self.world_size}")

    # ── 加载 ──────────────────────────────────────────────────────────────────

    def _load_pipeline(self):
        if self._pipeline is not None:
            return
        if self.backend == "mock":
            self._pipeline = "mock"
        elif self.backend == "phantom":
            self._load_phantom()   # 同时加载 T2V + S2V
        elif self.backend == "cogvideo":
            self._load_cogvideo()
        else:
            raise ValueError(f"不支持的 backend: {self.backend}")

    def _load_phantom(self):
        """
        延迟加载 Phantom，只预先导入模块，不加载权重。
        实际权重在第一次调用 generate() 时按需加载：
          · 无参考图 → 加载 T2V（WanT2V）
          · 有参考图 → 卸载 T2V，加载 S2V（Phantom_Wan_S2V）
        任何时刻 VRAM 里只有一个 14B 模型（~28GB）。
        """
        phantom_repo = "/root/paddlejob/workspace/env_run/output/lyx/Phantom"
        if phantom_repo not in sys.path:
            sys.path.insert(0, phantom_repo)

        try:
            import phantom_wan
            from phantom_wan.configs import WAN_CONFIGS, SIZE_CONFIGS
        except ImportError as e:
            raise ImportError(
                f"找不到 phantom_wan 模块，请确认路径: {phantom_repo}\n{e}"
            )

        # 保存模块引用和配置，推迟实际权重加载
        self._phantom_wan = phantom_wan
        self._phantom_WAN_CONFIGS = WAN_CONFIGS
        self._phantom_SIZE_CONFIGS = SIZE_CONFIGS
        self._t2v_pipeline = None   # 延迟加载
        self._pipeline = "phantom_pending"  # 标记已初始化但未加载权重
        print("[Generator] Phantom 模块加载完成（权重将在首次推理时按需加载）")

    @staticmethod
    def _offload_pipeline(pipe):
        """
        将 WanT2V / Phantom_Wan_S2V 的所有 CUDA 子模块移到 CPU，
        然后删除引用，强制 GC + empty_cache，彻底释放 VRAM。

        结构说明:
          pipe.model        → WanModel / FSDP(WanModel)  ← cpu() 或 FSDP destroy
          pipe.vae          → WanVAE (普通 class，NOT nn.Module)
                                └── .model → WanVAE_ (nn.Module)
          pipe.text_encoder → T5EncoderModel (普通 class，NOT nn.Module)
                                └── .model → T5Model / FSDP(T5Model)
        """
        import gc, torch
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        if pipe is None or isinstance(pipe, str):
            return
        # 1. DiT 主模型（可能被 FSDP 包裹）
        main_model = getattr(pipe, "model", None)
        if main_model is not None:
            try:
                if isinstance(main_model, FSDP):
                    FSDP.set_state_dict_type(main_model,
                        torch.distributed.fsdp.StateDictType.LOCAL_STATE_DICT)
                main_model.cpu()
            except Exception:
                pass
        # 2. VAE — WanVAE 是普通 class，真正的 nn.Module 在 .model 里
        vae_wrapper = getattr(pipe, "vae", None)
        if vae_wrapper is not None:
            vae_inner = getattr(vae_wrapper, "model", vae_wrapper)
            try:
                vae_inner.cpu()
            except Exception:
                pass
        # 3. T5 — T5EncoderModel 也是普通 class，nn.Module 在 .model 里（可能是 FSDP）
        te_wrapper = getattr(pipe, "text_encoder", None)
        if te_wrapper is not None:
            te_inner = getattr(te_wrapper, "model", te_wrapper)
            try:
                if isinstance(te_inner, FSDP):
                    FSDP.set_state_dict_type(te_inner,
                        torch.distributed.fsdp.StateDictType.LOCAL_STATE_DICT)
                te_inner.cpu()
            except Exception:
                pass
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"[Generator] VRAM after offload: {allocated:.2f} GiB allocated")

    def _ensure_t2v(self):
        """确保 T2V 权重已加载，S2V 已卸载"""
        if self._t2v_pipeline is not None:
            return
        import torch
        if self._pipeline not in (None, "mock", "phantom_pending"):
            print("[Generator] 卸载 S2V，释放 VRAM...")
            self._offload_pipeline(self._pipeline)
            self._pipeline = "phantom_pending"
        t2v_task = self.phantom_task.replace("s2v", "t2v")
        print(f"[Generator] 加载 WanT2V ({t2v_task}) rank={self.rank}...")
        self._t2v_pipeline = self._phantom_wan.WanT2V(
            config=self._phantom_WAN_CONFIGS[t2v_task],
            checkpoint_dir=self.wan2_t2v_dir,
            device_id=self.device.index,
            rank=self.rank,
            t5_fsdp=self.t5_fsdp,
            dit_fsdp=self.dit_fsdp,
            use_usp=self.use_usp,
        )
        print("[Generator] WanT2V 加载完成.")

    def _ensure_s2v(self):
        """确保 S2V 权重已加载，T2V 已卸载"""
        if self._pipeline not in (None, "mock", "phantom_pending"):
            return
        if self._t2v_pipeline is not None:
            print("[Generator] 卸载 T2V，释放 VRAM...")
            self._offload_pipeline(self._t2v_pipeline)
            self._t2v_pipeline = None
        print(f"[Generator] 加载 Phantom S2V ({self.phantom_task}) rank={self.rank}...")
        self._pipeline = self._phantom_wan.Phantom_Wan_S2V(
            config=self._phantom_WAN_CONFIGS[self.phantom_task],
            checkpoint_dir=self.wan2_t2v_dir,
            phantom_ckpt=self.phantom_ckpt,
            device_id=self.device.index,
            rank=self.rank,
            t5_fsdp=self.t5_fsdp,
            dit_fsdp=self.dit_fsdp,
            use_usp=self.use_usp,
        )
        print("[Generator] Phantom S2V 加载完成.")

    def _load_cogvideo(self):
        """加载 CogVideoX-5B-I2V（备用后端，首帧锁定姿态）"""
        try:
            import torch
            from diffusers import CogVideoXImageToVideoPipeline

            self._pipeline = CogVideoXImageToVideoPipeline.from_pretrained(
                self.cogvideo_model_id,
                torch_dtype=torch.bfloat16,
            )
            self._pipeline.to(self.device)
            print("[Generator] CogVideoX-5B loaded.")
        except ImportError as e:
            raise ImportError(f"请安装依赖: pip install diffusers transformers\n{e}")

    # ── 主生成接口 ────────────────────────────────────────────────────────────

    def generate(
        self,
        text_prompt: str,
        references: List[Image.Image],
        output_path: str,
        negative_prompt: str = "blurry, low quality, inconsistent character, deformed",
    ) -> str:
        """
        生成一段视频。

        路由逻辑（backend=phantom 时）：
          · references 为空 → WanT2V（纯文本生成，用于首镜头 / 新实体）
          · references 非空 → Phantom S2V（外观 conditioning，不锁定姿态）

        Args:
            text_prompt:    镜头文本描述
            references:     参考图列表（来自上一镜头 grounding，可为空）
            output_path:    输出 MP4 路径
            negative_prompt: 负面提示词

        Returns:
            output_path
        """
        self._load_pipeline()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if self.backend == "mock":
            return self._mock_generate(text_prompt, output_path)
        elif self.backend == "phantom":
            if references:
                print(f"[Generator] → Phantom S2V（{len(references)} 张参考图）")
                return self._phantom_s2v_generate(text_prompt, references, output_path)
            else:
                print("[Generator] → WanT2V（无参考图，首镜头 / 新实体）")
                return self._phantom_t2v_generate(text_prompt, output_path)
        elif self.backend == "cogvideo":
            ref_image = references[0] if references else Image.new("RGB", (self.config.width, self.config.height))
            ref_image = ref_image.resize((self.config.width, self.config.height))
            return self._cogvideo_generate(text_prompt, ref_image, negative_prompt, output_path)

    def _phantom_t2v_generate(self, prompt, output_path) -> str:
        """WanT2V：无参考图时使用，首镜头 / 实体首次出现"""
        self._ensure_t2v()
        size_key = f"{self.config.width}*{self.config.height}"
        frames = self._t2v_pipeline.generate(
            input_prompt=prompt,
            size=self._phantom_SIZE_CONFIGS[size_key],
            frame_num=self.config.num_frames,
            sampling_steps=self.config.num_inference_steps,
            guide_scale=self.config.guide_scale_text,
            seed=self.config.seed,
            offload_model=(self.world_size == 1),  # 多卡不 offload，否则每步 CPU↔GPU 极慢
        )
        # WanT2V.generate() returns None on non-rank-0 processes
        if self.rank == 0:
            self._save_frames_as_mp4(frames, output_path, self.config.fps)
            print(f"[Generator] T2V 视频已保存: {output_path}")
        return output_path

    def _phantom_s2v_generate(self, prompt, references, output_path) -> str:
        """Phantom S2V：有参考图时使用，外观 conditioning，姿态自由"""
        self._ensure_s2v()
        size_key = f"{self.config.width}*{self.config.height}"
        ref_imgs = [self._preprocess_ref_image(img, self.config.width, self.config.height)
                    for img in references[:4]]   # Phantom 最多 4 张
        frames = self._pipeline.generate(
            input_prompt=prompt,
            ref_images=ref_imgs,
            size=self._phantom_SIZE_CONFIGS[size_key],
            frame_num=self.config.num_frames,
            sampling_steps=self.config.num_inference_steps,
            guide_scale_img=self.config.guide_scale_img,
            guide_scale_text=self.config.guide_scale_text,
            seed=self.config.seed,
            offload_model=(self.world_size == 1),  # 多卡不 offload
        )
        # Phantom S2V also returns None on non-rank-0 processes
        if self.rank == 0:
            self._save_frames_as_mp4(frames, output_path, self.config.fps)
            print(f"[Generator] Phantom S2V 视频已保存: {output_path}")
        return output_path

    @staticmethod
    def _preprocess_ref_image(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
        """
        与 Phantom 官方 load_ref_images 完全一致：
        保持宽高比 resize，再用白色填充到精确的 (target_w, target_h)，
        确保所有参考图经 VAE 编码后 latent 空间维度相同。
        """
        from PIL import ImageOps
        img = img.convert("RGB")
        img_ratio = img.width / img.height
        target_ratio = target_w / target_h
        if img_ratio > target_ratio:
            new_w = target_w
            new_h = int(new_w / img_ratio)
        else:
            new_h = target_h
            new_w = int(new_h * img_ratio)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        dw = target_w - img.width
        dh = target_h - img.height
        padding = (dw // 2, dh // 2, dw - dw // 2, dh - dh // 2)
        return ImageOps.expand(img, padding, fill=(255, 255, 255))

    def _cogvideo_generate(self, prompt, ref_image, negative_prompt, output_path) -> str:
        import torch
        generator = torch.Generator(device=self.device).manual_seed(self.config.seed)
        output = self._pipeline(
            image=ref_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=self.config.height,
            width=self.config.width,
            num_frames=self.config.num_frames,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guide_scale_text,
            generator=generator,
        )
        self._save_frames_as_mp4(output.frames[0], output_path, self.config.fps)
        print(f"[Generator] CogVideoX 视频已保存: {output_path}")
        return output_path

    def _mock_generate(self, prompt: str, output_path: str) -> str:
        # Mock 模式下只有 rank 0 生成视频，其他 rank 等待
        if self.rank != 0:
            # 非 rank 0 等待 rank 0 完成
            if self.world_size > 1:
                import torch.distributed as dist
                dist.barrier()
            return output_path

        try:
            import cv2
            import numpy as np
            import subprocess

            tmp_path = output_path + ".tmp.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(tmp_path, fourcc, 24, (self.config.width, self.config.height))
            for i in range(self.config.num_frames):
                frame = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
                frame[:] = (int(255 * i / self.config.num_frames), 100, 200)
                cv2.putText(frame, f"[MOCK] {prompt[:40]}", (20, self.config.height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                out.write(frame)
            out.release()

            ret = subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_path,
                 "-vcodec", "libx264", "-pix_fmt", "yuv420p", output_path],
                capture_output=True
            )
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if ret.returncode != 0 and not os.path.exists(output_path):
                os.rename(tmp_path, output_path)

            print(f"[Generator] Mock 视频已保存: {output_path}")

            # rank 0 完成后，通知其他 rank
            if self.world_size > 1:
                import torch.distributed as dist
                dist.barrier()

            return output_path
        except ImportError:
            with open(output_path, "wb") as f:
                f.write(b"MOCK_VIDEO")
            if self.world_size > 1:
                import torch.distributed as dist
                dist.barrier()
            return output_path

    @staticmethod
    def _save_frames_as_mp4(frames, output_path: str, fps: int):
        import cv2
        import numpy as np
        import torch

        # Phantom/Wan VAE returns (C, T, H, W) tensor with values in (-1, 1).
        # Convert to a list of (H, W, 3) uint8 BGR numpy arrays first.
        if isinstance(frames, torch.Tensor) and frames.ndim == 4:
            # (C, T, H, W) → (T, C, H, W), normalize (-1,1) → (0,255)
            t = frames.detach().cpu().float()
            t = t.permute(1, 0, 2, 3)                      # (T, C, H, W)
            t = ((t.clamp(-1, 1) + 1) / 2 * 255).byte()   # uint8
            frame_list = []
            for f in t:
                arr = f.numpy().transpose(1, 2, 0)          # (H, W, C) RGB
                frame_list.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            h, w = frame_list[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            for f in frame_list:
                out.write(f)
            out.release()
            return

        def to_bgr(f):
            # CUDA tensor → CPU numpy
            if isinstance(f, torch.Tensor):
                f = f.detach().cpu().float()
                if f.max() <= 1.0:
                    f = (f * 255).clamp(0, 255)
                arr = f.numpy().astype(np.uint8)
                # (C,H,W) → (H,W,C)
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                    arr = arr.transpose(1, 2, 0)
            else:
                arr = np.array(f)
                if arr.dtype != np.uint8:
                    if arr.max() <= 1.0:
                        arr = (arr * 255).clip(0, 255)
                    arr = arr.astype(np.uint8)

            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            elif arr.shape[2] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            elif arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            return arr

        first = to_bgr(frames[0])
        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        out.write(first)
        for frame in frames[1:]:
            out.write(to_bgr(frame))
        out.release()


# ── 快速测试 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gen = Reference2VideoGenerator(backend="mock")
    dummy_ref = Image.new("RGB", (832, 480), color=(120, 80, 200))
    output = gen.generate(
        text_prompt="A detective walks through a rainy alley at night",
        references=[dummy_ref],
        output_path="./test_output/shot_mock.mp4",
    )
    print(f"输出: {output}")
