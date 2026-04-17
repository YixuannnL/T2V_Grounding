"""
visual_grounding/grounder.py
职责: 在视频帧中定位指定实体（Grounding DINO + SAM2），返回高质量裁切图
"""

import os
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
from PIL import Image


@dataclass
class GroundingResult:
    entity_id: str
    frame_path: str           # 原始帧路径
    crop_path: str            # 裁切图路径
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    score: float              # 检测置信度
    has_mask: bool            # 是否有精准 mask


class VideoGrounder:
    """
    使用 Grounding DINO + SAM2 在视频帧中定位实体

    依赖安装:
        pip install groundingdino-py
        pip install git+https://github.com/facebookresearch/sam2.git
    """

    # 本文件所在目录 → phase1_poc/visual_grounding/
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _WEIGHTS_DIR = os.path.join(_HERE, "..", "weights")

    def __init__(
        self,
        gdino_config: Optional[str] = None,
        gdino_weights: Optional[str] = None,
        sam2_config: str = "sam2_hiera_l.yaml",
        sam2_weights: Optional[str] = None,
        device: str = "cuda",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._gdino_model = None
        self._sam2_predictor = None
        self._gdino_cfg = gdino_config      # None → auto-resolve in _load_gdino
        self._gdino_weights = gdino_weights or os.path.join(
            self._WEIGHTS_DIR, "groundingdino_swinb_cogcoor.pth"
        )
        self._sam2_config = sam2_config
        self._sam2_weights = sam2_weights or os.path.join(
            self._WEIGHTS_DIR, "sam2_hiera_large.pt"
        )

    def _resolve_gdino_config(self) -> str:
        """优先用显式路径，否则从已安装的 groundingdino 包中取 config。"""
        if self._gdino_cfg and os.path.exists(self._gdino_cfg):
            return self._gdino_cfg
        # 尝试 weights/ 目录里的 config 副本
        local = os.path.join(self._WEIGHTS_DIR, "GroundingDINO_SwinB_cfg.py")
        if os.path.exists(local):
            return local
        # 从已安装包里取
        try:
            import groundingdino
            pkg_cfg = os.path.join(
                os.path.dirname(groundingdino.__file__),
                "config", "GroundingDINO_SwinB_cfg.py",
            )
            if os.path.exists(pkg_cfg):
                return pkg_cfg
        except Exception:
            pass
        raise FileNotFoundError(
            "GroundingDINO config not found. Searched:\n"
            f"  explicit: {self._gdino_cfg}\n"
            f"  weights/: {local}\n"
            "  installed groundingdino package config"
        )

    def _load_gdino(self):
        """懒加载 Grounding DINO"""
        if self._gdino_model is not None:
            return
        try:
            # transformers >= 5.x removed get_head_mask from BertModel/PreTrainedModel,
            # but groundingdino's BertModelWarper.__init__ still accesses it.
            # Patch it back before importing groundingdino.
            import transformers
            if not hasattr(transformers.BertModel, 'get_head_mask'):
                def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
                    if head_mask is not None:
                        import torch
                        if head_mask.dim() == 1:
                            head_mask = (head_mask.unsqueeze(0).unsqueeze(0)
                                         .unsqueeze(-1).unsqueeze(-1))
                            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
                        elif head_mask.dim() == 2:
                            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                        if is_attention_chunked:
                            head_mask = head_mask.unsqueeze(-1)
                    else:
                        head_mask = [None] * num_hidden_layers
                    return head_mask
                transformers.BertModel.get_head_mask = _get_head_mask
                print("[Grounder] Applied get_head_mask compatibility patch "
                      f"(transformers {transformers.__version__}).")

            # transformers >= 5.x changed get_extended_attention_mask signature:
            # old: (attention_mask, input_shape, device, dtype=None)
            # new: (attention_mask, input_shape, dtype=None)
            # BertModelWarper.forward passes 'device' as the 3rd positional arg,
            # which lands in 'dtype' under the new signature → TypeError.
            _PreTrained = transformers.modeling_utils.PreTrainedModel
            if not hasattr(_PreTrained, '_orig_get_ext_attn_mask'):
                import torch as _torch
                _orig_ext = _PreTrained.get_extended_attention_mask
                def _compat_get_extended_attention_mask(self, attention_mask, input_shape, dtype=None):
                    # If old-style caller passed a torch.device as dtype, discard it
                    if isinstance(dtype, _torch.device):
                        dtype = self.dtype
                    return _orig_ext(self, attention_mask, input_shape, dtype)
                _PreTrained._orig_get_ext_attn_mask = _orig_ext
                _PreTrained.get_extended_attention_mask = _compat_get_extended_attention_mask
                print("[Grounder] Applied get_extended_attention_mask compatibility patch "
                      f"(transformers {transformers.__version__}).")

            from groundingdino.util.inference import load_model
            cfg = self._resolve_gdino_config()
            print(f"[Grounder] DINO config: {cfg}")
            self._gdino_model = load_model(cfg, self._gdino_weights)
            self._gdino_model = self._gdino_model.to(self.device)
            print("[Grounder] Grounding DINO loaded.")
        except ImportError:
            raise ImportError("请先安装: pip install groundingdino-py")

    def _load_sam2(self):
        """懒加载 SAM2"""
        if self._sam2_predictor is not None:
            return
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            sam2_model = build_sam2(self._sam2_config, self._sam2_weights, device=self.device)
            self._sam2_predictor = SAM2ImagePredictor(sam2_model)
            print("[Grounder] SAM2 loaded.")
        except ImportError:
            raise ImportError("请先安装: pip install git+https://github.com/facebookresearch/sam2.git")

    def ground_in_frames(
        self,
        frame_paths: List[str],
        entity_text: str,
        entity_id: str,
        output_dir: str,
        max_results: int = 5,
        entity_type: str = "character",
        registered_fg_descriptions: List[str] = None,
    ) -> List[GroundingResult]:
        """
        在给定帧列表中检测实体，返回检测结果（按置信度降序）

        Args:
            frame_paths:  视频帧图像路径列表
            entity_text:  实体文本描述，如 "man in trench coat"
            entity_id:    实体唯一 ID
            output_dir:   裁切图保存目录
            max_results:  最多返回几个结果
            entity_type:  实体类型（character/object/location/style）
                          location 类型跳过 SAM2，直接保存完整帧
            registered_fg_descriptions: 数据库中已注册的前景实体描述列表
                                        仅用于 location 类型，提取背景时移除这些实体
        """
        self._load_gdino()
        # location 需要 SAM2 来抠前景
        self._load_sam2()

        os.makedirs(output_dir, exist_ok=True)
        results = []

        from groundingdino.util.inference import predict, load_image

        for frame_path in frame_paths:
            image_source, image_tensor = load_image(frame_path)

            # location 类型：抠掉前景，保留干净背景
            if entity_type == "location":
                frame_stem = Path(frame_path).stem
                crop_path = os.path.join(output_dir, f"{entity_id}_{frame_stem}_det00_crop.jpg")
                bg_img = self._extract_background(
                    image_source, image_tensor, registered_fg_descriptions
                )
                cv2.imwrite(crop_path, cv2.cvtColor(bg_img, cv2.COLOR_RGB2BGR))
                h, w = image_source.shape[:2]
                results.append(GroundingResult(
                    entity_id=entity_id,
                    frame_path=frame_path,
                    crop_path=crop_path,
                    bbox=(0, 0, w, h),
                    score=1.0,
                    has_mask=False,
                ))
                continue

            boxes, logits, phrases = predict(
                model=self._gdino_model,
                image=image_tensor,
                caption=entity_text,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )

            if len(boxes) == 0:
                continue

            h, w = image_source.shape[:2]
            frame_stem = Path(frame_path).stem

            for det_idx in range(len(boxes)):
                box = boxes[det_idx]
                score = logits[det_idx].item()

                cx, cy, bw, bh = box.tolist()
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # 用 SAM2 精细化 mask
                crop_img = self._extract_with_sam2(
                    image_source, (x1, y1, x2, y2), frame_path, entity_id, output_dir
                )

                if crop_img is None:
                    crop_img = image_source[y1:y2, x1:x2]

                crop_path = os.path.join(output_dir, f"{entity_id}_{frame_stem}_det{det_idx:02d}_crop.jpg")
                cv2.imwrite(crop_path, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))

                results.append(GroundingResult(
                    entity_id=entity_id,
                    frame_path=frame_path,
                    crop_path=crop_path,
                    bbox=(x1, y1, x2, y2),
                    score=score,
                    has_mask=True,
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:max_results]

    def _extract_with_sam2(
        self,
        image_rgb: np.ndarray,
        bbox: Tuple[int, int, int, int],
        frame_path: str,
        entity_id: str,
        output_dir: str,
    ) -> Optional[np.ndarray]:
        """用 SAM2 做精准前景分割，返回裁切图（带白色背景的前景）"""
        try:
            pil_image = Image.fromarray(image_rgb)
            self._sam2_predictor.set_image(pil_image)

            x1, y1, x2, y2 = bbox
            input_box = np.array([[x1, y1, x2, y2]])
            masks, scores, _ = self._sam2_predictor.predict(
                box=input_box,
                multimask_output=False,
            )

            mask = masks[0].astype(np.uint8)  # (H, W)

            # 提取前景，背景填白
            result = image_rgb.copy()
            result[mask == 0] = 255

            # 裁切到 bbox 范围
            return result[y1:y2, x1:x2]
        except Exception as e:
            print(f"[Grounder] SAM2 分割失败: {e}，使用直接裁切")
            return None

    def _extract_background(
        self,
        image_rgb: np.ndarray,
        image_tensor,
        registered_fg_descriptions: List[str] = None,
    ) -> np.ndarray:
        """
        检测帧中所有前景人物/物体，用 SAM2 得到 union mask，
        再用 cv2.inpaint 填充前景区域，返回干净的背景帧。

        Args:
            image_rgb:    原始帧（numpy RGB）
            image_tensor: 已由 load_image 生成的 DINO 输入 tensor
            registered_fg_descriptions: 数据库中已注册的前景实体描述列表
                                        这些实体在提取背景时需要被移除
        """
        from groundingdino.util.inference import predict

        # 构建前景检测 prompt
        # 基础 prompt（通用的人物/物体检测词）
        base_prompts = ["person", "people", "man", "woman", "baby", "child", "object", "bag", "item"]

        # 添加数据库中已注册的前景实体描述
        # 这样可以更精准地检测并移除特定物体（如 bassinet）
        if registered_fg_descriptions:
            base_prompts.extend(registered_fg_descriptions)
            print(f"[Grounder] Location background extraction: "
                  f"removing {len(registered_fg_descriptions)} registered foreground entities")

        caption = " . ".join(base_prompts)

        boxes, logits, _ = predict(
            model=self._gdino_model,
            image=image_tensor,
            caption=caption,
            box_threshold=0.30,
            text_threshold=0.25,
        )

        h, w = image_rgb.shape[:2]
        union_mask = np.zeros((h, w), dtype=np.uint8)

        if len(boxes) > 0:
            pil_image = Image.fromarray(image_rgb)
            self._sam2_predictor.set_image(pil_image)

            for box in boxes:
                cx, cy, bw, bh = box.tolist()
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                try:
                    masks, _, _ = self._sam2_predictor.predict(
                        box=np.array([[x1, y1, x2, y2]]),
                        multimask_output=False,
                    )
                    union_mask = np.logical_or(union_mask, masks[0]).astype(np.uint8)
                except Exception:
                    pass

        if union_mask.sum() == 0:
            # 没检测到前景，直接返回原帧
            return image_rgb

        # 轻微膨胀 mask，避免边缘残影
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated = cv2.dilate(union_mask, kernel, iterations=1)

        # cv2.inpaint 修复前景区域（Telea 算法，快速）
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        result_bgr = cv2.inpaint(img_bgr, dilated, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def extract_frames(video_path: str, output_dir: str, fps: float = 2.0) -> List[str]:
    """从视频文件均匀采帧，返回帧图像路径列表"""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / fps))

    paths = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(path, frame)
            paths.append(path)
        frame_idx += 1
    cap.release()
    print(f"[FrameExtractor] 从 {video_path} 提取了 {len(paths)} 帧")
    return paths


# ── 快速测试（需要 GPU + 模型权重）────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("用法: python grounder.py <video_path> <entity_text>")
        print("示例: python grounder.py test.mp4 'man in trench coat'")
        sys.exit(1)

    video_path = sys.argv[1]
    entity_text = sys.argv[2]

    frames_dir = "./tmp_frames"
    output_dir = "./tmp_crops"

    frames = extract_frames(video_path, frames_dir)

    grounder = VideoGrounder()
    results = grounder.ground_in_frames(frames, entity_text, "test_entity", output_dir)

    print(f"\n找到 {len(results)} 个匹配结果:")
    for r in results:
        print(f"  frame={r.frame_path}  score={r.score:.3f}  crop={r.crop_path}")
