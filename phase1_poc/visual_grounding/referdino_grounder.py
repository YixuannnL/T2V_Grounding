"""
visual_grounding/referdino_grounder.py
职责: 使用 ReferDINO 在视频帧中定位并分割指定实体（一步到位，替代 DINO + SAM）

ReferDINO 优势:
  1. 端到端分割: 直接输出 pixel-level mask，无需 SAM
  2. 时序一致性: object-consistent temporal enhancer，跨帧追踪同一目标
  3. 多实体联合理解: 支持同时输入多个实体描述，利用对比消歧
  4. 速度快: 51 FPS 实时推理

参考: https://github.com/iSEE-Laboratory/ReferDINO
"""

import os
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as Func


@dataclass
class GroundingResult:
    """单个实体的 grounding 结果"""
    entity_id: str
    frame_path: str           # 原始帧路径
    crop_path: str            # 裁切图路径（白底前景）
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    score: float              # 检测置信度
    has_mask: bool            # 是否有精准 mask（ReferDINO 始终为 True）
    frame_idx: int = 0        # 帧索引（用于时序排序）


@dataclass
class MultiEntityGroundingResult:
    """多实体联合 grounding 结果"""
    results_by_entity: Dict[str, List[GroundingResult]] = field(default_factory=dict)

    def get_entity_results(self, entity_id: str) -> List[GroundingResult]:
        return self.results_by_entity.get(entity_id, [])

    def add_result(self, entity_id: str, result: GroundingResult):
        if entity_id not in self.results_by_entity:
            self.results_by_entity[entity_id] = []
        self.results_by_entity[entity_id].append(result)


class ReferDINOGrounder:
    """
    使用 ReferDINO 在视频帧中定位并分割实体

    核心特性:
      - 端到端 referring video object segmentation
      - 支持多实体联合检测（利用对比消歧）
      - 时序一致的 mask 输出

    依赖安装:
        git clone https://github.com/iSEE-Laboratory/ReferDINO.git
        cd ReferDINO && pip install -r requirements.txt
        cd models/GroundingDINO/ops && python setup.py build install
    """

    _HERE = os.path.dirname(os.path.abspath(__file__))
    _WEIGHTS_DIR = os.path.join(_HERE, "..", "weights")

    # 图像预处理（与 ReferDINO 官方一致）
    _transform = T.Compose([
        T.Resize(360),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def __init__(
        self,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        mask_threshold: float = 0.5,
        enable_amp: bool = True,
    ):
        """
        Args:
            config_path: ReferDINO 配置文件路径
            checkpoint_path: 模型权重路径
            device: 运行设备
            mask_threshold: mask 二值化阈值
            enable_amp: 是否启用混合精度推理
        """
        self.device = device
        self.mask_threshold = mask_threshold
        self.enable_amp = enable_amp
        self._model = None

        # 默认路径
        self._config_path = config_path or os.path.join(
            self._WEIGHTS_DIR, "referdino", "configs", "ytvos_swinb.yaml"
        )
        self._checkpoint_path = checkpoint_path or os.path.join(
            self._WEIGHTS_DIR, "referdino", "ryt_mevis_swinb.pth"
        )

    def _load_model(self):
        """懒加载 ReferDINO 模型"""
        if self._model is not None:
            return

        try:
            # 添加 ReferDINO 到 path
            referdino_path = os.path.join(self._WEIGHTS_DIR, "referdino")
            if referdino_path not in os.sys.path:
                os.sys.path.insert(0, referdino_path)

            from ruamel.yaml import YAML
            from easydict import EasyDict
            from models import build_model

            # 加载配置
            with open(self._config_path) as f:
                yaml = YAML(typ='safe', pure=True)
                config = yaml.load(f)
            config = {k: v['value'] for k, v in config.items()}
            args = EasyDict(config)
            args.device = self.device

            # 构建模型
            self._model, _, _ = build_model(args)
            self._model.to(self.device)

            # 加载权重
            checkpoint = torch.load(self._checkpoint_path, map_location='cpu')
            state_dict = checkpoint["model_state_dict"]
            self._model.load_state_dict(state_dict, strict=False)
            self._model.eval()

            print(f"[ReferDINO] 模型加载完成: {self._checkpoint_path}")

        except ImportError as e:
            raise ImportError(
                f"ReferDINO 依赖未安装: {e}\n"
                "请按以下步骤安装:\n"
                "  git clone https://github.com/iSEE-Laboratory/ReferDINO.git weights/referdino\n"
                "  cd weights/referdino && pip install -r requirements.txt\n"
                "  cd models/GroundingDINO/ops && python setup.py build install"
            )

    def ground_single_entity(
        self,
        frame_paths: List[str],
        entity_text: str,
        entity_id: str,
        output_dir: str,
        max_results: int = 5,
        entity_type: str = "character",
    ) -> List[GroundingResult]:
        """
        在视频帧中检测单个实体

        Args:
            frame_paths: 视频帧图像路径列表
            entity_text: 实体文本描述
            entity_id: 实体唯一 ID
            output_dir: 裁切图保存目录
            max_results: 最多返回几个结果
            entity_type: 实体类型（character/object/location）

        Returns:
            List[GroundingResult]: 检测结果列表
        """
        multi_result = self.ground_multiple_entities(
            frame_paths=frame_paths,
            entities=[{"entity_id": entity_id, "text": entity_text, "type": entity_type}],
            output_dir=output_dir,
            max_results_per_entity=max_results,
        )
        return multi_result.get_entity_results(entity_id)

    def ground_multiple_entities(
        self,
        frame_paths: List[str],
        entities: List[Dict],
        output_dir: str,
        max_results_per_entity: int = 5,
    ) -> MultiEntityGroundingResult:
        """
        在视频帧中同时检测多个实体（利用对比消歧，效果更好）

        这是 ReferDINO 的核心优势：多实体同时输入时，模型可以利用
        对比关系来消歧。例如 "woman in white bee suit" 和 "boy in white bee suit"
        同时输入时，模型能更准确地区分两者。

        Args:
            frame_paths: 视频帧图像路径列表
            entities: 实体列表，每个实体包含:
                - entity_id: str
                - text: str (描述文本)
                - type: str (character/object/location)
            output_dir: 裁切图保存目录
            max_results_per_entity: 每个实体最多返回几个结果

        Returns:
            MultiEntityGroundingResult: 多实体检测结果
        """
        self._load_model()
        os.makedirs(output_dir, exist_ok=True)

        result = MultiEntityGroundingResult()

        # 加载所有帧
        frames_pil = []
        for path in frame_paths:
            img = Image.open(path).convert('RGB')
            frames_pil.append(img)

        if not frames_pil:
            return result

        origin_w, origin_h = frames_pil[0].size

        # 预处理
        imgs = torch.stack([self._transform(img) for img in frames_pil], dim=0)
        imgs = imgs.to(self.device)

        # 为了兼容 ReferDINO 的输入格式
        from misc import nested_tensor_from_videos_list
        samples = nested_tensor_from_videos_list(imgs[None], size_divisibility=1)
        img_h, img_w = imgs.shape[-2:]
        size = torch.as_tensor([int(img_h), int(img_w)]).to(self.device)
        target = {"size": size}

        video_len = len(frames_pil)

        # 对每个实体进行推理
        # 注意：ReferDINO 的多实体可以通过多次调用或修改模型来支持
        # 这里先用逐实体推理，后续可优化为真正的联合推理
        for entity in entities:
            entity_id = entity["entity_id"]
            entity_text = entity["text"].lower().strip()
            entity_type = entity.get("type", "character")

            entity_output_dir = os.path.join(output_dir, entity_id)
            os.makedirs(entity_output_dir, exist_ok=True)

            # ReferDINO 推理
            with torch.no_grad():
                from torch.cuda.amp import autocast
                with autocast(self.enable_amp):
                    outputs = self._model.infer(samples, [entity_text], [target])

            pred_logits = outputs["pred_logits"][0]  # [t, q, k]
            pred_masks = outputs["pred_masks"][0]    # [t, q, h, w]
            pred_boxes = outputs["pred_boxes"][0]    # [t, q, 4]

            # 选择最佳 query
            pred_scores = pred_logits.sigmoid().mean(0)  # [q, K]
            max_scores, _ = pred_scores.max(-1)          # [q,]
            _, max_ind = max_scores.max(-1)              # [1,]
            max_inds = max_ind.repeat(video_len)

            # 提取对应的 mask 和 bbox
            pred_masks = pred_masks[range(video_len), max_inds, ...]  # [t, h, w]
            pred_boxes = pred_boxes[range(video_len), max_inds].cpu().numpy()  # [t, 4]
            confidence = max_scores[max_ind].item()

            # 处理 mask
            pred_masks = pred_masks.unsqueeze(0)
            pred_masks = pred_masks[:, :, :img_h, :img_w].cpu()
            pred_masks = F.interpolate(
                pred_masks,
                size=(origin_h, origin_w),
                mode='bilinear',
                align_corners=False
            )
            pred_masks = (pred_masks.sigmoid() > self.mask_threshold).squeeze(0).numpy()  # [t, h, w]

            # 处理每一帧
            entity_results = []
            for t, (frame_path, frame_pil) in enumerate(zip(frame_paths, frames_pil)):
                mask = pred_masks[t]  # [h, w]
                box_cxcywh = pred_boxes[t]  # [4] in cxcywh format, normalized

                # 转换 bbox 格式
                bbox = self._cxcywh_to_xyxy(box_cxcywh, origin_w, origin_h)
                x1, y1, x2, y2 = bbox

                # 如果 mask 为空，跳过
                if mask.sum() == 0:
                    continue

                # 根据实体类型处理
                if entity_type == "location":
                    # Location: 反转 mask，保留背景
                    crop_img = self._extract_background(
                        np.array(frame_pil), mask
                    )
                    crop_path = os.path.join(
                        entity_output_dir,
                        f"{entity_id}_{Path(frame_path).stem}_bg.jpg"
                    )
                else:
                    # Character/Object: 提取前景，白底
                    crop_img = self._extract_foreground_with_mask(
                        np.array(frame_pil), mask, bbox
                    )
                    crop_path = os.path.join(
                        entity_output_dir,
                        f"{entity_id}_{Path(frame_path).stem}_crop.jpg"
                    )

                cv2.imwrite(crop_path, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))

                entity_results.append(GroundingResult(
                    entity_id=entity_id,
                    frame_path=frame_path,
                    crop_path=crop_path,
                    bbox=bbox,
                    score=confidence,
                    has_mask=True,
                    frame_idx=t,
                ))

            # 按置信度排序，取 top-k
            entity_results.sort(key=lambda r: r.score, reverse=True)
            for r in entity_results[:max_results_per_entity]:
                result.add_result(entity_id, r)

        return result

    def ground_with_joint_caption(
        self,
        frame_paths: List[str],
        entities: List[Dict],
        output_dir: str,
        max_results_per_entity: int = 5,
    ) -> MultiEntityGroundingResult:
        """
        使用联合 caption 进行多实体检测（用户发现的效果更好的方式）

        将多个实体描述用 " . " 连接成一个 caption，让模型同时看到所有实体，
        利用对比关系消歧。

        例如:
            entities = [
                {"entity_id": "woman", "text": "woman in white bee suit"},
                {"entity_id": "boy", "text": "boy in white bee suit"},
            ]
            → joint_caption = "woman in white bee suit . boy in white bee suit"

        【重要修复】Location 背景提取：
            对于 location 类型的实体，不使用 ReferDINO 对 location 描述检测出的 mask，
            而是合并同一帧中所有前景实体（character, object）的 mask，然后 inpaint 移除前景。
            这样才能得到干净的背景，而不是带有人物的场景图。

        Args:
            frame_paths: 视频帧图像路径列表
            entities: 实体列表
            output_dir: 裁切图保存目录
            max_results_per_entity: 每个实体最多返回几个结果

        Returns:
            MultiEntityGroundingResult: 多实体检测结果
        """
        self._load_model()
        os.makedirs(output_dir, exist_ok=True)

        result = MultiEntityGroundingResult()

        if not entities or not frame_paths:
            return result

        # ── 分离前景实体和 location 实体 ─────────────────────────────────────────
        foreground_entities = [e for e in entities if e.get("type", "character") != "location"]
        location_entities = [e for e in entities if e.get("type", "character") == "location"]

        # 如果没有前景实体，只有 location，则跳过（无法提取背景）
        if not foreground_entities and location_entities:
            print(f"[ReferDINO] ⚠️ 只有 location 实体，没有前景实体，无法提取干净背景")
            return result

        # ── 只对前景实体构建联合 caption ─────────────────────────────────────────
        # Location 不参与检测，它的背景图是通过 inpaint 前景得到的
        if foreground_entities:
            joint_caption = " . ".join([e["text"].lower().strip() for e in foreground_entities])
            print(f"[ReferDINO] 联合检测 caption (前景): {joint_caption}")
        else:
            joint_caption = ""

        # 加载所有帧
        frames_pil = []
        for path in frame_paths:
            img = Image.open(path).convert('RGB')
            frames_pil.append(img)

        origin_w, origin_h = frames_pil[0].size

        # ── 如果有前景实体，进行检测 ─────────────────────────────────────────────
        all_foreground_masks = None  # [t, h, w] 合并后的前景 mask

        if foreground_entities and joint_caption:
            # 预处理
            imgs = torch.stack([self._transform(img) for img in frames_pil], dim=0)
            imgs = imgs.to(self.device)

            from misc import nested_tensor_from_videos_list
            samples = nested_tensor_from_videos_list(imgs[None], size_divisibility=1)
            img_h, img_w = imgs.shape[-2:]
            size = torch.as_tensor([int(img_h), int(img_w)]).to(self.device)
            target = {"size": size}

            # 联合推理
            with torch.no_grad():
                from torch.cuda.amp import autocast
                with autocast(self.enable_amp):
                    outputs = self._model.infer(samples, [joint_caption], [target])

            pred_logits = outputs["pred_logits"][0]  # [t, q, k]
            pred_masks = outputs["pred_masks"][0]    # [t, q, h, w]
            pred_boxes = outputs["pred_boxes"][0]    # [t, q, 4]

            # 选择 top-k queries（每个 entity 一个）
            num_fg_entities = len(foreground_entities)
            pred_scores = pred_logits.sigmoid().mean(0)  # [q, K]
            max_scores, _ = pred_scores.max(-1)          # [q,]

            # 取 top-k 个 query
            topk_scores, topk_indices = max_scores.topk(min(num_fg_entities, max_scores.size(0)))

            # 初始化合并后的前景 mask（用于 location 背景提取）
            all_foreground_masks = torch.zeros(
                (len(frames_pil), origin_h, origin_w),
                dtype=torch.bool
            )

            # 为每个前景 entity 分配最近的 query
            for entity_idx, entity in enumerate(foreground_entities):
                if entity_idx >= len(topk_indices):
                    break

                query_idx = topk_indices[entity_idx].item()
                entity_id = entity["entity_id"]
                entity_type = entity.get("type", "character")
                confidence = topk_scores[entity_idx].item()

                entity_output_dir = os.path.join(output_dir, entity_id)
                os.makedirs(entity_output_dir, exist_ok=True)

                # 提取该 query 的 mask 和 bbox
                entity_masks = pred_masks[:, query_idx, ...]  # [t, h, w]
                entity_boxes = pred_boxes[:, query_idx, :].cpu().numpy()  # [t, 4]

                # 处理 mask
                entity_masks = entity_masks.unsqueeze(0)
                entity_masks = entity_masks[:, :, :img_h, :img_w].cpu()
                entity_masks = F.interpolate(
                    entity_masks,
                    size=(origin_h, origin_w),
                    mode='bilinear',
                    align_corners=False
                )
                entity_masks = (entity_masks.sigmoid() > self.mask_threshold).squeeze(0)  # [t, h, w]

                # ── 累积到 all_foreground_masks（用于后续 location 背景提取）────────
                all_foreground_masks = all_foreground_masks | entity_masks

                entity_masks_np = entity_masks.numpy()

                # 处理每一帧
                for t, (frame_path, frame_pil) in enumerate(zip(frame_paths, frames_pil)):
                    mask = entity_masks_np[t]
                    box_cxcywh = entity_boxes[t]
                    bbox = self._cxcywh_to_xyxy(box_cxcywh, origin_w, origin_h)

                    if mask.sum() == 0:
                        continue

                    # 前景实体：提取前景，白底
                    crop_img = self._extract_foreground_with_mask(
                        np.array(frame_pil), mask, bbox
                    )
                    crop_path = os.path.join(
                        entity_output_dir,
                        f"{entity_id}_{Path(frame_path).stem}_crop.jpg"
                    )

                    cv2.imwrite(crop_path, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))

                    result.add_result(entity_id, GroundingResult(
                        entity_id=entity_id,
                        frame_path=frame_path,
                        crop_path=crop_path,
                        bbox=bbox,
                        score=confidence,
                        has_mask=True,
                        frame_idx=t,
                    ))

        # ── 处理 location 实体：用合并后的前景 mask 来 inpaint ─────────────────────
        if location_entities and all_foreground_masks is not None:
            all_foreground_masks_np = all_foreground_masks.numpy()

            for location_entity in location_entities:
                entity_id = location_entity["entity_id"]
                entity_output_dir = os.path.join(output_dir, entity_id)
                os.makedirs(entity_output_dir, exist_ok=True)

                # 对每一帧提取背景
                for t, (frame_path, frame_pil) in enumerate(zip(frame_paths, frames_pil)):
                    fg_mask = all_foreground_masks_np[t]
                    frame_np = np.array(frame_pil)

                    # 计算前景 mask 覆盖率
                    mask_coverage = fg_mask.sum() / (fg_mask.shape[0] * fg_mask.shape[1])

                    # 如果没有前景 mask 或覆盖率过低，尝试 fallback 到人脸检测
                    if fg_mask.sum() == 0 or mask_coverage < 0.01:
                        print(f"[ReferDINO] ⚠️ 帧 {t} 前景 mask 覆盖率过低 ({mask_coverage:.2%})，"
                              f"尝试 fallback 人脸检测...")
                        fg_mask = self._fallback_face_detection_mask(frame_np)
                        if fg_mask is None or fg_mask.sum() == 0:
                            print(f"[ReferDINO] ⚠️ 帧 {t} fallback 也失败，跳过 location 背景提取")
                            continue

                    # 用合并后的前景 mask 来 inpaint
                    bg_img = self._extract_background(frame_np, fg_mask)
                    crop_path = os.path.join(
                        entity_output_dir,
                        f"{entity_id}_{Path(frame_path).stem}_bg.jpg"
                    )

                    cv2.imwrite(crop_path, cv2.cvtColor(bg_img, cv2.COLOR_RGB2BGR))

                    # 计算 location 的 "bbox"（其实是整个画面）
                    bbox = (0, 0, origin_w, origin_h)

                    result.add_result(entity_id, GroundingResult(
                        entity_id=entity_id,
                        frame_path=frame_path,
                        crop_path=crop_path,
                        bbox=bbox,
                        score=0.5,  # location 的置信度固定为中等
                        has_mask=True,
                        frame_idx=t,
                    ))

                print(f"[ReferDINO] ✅ Location '{entity_id}' 背景提取完成 "
                      f"(使用 {len(foreground_entities)} 个前景实体的合并 mask)")

        return result

    def _fallback_face_detection_mask(
        self,
        image_rgb: np.ndarray,
        expansion_ratio: float = 2.5,
    ) -> Optional[np.ndarray]:
        """
        当 ReferDINO 前景检测失败时，使用人脸检测作为 fallback

        检测到人脸后，将 bbox 扩展为更大区域（假设人物占据人脸周围的区域）
        """
        try:
            # 尝试使用 OpenCV 的人脸检测
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))

            if len(faces) == 0:
                # 尝试侧脸检测
                profile_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_profileface.xml'
                )
                faces = profile_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))

            if len(faces) == 0:
                return None

            h, w = image_rgb.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

            for (x, y, fw, fh) in faces:
                # 扩展人脸区域，假设人物身体在人脸下方
                cx, cy = x + fw // 2, y + fh // 2
                expanded_w = int(fw * expansion_ratio)
                expanded_h = int(fh * expansion_ratio * 1.5)  # 身体比脸长

                x1 = max(0, cx - expanded_w // 2)
                y1 = max(0, y - int(fh * 0.3))  # 头顶留一点
                x2 = min(w, cx + expanded_w // 2)
                y2 = min(h, cy + expanded_h)

                mask[y1:y2, x1:x2] = 1

            print(f"[ReferDINO] Fallback 检测到 {len(faces)} 个人脸，生成扩展 mask")
            return mask.astype(bool)

        except Exception as e:
            print(f"[ReferDINO] ⚠️ Fallback 人脸检测失败: {e}")
            return None

    def _cxcywh_to_xyxy(
        self,
        box: np.ndarray,
        img_w: int,
        img_h: int
    ) -> Tuple[int, int, int, int]:
        """将 cxcywh 格式的归一化 bbox 转换为 xyxy 格式的像素坐标"""
        cx, cy, w, h = box
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        return (x1, y1, x2, y2)

    def _extract_foreground_with_mask(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """使用 mask 提取前景，背景填白，裁切到 bbox 范围"""
        x1, y1, x2, y2 = bbox

        # 背景填白
        result = image_rgb.copy()
        result[mask == 0] = 255

        # 裁切到 bbox 范围
        return result[y1:y2, x1:x2]

    def _extract_background(
        self,
        image_rgb: np.ndarray,
        foreground_mask: np.ndarray,
    ) -> np.ndarray:
        """
        移除前景，用 inpaint 填充，返回干净的背景

        Args:
            image_rgb: 原始图像
            foreground_mask: 前景 mask（需要移除的区域）
        """
        # 轻微膨胀 mask，避免边缘残影
        mask_uint8 = foreground_mask.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated = cv2.dilate(mask_uint8, kernel, iterations=1)

        # cv2.inpaint 修复前景区域
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        result_bgr = cv2.inpaint(img_bgr, dilated, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


# ── 兼容旧接口的适配器 ──────────────────────────────────────────────────────────

class VideoGrounderAdapter:
    """
    适配器类，使 ReferDINOGrounder 兼容原有 VideoGrounder 接口

    用法:
        # 替换原有的 VideoGrounder
        # grounder = VideoGrounder()
        grounder = VideoGrounderAdapter()  # 内部使用 ReferDINO

        results = grounder.ground_in_frames(
            frame_paths, entity_text, entity_id, output_dir
        )
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        box_threshold: float = 0.35,  # 保留参数但不使用（兼容性）
        text_threshold: float = 0.25,  # 保留参数但不使用
    ):
        self._referdino = ReferDINOGrounder(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        # 这些参数保留以兼容旧代码，但 ReferDINO 不使用它们
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device

    def ground_in_frames(
        self,
        frame_paths: List[str],
        entity_text: str,
        entity_id: str,
        output_dir: str,
        max_results: int = 5,
        entity_type: str = "character",
        registered_fg_descriptions: List[str] = None,  # ReferDINO 不需要
    ) -> List[GroundingResult]:
        """
        兼容原有 VideoGrounder.ground_in_frames 接口
        """
        return self._referdino.ground_single_entity(
            frame_paths=frame_paths,
            entity_text=entity_text,
            entity_id=entity_id,
            output_dir=output_dir,
            max_results=max_results,
            entity_type=entity_type,
        )

    def ground_multiple_entities_jointly(
        self,
        frame_paths: List[str],
        entities: List[Dict],
        output_dir: str,
        max_results_per_entity: int = 5,
        use_joint_caption: bool = True,
    ) -> MultiEntityGroundingResult:
        """
        多实体联合检测（新接口，利用 ReferDINO 优势）

        Args:
            use_joint_caption: 是否使用联合 caption（用户发现效果更好）
        """
        if use_joint_caption:
            return self._referdino.ground_with_joint_caption(
                frame_paths=frame_paths,
                entities=entities,
                output_dir=output_dir,
                max_results_per_entity=max_results_per_entity,
            )
        else:
            return self._referdino.ground_multiple_entities(
                frame_paths=frame_paths,
                entities=entities,
                output_dir=output_dir,
                max_results_per_entity=max_results_per_entity,
            )


# ── 工具函数（与原 grounder.py 兼容）─────────────────────────────────────────────

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


# ── 快速测试 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("用法: python referdino_grounder.py <video_path> <entity_text1> [entity_text2] ...")
        print("示例: python referdino_grounder.py test.mp4 'woman in white bee suit' 'boy in white bee suit'")
        sys.exit(1)

    video_path = sys.argv[1]
    entity_texts = sys.argv[2:]

    frames_dir = "./tmp_frames"
    output_dir = "./tmp_crops_referdino"

    # 抽帧
    frames = extract_frames(video_path, frames_dir)

    # 构建实体列表
    entities = [
        {"entity_id": f"entity_{i}", "text": text, "type": "character"}
        for i, text in enumerate(entity_texts)
    ]

    # 使用 ReferDINO 进行多实体联合检测
    grounder = ReferDINOGrounder()
    results = grounder.ground_with_joint_caption(
        frame_paths=frames,
        entities=entities,
        output_dir=output_dir,
    )

    print(f"\n检测结果:")
    for entity_id, entity_results in results.results_by_entity.items():
        print(f"\n  {entity_id}: {len(entity_results)} 个结果")
        for r in entity_results[:3]:
            print(f"    frame={r.frame_idx}  score={r.score:.3f}  crop={r.crop_path}")
