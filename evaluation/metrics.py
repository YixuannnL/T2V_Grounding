"""
evaluation/metrics.py
职责: 自动化评测指标
  - FaceID Consistency (跨镜头人脸特征余弦相似度)
  - CLIP-I Score (参考图 vs 生成帧)
  - CLIP-T Score (文本对齐)
"""

import os
import json
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from PIL import Image


@dataclass
class ShotMetrics:
    shot_id: int
    clip_i_score: float    # 参考图 vs 生成帧图像相似度
    clip_t_score: float    # 文本 vs 生成帧对齐分
    faceid_score: float    # 人脸 ID 相似度（无人物则为 -1）


@dataclass
class EvalReport:
    total_shots: int
    avg_clip_i: float
    avg_clip_t: float
    avg_faceid: float       # 仅对有人物的镜头
    per_shot: List[ShotMetrics]
    baseline_clip_i: Optional[float] = None   # 纯文本 baseline 的 CLIP-I


class ConsistencyEvaluator:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._clip_model = None
        self._clip_preprocess = None
        self._face_analyzer = None

    def _load_clip(self):
        if self._clip_model is not None:
            return
        import clip
        import torch
        self._clip_model, self._clip_preprocess = clip.load("ViT-B/32", device=self.device)

    def _load_face(self):
        if self._face_analyzer is not None:
            return
        try:
            from insightface.app import FaceAnalysis
            self._face_analyzer = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self._face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        except ImportError:
            print("[Eval] InsightFace 未安装，跳过 FaceID 评测")

    # ── CLIP-I ────────────────────────────────────────────────────────────────
    def clip_image_similarity(self, img_path_a: str, img_path_b: str) -> float:
        """计算两张图像的 CLIP 特征余弦相似度"""
        self._load_clip()
        import torch

        def load(p):
            return self._clip_preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            fa = self._clip_model.encode_image(load(img_path_a))
            fb = self._clip_model.encode_image(load(img_path_b))
            return torch.nn.functional.cosine_similarity(fa, fb).item()

    # ── CLIP-T ────────────────────────────────────────────────────────────────
    def clip_text_similarity(self, img_path: str, text: str) -> float:
        """计算图像与文本的 CLIP 对齐分"""
        self._load_clip()
        import torch
        import clip

        img_tensor = self._clip_preprocess(
            Image.open(img_path).convert("RGB")
        ).unsqueeze(0).to(self.device)
        text_tensor = clip.tokenize([text[:77]]).to(self.device)

        with torch.no_grad():
            img_feat = self._clip_model.encode_image(img_tensor)
            txt_feat = self._clip_model.encode_text(text_tensor)
            return torch.nn.functional.cosine_similarity(img_feat, txt_feat).item()

    # ── FaceID ────────────────────────────────────────────────────────────────
    def faceid_similarity(self, img_path_a: str, img_path_b: str) -> float:
        """计算两张图像中人脸特征的余弦相似度，若无人脸返回 -1"""
        self._load_face()
        if self._face_analyzer is None:
            return -1.0

        def get_embedding(path):
            img = cv2.imread(path)
            if img is None:
                return None
            faces = self._face_analyzer.get(img)
            if not faces:
                return None
            return faces[0].embedding  # 取第一个人脸

        emb_a = get_embedding(img_path_a)
        emb_b = get_embedding(img_path_b)

        if emb_a is None or emb_b is None:
            return -1.0

        cos = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
        return float(cos)

    # ── 视频帧提取（取关键帧）────────────────────────────────────────────────
    @staticmethod
    def extract_keyframe(video_path: str, frame_idx: int = 12) -> Optional[str]:
        """从视频提取指定帧，保存为临时图像，返回路径"""
        tmp_path = f"/tmp/eval_keyframe_{os.path.basename(video_path)}.jpg"
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        cv2.imwrite(tmp_path, frame)
        return tmp_path

    # ── 主评测接口 ────────────────────────────────────────────────────────────
    def evaluate_shot(
        self,
        shot_id: int,
        video_path: str,
        text_prompt: str,
        reference_paths: List[str],
    ) -> ShotMetrics:
        keyframe = self.extract_keyframe(video_path)
        if keyframe is None:
            return ShotMetrics(shot_id=shot_id, clip_i_score=0, clip_t_score=0, faceid_score=-1)

        # CLIP-T
        try:
            clip_t = self.clip_text_similarity(keyframe, text_prompt)
        except Exception:
            clip_t = 0.0

        # CLIP-I & FaceID（对每张参考图取平均）
        clip_i_scores, faceid_scores = [], []
        for ref_path in reference_paths:
            if not os.path.exists(ref_path):
                continue
            try:
                clip_i_scores.append(self.clip_image_similarity(keyframe, ref_path))
            except Exception:
                pass
            fid = self.faceid_similarity(keyframe, ref_path)
            if fid >= 0:
                faceid_scores.append(fid)

        return ShotMetrics(
            shot_id=shot_id,
            clip_i_score=float(np.mean(clip_i_scores)) if clip_i_scores else 0.0,
            clip_t_score=clip_t,
            faceid_score=float(np.mean(faceid_scores)) if faceid_scores else -1.0,
        )

    def evaluate_pipeline(self, report_path: str) -> EvalReport:
        """读取 pipeline_report.json，对所有镜头做自动化评测"""
        with open(report_path, "r") as f:
            report = json.load(f)

        per_shot = []
        for shot in report["shots"]:
            shot_id = shot["shot_id"]
            video_path = shot["video_path"]
            text = shot["text"]
            refs = []
            for crops in shot.get("reference_used", {}).values():
                refs.extend(crops)

            metrics = self.evaluate_shot(shot_id, video_path, text, refs)
            per_shot.append(metrics)
            print(f"  Shot {shot_id}: CLIP-I={metrics.clip_i_score:.3f}  "
                  f"CLIP-T={metrics.clip_t_score:.3f}  FaceID={metrics.faceid_score:.3f}")

        faceid_valid = [m.faceid_score for m in per_shot if m.faceid_score >= 0]

        return EvalReport(
            total_shots=len(per_shot),
            avg_clip_i=float(np.mean([m.clip_i_score for m in per_shot])),
            avg_clip_t=float(np.mean([m.clip_t_score for m in per_shot])),
            avg_faceid=float(np.mean(faceid_valid)) if faceid_valid else -1.0,
            per_shot=per_shot,
        )
