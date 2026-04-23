"""
reference_manager/smart_registry.py
职责: Agentic 自优化参考图库 —— 智能注册、淘汰、多样性管理

核心改进：
1. 注册时智能过滤（质量门槛 + 相似度去重）
2. 定期淘汰低质量/冗余参考
3. 保持 Top-K 多样性（不同角度、光线）
4. VLM 驱动的参考图质量审计

设计原则：
- 每个实体保持精而不是多
- 优先保留：高质量 + 早期 shot + 多样角度
- 淘汰：低质量 + 冗余 + 过时
"""

import os
import json
import sqlite3
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple
from datetime import datetime
from enum import Enum

# 假设这些模块存在
# from ..utils.llm_client import LLMClient


class EvictionReason(Enum):
    """淘汰原因枚举"""
    LOW_QUALITY = "low_quality"           # 质量分过低
    REDUNDANT = "redundant"               # 与其他参考图过于相似
    SUPERSEDED = "superseded"             # 被更高质量的替代
    EXPIRED = "expired"                   # 过旧（可选策略）
    MANUAL = "manual"                     # 人工标记淘汰


@dataclass
class ReferenceEntry:
    """参考图记录"""
    entity_id: str
    shot_id: int
    frame_path: str
    crop_path: str
    quality_score: float
    source: str                           # "grounding" | "bootstrapped" | "manual"
    created_at: str = ""
    id_confidence: float = 1.0            # 人脸检测置信度
    embedding: Optional[bytes] = None     # CLIP/DINO embedding (用于相似度计算)
    pose_tag: str = ""                    # 姿态标签: "frontal" | "profile" | "back" | "closeup"
    lighting_tag: str = ""                # 光线标签: "bright" | "dim" | "backlit" | "golden"
    is_anchor: bool = False               # 是否为锚点参考图（受保护，不会被淘汰）

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class EvictionRecord:
    """淘汰记录（用于审计和调试）"""
    entity_id: str
    crop_path: str
    quality_score: float
    reason: str
    evicted_at: str
    replaced_by: Optional[str] = None     # 替代它的参考图路径


@dataclass
class RegistryConfig:
    """Registry 配置"""
    # 注册阈值
    min_quality_to_register: float = 0.4      # 低于此分数直接拒绝注册
    min_id_confidence: float = 0.3            # 人脸置信度阈值

    # 容量管理
    max_refs_per_entity: int = 10             # 每个实体最多保留多少参考图
    max_refs_per_shot: int = 2                # 每个 shot 每个实体最多注册多少张

    # 相似度去重
    similarity_threshold: float = 0.92        # CLIP 相似度超过此阈值视为重复

    # 多样性要求
    min_pose_diversity: int = 2               # 至少保留几种不同姿态
    protected_anchor_count: int = 2           # 每个实体保护几个锚点不被淘汰

    # 淘汰策略
    eviction_quality_threshold: float = 0.5   # 低于此分数可被淘汰
    eviction_check_interval: int = 5          # 每处理多少个 shot 触发一次淘汰检查


class SmartEntityRegistry:
    """
    Agentic 自优化参考图库

    与旧版 EntityRegistry 的区别：
    1. 注册时有智能过滤（不是什么都收）
    2. 有主动淘汰机制
    3. 维护多样性（不同角度、光线）
    4. 支持 VLM 审计
    """

    def __init__(
        self,
        db_path: str = "./smart_registry.db",
        config: Optional[RegistryConfig] = None,
        clip_model: Optional[any] = None,  # 可选的 CLIP 模型用于 embedding
    ):
        self.db_path = db_path
        self.config = config or RegistryConfig()
        self.clip_model = clip_model
        self._shot_counter = 0  # 用于触发定期淘汰
        self._init_db()

    def _init_db(self):
        """初始化数据库 schema"""
        conn = sqlite3.connect(self.db_path)

        # 主表：参考图记录
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ref_entries (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id       TEXT NOT NULL,
                shot_id         INTEGER NOT NULL,
                frame_path      TEXT,
                crop_path       TEXT NOT NULL UNIQUE,
                quality_score   REAL NOT NULL,
                source          TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                id_confidence   REAL DEFAULT 1.0,
                embedding       BLOB,
                pose_tag        TEXT DEFAULT '',
                lighting_tag    TEXT DEFAULT '',
                is_anchor       INTEGER DEFAULT 0,
                is_active       INTEGER DEFAULT 1
            )
        """)

        # 淘汰记录表（用于审计）
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eviction_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id       TEXT NOT NULL,
                crop_path       TEXT NOT NULL,
                quality_score   REAL,
                reason          TEXT NOT NULL,
                evicted_at      TEXT NOT NULL,
                replaced_by     TEXT
            )
        """)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_entity ON ref_entries(entity_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_active ON ref_entries(is_active)")
        conn.commit()
        conn.close()

    # ══════════════════════════════════════════════════════════════════════════════
    # 智能注册
    # ══════════════════════════════════════════════════════════════════════════════

    def register(
        self,
        entity_id: str,
        entry: ReferenceEntry,
        force: bool = False,
    ) -> Tuple[bool, str]:
        """
        智能注册参考图

        与旧版区别：不是直接 INSERT，而是经过多层过滤：
        1. 质量门槛检查
        2. 人脸置信度检查（character）
        3. 相似度去重
        4. 容量检查（超限则触发淘汰）

        Args:
            entity_id: 实体 ID
            entry: 参考图记录
            force: 是否强制注册（跳过过滤）

        Returns:
            (success, reason): 是否成功注册，以及原因说明
        """
        cfg = self.config

        # ── Check 1: 质量门槛 ──────────────────────────────────────────────────
        if not force and entry.quality_score < cfg.min_quality_to_register:
            reason = f"质量分 {entry.quality_score:.3f} < 阈值 {cfg.min_quality_to_register}"
            print(f"[SmartRegistry] 拒绝注册 {entity_id}: {reason}")
            return False, reason

        # ── Check 2: 人脸置信度（仅 character）────────────────────────────────
        if not force and entity_id.startswith("char_"):
            if entry.id_confidence < cfg.min_id_confidence:
                reason = f"人脸置信度 {entry.id_confidence:.2f} < 阈值 {cfg.min_id_confidence}"
                print(f"[SmartRegistry] 拒绝注册 {entity_id}: {reason}")
                return False, reason

        # ── Check 3: 相似度去重 ────────────────────────────────────────────────
        if not force and self.clip_model is not None:
            is_duplicate, similar_path = self._check_duplicate(entity_id, entry)
            if is_duplicate:
                reason = f"与已有参考图 {similar_path} 过于相似"
                print(f"[SmartRegistry] 拒绝注册 {entity_id}: {reason}")
                return False, reason

        # ── Check 4: 同 shot 容量限制 ──────────────────────────────────────────
        shot_count = self._count_refs_in_shot(entity_id, entry.shot_id)
        if shot_count >= cfg.max_refs_per_shot:
            # 同一 shot 已有足够参考，检查是否新的更好
            worst_in_shot = self._get_worst_in_shot(entity_id, entry.shot_id)
            if worst_in_shot and entry.quality_score > worst_in_shot.quality_score:
                # 新的更好，替换最差的
                self._evict_entry(
                    worst_in_shot,
                    EvictionReason.SUPERSEDED,
                    replaced_by=entry.crop_path
                )
                print(f"[SmartRegistry] {entity_id}: 替换 shot {entry.shot_id} 中的低质量参考")
            else:
                reason = f"shot {entry.shot_id} 已有 {shot_count} 张，且质量不优于现有"
                print(f"[SmartRegistry] 拒绝注册 {entity_id}: {reason}")
                return False, reason

        # ── Check 5: 总容量限制 ────────────────────────────────────────────────
        total_count = self._count_active_refs(entity_id)
        if total_count >= cfg.max_refs_per_entity:
            # 触发淘汰，腾出空间
            # 策略：优先淘汰低于阈值的；如果都高于阈值，则淘汰最低的（如果新的比它高）
            evicted = self._evict_lowest_quality(entity_id)
            if not evicted:
                # 所有参考图都高于淘汰阈值，尝试替换最低质量的（如果新的更好）
                evicted = self._evict_if_better(entity_id, entry.quality_score)
                if not evicted:
                    reason = f"已达容量上限 {cfg.max_refs_per_entity}，且新参考图质量不优于现有最低"
                    print(f"[SmartRegistry] 拒绝注册 {entity_id}: {reason}")
                    return False, reason

        # ── 执行注册 ────────────────────────────────────────────────────────────
        self._insert_entry(entity_id, entry)

        # ── 检查是否应设为锚点 ──────────────────────────────────────────────────
        self._maybe_promote_to_anchor(entity_id, entry)

        print(f"[SmartRegistry] 注册成功 {entity_id} | shot={entry.shot_id} | "
              f"score={entry.quality_score:.3f} | id_conf={entry.id_confidence:.2f}")
        return True, "success"

    def register_batch(
        self,
        entity_id: str,
        entries: List[ReferenceEntry],
    ) -> Dict[str, Tuple[bool, str]]:
        """
        批量智能注册

        会按质量降序处理，优先注册高质量的
        """
        # 按质量降序排序
        sorted_entries = sorted(entries, key=lambda e: e.quality_score, reverse=True)
        results = {}

        for entry in sorted_entries:
            success, reason = self.register(entity_id, entry)
            results[entry.crop_path] = (success, reason)

        return results

    def _insert_entry(self, entity_id: str, entry: ReferenceEntry):
        """实际执行 INSERT"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO ref_entries
            (entity_id, shot_id, frame_path, crop_path, quality_score, source,
             created_at, id_confidence, embedding, pose_tag, lighting_tag, is_anchor, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (
            entity_id,
            entry.shot_id,
            entry.frame_path or "",
            entry.crop_path,
            entry.quality_score,
            entry.source,
            entry.created_at,
            entry.id_confidence,
            entry.embedding,
            entry.pose_tag,
            entry.lighting_tag,
            1 if entry.is_anchor else 0,
        ))
        conn.commit()
        conn.close()

    # ══════════════════════════════════════════════════════════════════════════════
    # 相似度检测
    # ══════════════════════════════════════════════════════════════════════════════

    def _check_duplicate(
        self,
        entity_id: str,
        entry: ReferenceEntry,
    ) -> Tuple[bool, Optional[str]]:
        """
        检查新参考图是否与已有的过于相似

        使用 CLIP embedding 余弦相似度
        """
        if self.clip_model is None:
            return False, None

        # 计算新图的 embedding
        try:
            new_embedding = self._compute_embedding(entry.crop_path)
        except Exception as e:
            print(f"[SmartRegistry] 计算 embedding 失败: {e}")
            return False, None

        # 获取该实体所有活跃参考图的 embedding
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT crop_path, embedding FROM ref_entries
            WHERE entity_id = ? AND is_active = 1 AND embedding IS NOT NULL
        """, (entity_id,)).fetchall()
        conn.close()

        for crop_path, emb_bytes in rows:
            if emb_bytes is None:
                continue
            existing_emb = np.frombuffer(emb_bytes, dtype=np.float32)
            similarity = self._cosine_similarity(new_embedding, existing_emb)

            if similarity >= self.config.similarity_threshold:
                return True, crop_path

        return False, None

    def _compute_embedding(self, image_path: str) -> np.ndarray:
        """计算图像的 CLIP embedding"""
        if self.clip_model is None:
            raise ValueError("CLIP model not loaded")

        import torch
        from PIL import Image

        # 假设 self.clip_model 是 (model, preprocess) tuple
        model, preprocess = self.clip_model

        # 获取模型所在设备
        device = next(model.parameters()).device

        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)  # 移动到模型所在设备

        with torch.no_grad():
            embedding = model.encode_image(image_input)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu().numpy().flatten().astype(np.float32)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    # ══════════════════════════════════════════════════════════════════════════════
    # 淘汰机制
    # ══════════════════════════════════════════════════════════════════════════════

    def _evict_entry(
        self,
        entry: ReferenceEntry,
        reason: EvictionReason,
        replaced_by: Optional[str] = None,
    ):
        """
        淘汰一条参考图记录

        不是物理删除，而是标记为 is_active=0，并记录淘汰日志
        """
        conn = sqlite3.connect(self.db_path)

        # 标记为非活跃
        conn.execute("""
            UPDATE ref_entries SET is_active = 0
            WHERE crop_path = ?
        """, (entry.crop_path,))

        # 记录淘汰日志
        conn.execute("""
            INSERT INTO eviction_log
            (entity_id, crop_path, quality_score, reason, evicted_at, replaced_by)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            entry.entity_id,
            entry.crop_path,
            entry.quality_score,
            reason.value,
            datetime.now().isoformat(),
            replaced_by,
        ))

        conn.commit()
        conn.close()

        print(f"[SmartRegistry] 淘汰 {entry.entity_id}: {entry.crop_path} "
              f"(reason={reason.value}, score={entry.quality_score:.3f})")

    def _evict_lowest_quality(self, entity_id: str) -> bool:
        """
        淘汰该实体质量最低的非锚点参考图

        Returns:
            是否成功淘汰
        """
        conn = sqlite3.connect(self.db_path)

        # 找质量最低的非锚点、非保护的参考图
        row = conn.execute("""
            SELECT entity_id, shot_id, frame_path, crop_path, quality_score,
                   source, created_at, id_confidence
            FROM ref_entries
            WHERE entity_id = ? AND is_active = 1 AND is_anchor = 0
                  AND quality_score < ?
            ORDER BY quality_score ASC
            LIMIT 1
        """, (entity_id, self.config.eviction_quality_threshold)).fetchone()

        conn.close()

        if row is None:
            return False

        entry = ReferenceEntry(
            entity_id=row[0],
            shot_id=row[1],
            frame_path=row[2],
            crop_path=row[3],
            quality_score=row[4],
            source=row[5],
            created_at=row[6],
            id_confidence=row[7],
        )

        self._evict_entry(entry, EvictionReason.LOW_QUALITY)
        return True

    def _evict_if_better(self, entity_id: str, new_quality: float) -> bool:
        """
        如果新参考图质量高于现有最低的，则淘汰最低的

        用于处理所有参考图都高于阈值但需要腾出空间的情况

        Args:
            entity_id: 实体 ID
            new_quality: 新参考图的质量分

        Returns:
            是否成功淘汰
        """
        conn = sqlite3.connect(self.db_path)

        # 找质量最低的非锚点参考图（不限于低于阈值的）
        row = conn.execute("""
            SELECT entity_id, shot_id, frame_path, crop_path, quality_score,
                   source, created_at, id_confidence
            FROM ref_entries
            WHERE entity_id = ? AND is_active = 1 AND is_anchor = 0
            ORDER BY quality_score ASC
            LIMIT 1
        """, (entity_id,)).fetchone()

        conn.close()

        if row is None:
            return False

        lowest_quality = row[4]

        # 只有新参考图质量更高时才替换
        if new_quality <= lowest_quality:
            return False

        entry = ReferenceEntry(
            entity_id=row[0],
            shot_id=row[1],
            frame_path=row[2],
            crop_path=row[3],
            quality_score=row[4],
            source=row[5],
            created_at=row[6],
            id_confidence=row[7],
        )

        self._evict_entry(entry, EvictionReason.SUPERSEDED)
        return True

    def run_eviction_audit(self, entity_id: Optional[str] = None) -> Dict[str, int]:
        """
        运行淘汰审计

        检查所有（或指定）实体的参考图，淘汰低质量/冗余的

        Returns:
            {entity_id: evicted_count}
        """
        results = {}

        if entity_id:
            entities = [entity_id]
        else:
            entities = self.get_all_entities()

        for eid in entities:
            evicted = 0

            # 1. 淘汰低质量
            while True:
                count = self._count_active_refs(eid)
                if count <= self.config.protected_anchor_count:
                    break
                if not self._evict_lowest_quality(eid):
                    break
                evicted += 1

            # 2. 淘汰冗余（相似度过高）
            evicted += self._evict_redundant(eid)

            if evicted > 0:
                results[eid] = evicted

        return results

    def _evict_redundant(self, entity_id: str) -> int:
        """淘汰相似度过高的冗余参考图"""
        if self.clip_model is None:
            return 0

        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT crop_path, embedding, quality_score, is_anchor
            FROM ref_entries
            WHERE entity_id = ? AND is_active = 1 AND embedding IS NOT NULL
            ORDER BY quality_score DESC
        """, (entity_id,)).fetchall()
        conn.close()

        if len(rows) < 2:
            return 0

        # 保留的参考图
        kept_embeddings = []
        to_evict = []

        for crop_path, emb_bytes, score, is_anchor in rows:
            if emb_bytes is None:
                continue

            emb = np.frombuffer(emb_bytes, dtype=np.float32)

            # 锚点始终保留
            if is_anchor:
                kept_embeddings.append((crop_path, emb))
                continue

            # 检查与已保留的是否相似
            is_redundant = False
            for _, kept_emb in kept_embeddings:
                if self._cosine_similarity(emb, kept_emb) >= self.config.similarity_threshold:
                    is_redundant = True
                    break

            if is_redundant:
                to_evict.append((crop_path, score))
            else:
                kept_embeddings.append((crop_path, emb))

        # 执行淘汰
        for crop_path, score in to_evict:
            entry = ReferenceEntry(
                entity_id=entity_id,
                shot_id=0,
                frame_path="",
                crop_path=crop_path,
                quality_score=score,
                source="",
            )
            self._evict_entry(entry, EvictionReason.REDUNDANT)

        return len(to_evict)

    # ══════════════════════════════════════════════════════════════════════════════
    # 锚点管理
    # ══════════════════════════════════════════════════════════════════════════════

    def _maybe_promote_to_anchor(self, entity_id: str, entry: ReferenceEntry):
        """
        检查是否应将新参考图提升为锚点

        锚点条件：
        1. 高质量 (>= 0.85)
        2. 早期 shot
        3. 正脸 (id_confidence >= 0.7)
        """
        # 已经是锚点
        if entry.is_anchor:
            return

        # 检查现有锚点数量
        conn = sqlite3.connect(self.db_path)
        anchor_count = conn.execute("""
            SELECT COUNT(*) FROM ref_entries
            WHERE entity_id = ? AND is_active = 1 AND is_anchor = 1
        """, (entity_id,)).fetchone()[0]
        conn.close()

        if anchor_count >= self.config.protected_anchor_count:
            return

        # 检查是否满足锚点条件
        is_high_quality = entry.quality_score >= 0.85
        is_frontal = entry.id_confidence >= 0.7 if entity_id.startswith("char_") else True
        is_early = entry.shot_id <= 3  # 前 3 个 shot

        if is_high_quality and is_frontal and is_early:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                UPDATE ref_entries SET is_anchor = 1
                WHERE crop_path = ?
            """, (entry.crop_path,))
            conn.commit()
            conn.close()
            print(f"[SmartRegistry] 提升为锚点: {entity_id} | {entry.crop_path}")

    def set_anchor(self, entity_id: str, crop_path: str, is_anchor: bool = True):
        """手动设置/取消锚点"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE ref_entries SET is_anchor = ?
            WHERE entity_id = ? AND crop_path = ?
        """, (1 if is_anchor else 0, entity_id, crop_path))
        conn.commit()
        conn.close()

    # ══════════════════════════════════════════════════════════════════════════════
    # 查询接口（兼容旧版）
    # ══════════════════════════════════════════════════════════════════════════════

    def query(
        self,
        entity_id: str,
        top_k: int = 3,
        min_quality: float = 0.3,
        anchor_strategy: str = "earliest_high_quality",
        active_only: bool = True,
    ) -> List[ReferenceEntry]:
        """
        查询参考图（兼容旧版接口）

        改进：
        1. 默认只返回活跃的（未被淘汰的）
        2. 锚点优先
        """
        conn = sqlite3.connect(self.db_path)

        active_clause = "AND is_active = 1" if active_only else ""

        if anchor_strategy == "earliest_high_quality":
            # 锚点优先 → 早期 shot 优先 → 质量优先
            order = "is_anchor DESC, shot_id ASC, quality_score DESC"
        elif anchor_strategy == "best_quality":
            order = "is_anchor DESC, quality_score DESC, shot_id ASC"
        else:
            order = "is_anchor DESC, shot_id DESC, quality_score DESC"

        rows = conn.execute(f"""
            SELECT entity_id, shot_id, frame_path, crop_path, quality_score,
                   source, created_at, id_confidence, pose_tag, lighting_tag, is_anchor
            FROM ref_entries
            WHERE entity_id = ? AND quality_score >= ? {active_clause}
            ORDER BY {order}
            LIMIT ?
        """, (entity_id, min_quality, top_k)).fetchall()
        conn.close()

        return [
            ReferenceEntry(
                entity_id=r[0],
                shot_id=r[1],
                frame_path=r[2],
                crop_path=r[3],
                quality_score=r[4],
                source=r[5],
                created_at=r[6],
                id_confidence=r[7],
                pose_tag=r[8],
                lighting_tag=r[9],
                is_anchor=bool(r[10]),
            )
            for r in rows
        ]

    def query_anchor(
        self,
        entity_id: str,
        min_quality: float = 0.5,
        high_quality_threshold: float = 0.85,
    ) -> Optional[ReferenceEntry]:
        """获取锚点参考图（兼容旧版接口）"""
        refs = self.query(entity_id, top_k=1, min_quality=min_quality)
        return refs[0] if refs else None

    def has_references(self, entity_id: str, active_only: bool = True) -> bool:
        """检查是否有参考图"""
        conn = sqlite3.connect(self.db_path)
        active_clause = "AND is_active = 1" if active_only else ""
        count = conn.execute(f"""
            SELECT COUNT(*) FROM ref_entries
            WHERE entity_id = ? {active_clause}
        """, (entity_id,)).fetchone()[0]
        conn.close()
        return count > 0

    def get_all_entities(self, active_only: bool = True) -> List[str]:
        """获取所有实体 ID"""
        conn = sqlite3.connect(self.db_path)
        active_clause = "WHERE is_active = 1" if active_only else ""
        rows = conn.execute(f"""
            SELECT DISTINCT entity_id FROM ref_entries {active_clause}
        """).fetchall()
        conn.close()
        return [r[0] for r in rows]

    # ══════════════════════════════════════════════════════════════════════════════
    # 辅助方法
    # ══════════════════════════════════════════════════════════════════════════════

    def _count_active_refs(self, entity_id: str) -> int:
        """统计活跃参考图数量"""
        conn = sqlite3.connect(self.db_path)
        count = conn.execute("""
            SELECT COUNT(*) FROM ref_entries
            WHERE entity_id = ? AND is_active = 1
        """, (entity_id,)).fetchone()[0]
        conn.close()
        return count

    def _count_refs_in_shot(self, entity_id: str, shot_id: int) -> int:
        """统计某 shot 中的参考图数量"""
        conn = sqlite3.connect(self.db_path)
        count = conn.execute("""
            SELECT COUNT(*) FROM ref_entries
            WHERE entity_id = ? AND shot_id = ? AND is_active = 1
        """, (entity_id, shot_id)).fetchone()[0]
        conn.close()
        return count

    def _get_worst_in_shot(self, entity_id: str, shot_id: int) -> Optional[ReferenceEntry]:
        """获取某 shot 中质量最低的参考图"""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("""
            SELECT entity_id, shot_id, frame_path, crop_path, quality_score,
                   source, created_at, id_confidence
            FROM ref_entries
            WHERE entity_id = ? AND shot_id = ? AND is_active = 1 AND is_anchor = 0
            ORDER BY quality_score ASC
            LIMIT 1
        """, (entity_id, shot_id)).fetchone()
        conn.close()

        if row is None:
            return None

        return ReferenceEntry(
            entity_id=row[0],
            shot_id=row[1],
            frame_path=row[2],
            crop_path=row[3],
            quality_score=row[4],
            source=row[5],
            created_at=row[6],
            id_confidence=row[7],
        )

    def stats(self) -> dict:
        """返回统计信息"""
        conn = sqlite3.connect(self.db_path)

        total_active = conn.execute(
            "SELECT COUNT(*) FROM ref_entries WHERE is_active = 1"
        ).fetchone()[0]

        total_evicted = conn.execute(
            "SELECT COUNT(*) FROM ref_entries WHERE is_active = 0"
        ).fetchone()[0]

        total_entities = conn.execute(
            "SELECT COUNT(DISTINCT entity_id) FROM ref_entries WHERE is_active = 1"
        ).fetchone()[0]

        total_anchors = conn.execute(
            "SELECT COUNT(*) FROM ref_entries WHERE is_active = 1 AND is_anchor = 1"
        ).fetchone()[0]

        eviction_by_reason = dict(conn.execute("""
            SELECT reason, COUNT(*) FROM eviction_log GROUP BY reason
        """).fetchall())

        conn.close()

        return {
            "active_references": total_active,
            "evicted_references": total_evicted,
            "total_entities": total_entities,
            "total_anchors": total_anchors,
            "eviction_by_reason": eviction_by_reason,
            "db_path": self.db_path,
        }

    def get_eviction_log(self, entity_id: Optional[str] = None, limit: int = 100) -> List[EvictionRecord]:
        """获取淘汰日志"""
        conn = sqlite3.connect(self.db_path)

        if entity_id:
            rows = conn.execute("""
                SELECT entity_id, crop_path, quality_score, reason, evicted_at, replaced_by
                FROM eviction_log
                WHERE entity_id = ?
                ORDER BY evicted_at DESC
                LIMIT ?
            """, (entity_id, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT entity_id, crop_path, quality_score, reason, evicted_at, replaced_by
                FROM eviction_log
                ORDER BY evicted_at DESC
                LIMIT ?
            """, (limit,)).fetchall()

        conn.close()

        return [
            EvictionRecord(
                entity_id=r[0],
                crop_path=r[1],
                quality_score=r[2],
                reason=r[3],
                evicted_at=r[4],
                replaced_by=r[5],
            )
            for r in rows
        ]


# ══════════════════════════════════════════════════════════════════════════════════
# VLM 驱动的智能审计 Agent
# ══════════════════════════════════════════════════════════════════════════════════

class ReferenceAuditAgent:
    """
    VLM 驱动的参考图审计 Agent

    功能：
    1. 分析参考图质量（姿态、光线、清晰度）
    2. 检测跨实体的视觉冲突（是否误标注）
    3. 建议哪些参考图应该被淘汰/保留
    """

    AUDIT_PROMPT = """你是一个专业的影视参考图审计员。

请分析以下参考图，并给出评估：

实体 ID: {entity_id}
实体类型: {entity_type}
实体描述: {entity_description}

请对每张图片评估以下维度（1-5分）：
1. identity_match: 是否匹配实体描述
2. image_quality: 图像清晰度
3. pose_usefulness: 姿态对于视频生成的有用程度（正脸 > 3/4侧脸 > 全侧 > 背影）
4. lighting_quality: 光线质量（自然、清晰 > 过曝/欠曝/逆光）

并给出：
- pose_tag: "frontal" | "three_quarter" | "profile" | "back"
- lighting_tag: "natural" | "bright" | "dim" | "backlit" | "mixed"
- keep_recommendation: true | false
- reason: 简要理由

输出 JSON 格式：
{{
  "images": [
    {{
      "path": "...",
      "identity_match": 4,
      "image_quality": 5,
      "pose_usefulness": 5,
      "lighting_quality": 4,
      "pose_tag": "frontal",
      "lighting_tag": "natural",
      "keep_recommendation": true,
      "reason": "High quality frontal face, ideal anchor"
    }},
    ...
  ],
  "summary": "整体评估总结",
  "suggested_evictions": ["path1", "path2"]  // 建议淘汰的路径
}}
"""

    def __init__(self, llm_client, registry: SmartEntityRegistry):
        self.llm = llm_client
        self.registry = registry

    def audit_entity(
        self,
        entity_id: str,
        entity_description: str = "",
    ) -> dict:
        """
        审计单个实体的所有参考图

        Returns:
            VLM 的评估结果
        """
        refs = self.registry.query(entity_id, top_k=20, min_quality=0.0)
        if not refs:
            return {"error": f"No references found for {entity_id}"}

        entity_type = "character" if entity_id.startswith("char_") else \
                      "location" if entity_id.startswith("loc_") else "object"

        # 构建 prompt
        image_paths = [r.crop_path for r in refs]

        # 这里需要 VLM 能够处理多图输入
        # 实际实现时根据具体 VLM API 调整
        prompt = self.AUDIT_PROMPT.format(
            entity_id=entity_id,
            entity_type=entity_type,
            entity_description=entity_description or entity_id,
        )

        # TODO: 调用 VLM 分析图片
        # result = self.llm.analyze_images(image_paths, prompt)

        # 占位返回
        return {
            "entity_id": entity_id,
            "image_count": len(refs),
            "status": "audit_pending",
            "note": "VLM integration required",
        }

    def auto_audit_all(self) -> dict:
        """
        自动审计所有实体

        Returns:
            {entity_id: audit_result}
        """
        entities = self.registry.get_all_entities()
        results = {}

        for entity_id in entities:
            results[entity_id] = self.audit_entity(entity_id)

        return results


# ══════════════════════════════════════════════════════════════════════════════════
# 快速测试
# ══════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_smart.db")

        config = RegistryConfig(
            min_quality_to_register=0.4,
            max_refs_per_entity=5,
            max_refs_per_shot=2,
        )

        registry = SmartEntityRegistry(db_path, config)

        # 模拟注册多个参考图
        print("\n=== 测试智能注册 ===")
        for shot_id in range(1, 6):
            for det_idx in range(3):
                quality = 0.3 + shot_id * 0.1 + det_idx * 0.05
                entry = ReferenceEntry(
                    entity_id="char_alex",
                    shot_id=shot_id,
                    frame_path=f"/frames/shot{shot_id}/frame.jpg",
                    crop_path=f"/crops/char_alex_shot{shot_id}_det{det_idx}.jpg",
                    quality_score=quality,
                    source="grounding",
                    id_confidence=0.5 + det_idx * 0.2,
                )
                success, reason = registry.register("char_alex", entry)
                status = "✓" if success else "✗"
                print(f"  {status} shot={shot_id} det={det_idx} score={quality:.2f} | {reason}")

        print("\n=== Registry 统计 ===")
        stats = registry.stats()
        for k, v in stats.items():
            print(f"  {k}: {v}")

        print("\n=== 查询结果 ===")
        refs = registry.query("char_alex", top_k=5)
        for r in refs:
            anchor = "🔒" if r.is_anchor else "  "
            print(f"  {anchor} shot={r.shot_id} score={r.quality_score:.2f} | {r.crop_path}")

        print("\n=== 淘汰日志 ===")
        eviction_log = registry.get_eviction_log("char_alex")
        for e in eviction_log[:5]:
            print(f"  {e.reason}: {e.crop_path} (score={e.quality_score:.2f})")
