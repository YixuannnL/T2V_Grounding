"""
reference_manager/registry.py
职责: 持久化管理每个实体在历史镜头中的视觉参考记录
存储后端: SQLite（PoC 阶段）
"""

import os
import json
import sqlite3
from dataclasses import dataclass, asdict
from typing import List, Optional
from datetime import datetime


@dataclass
class ReferenceEntry:
    entity_id: str
    shot_id: int              # 来自哪个镜头（0 = 冷启动 T2I 生成）
    frame_path: str           # 原始帧路径（可为空，如 T2I 生成）
    crop_path: str            # 裁切/参考图路径
    quality_score: float      # 综合质量分
    source: str               # "grounding" | "bootstrapped" | "manual"
    created_at: str = ""
    id_confidence: float = 1.0  # 人脸检测置信度（仅 character 有意义，其他类型默认 1.0）

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class EntityRegistry:
    """
    主体视觉参考库

    使用示例:
        registry = EntityRegistry("./registry.db")
        registry.register("char_alex", ReferenceEntry(...))
        refs = registry.query("char_alex", top_k=3)
    """

    def __init__(self, db_path: str = "./entity_registry.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ref_entries (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id   TEXT NOT NULL,
                shot_id     INTEGER NOT NULL,
                frame_path  TEXT,
                crop_path   TEXT NOT NULL,
                quality_score REAL NOT NULL,
                source      TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                id_confidence REAL DEFAULT 1.0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entity ON ref_entries(entity_id)")
        # 兼容旧数据库：如果 id_confidence 列不存在，添加它
        try:
            conn.execute("ALTER TABLE ref_entries ADD COLUMN id_confidence REAL DEFAULT 1.0")
        except sqlite3.OperationalError:
            pass  # 列已存在
        conn.commit()
        conn.close()

    # ── 写入 ─────────────────────────────────────────────────────────────────
    def register(self, entity_id: str, entry: ReferenceEntry):
        """注册一条新参考记录"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO ref_entries
            (entity_id, shot_id, frame_path, crop_path, quality_score, source, created_at, id_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entity_id,
            entry.shot_id,
            entry.frame_path or "",
            entry.crop_path,
            entry.quality_score,
            entry.source,
            entry.created_at,
            entry.id_confidence,
        ))
        conn.commit()
        conn.close()
        print(f"[Registry] 注册 {entity_id} | shot={entry.shot_id} | score={entry.quality_score:.3f} | id_conf={entry.id_confidence:.2f}")

    def register_batch(self, entity_id: str, entries: List[ReferenceEntry]):
        for e in entries:
            self.register(entity_id, e)

    # ── 查询 ─────────────────────────────────────────────────────────────────
    def query(
        self,
        entity_id: str,
        top_k: int = 3,
        min_quality: float = 0.3,
        anchor_strategy: str = "earliest_good",
    ) -> List[ReferenceEntry]:
        """
        查询某实体的最优参考帧

        Args:
            entity_id:        实体 ID
            top_k:            返回条数
            min_quality:      最低质量分过滤
            anchor_strategy:  选择策略
                - "earliest_good": 【推荐】优先最早出现的高质量参考图，防止误差累积
                - "best_quality":  纯按质量排序（可能选到后面已偏移的 shot）
                - "most_recent":   最近优先（旧逻辑，会导致误差累积）

        Returns:
            按策略排序的参考图列表
        """
        conn = sqlite3.connect(self.db_path)

        if anchor_strategy == "earliest_good":
            # 核心策略：优先最早 shot，同 shot 内按质量排序
            # 这样可以锚定最初的高质量参考，避免误差累积
            order = "shot_id ASC, quality_score DESC"
        elif anchor_strategy == "best_quality":
            order = "quality_score DESC, shot_id ASC"
        else:  # most_recent (旧逻辑，保留兼容性)
            order = "shot_id DESC, quality_score DESC"

        rows = conn.execute(f"""
            SELECT entity_id, shot_id, frame_path, crop_path, quality_score, source, created_at, id_confidence
            FROM ref_entries
            WHERE entity_id = ? AND quality_score >= ?
            ORDER BY {order}
            LIMIT ?
        """, (entity_id, min_quality, top_k)).fetchall()
        conn.close()

        # 兼容旧数据：如果只有 7 列，补上默认的 id_confidence=1.0
        results = []
        for row in rows:
            if len(row) == 7:
                results.append(ReferenceEntry(*row, id_confidence=1.0))
            else:
                results.append(ReferenceEntry(*row))
        return results

    def query_anchor(
        self,
        entity_id: str,
        min_quality: float = 0.5,
        high_quality_threshold: float = 0.85,
    ) -> Optional[ReferenceEntry]:
        """
        获取实体的"锚点"参考图：最早出现的高质量正脸参考

        改进策略 "earliest_high_quality"：
        ─────────────────────────────────────────────────────────
        核心问题：Shot 1 可能是侧脸/部分脸，Shot 2+ 的 close-up 可能有更清晰的正脸。

        规则（按优先级）：
        1. 如果最早 shot 有 ≥ high_quality_threshold (默认 0.85) 的参考图 → 选它
        2. 如果最早 shot 只有中等质量 (min_quality ~ high_quality_threshold)：
           - 查找所有 shot 中最早的高质量参考图 (≥ high_quality_threshold)
           - 如果找到 → 选它（高质量正脸优先于早期出现）
           - 如果没有 → 回退选最早 shot 的最佳参考（防止误差累积）
        3. 同等高质量时，选 shot_id 更小的（更早出现）

        Args:
            entity_id:              实体 ID
            min_quality:            基础质量阈值
            high_quality_threshold: 高质量阈值（超过此分数视为"大正脸"）

        Returns:
            最佳锚点参考图，或 None
        """
        conn = sqlite3.connect(self.db_path)

        # Step 1: 找最早 shot 中质量最高的参考图
        earliest_best = conn.execute("""
            SELECT entity_id, shot_id, frame_path, crop_path, quality_score, source, created_at, id_confidence
            FROM ref_entries
            WHERE entity_id = ? AND quality_score >= ?
            ORDER BY shot_id ASC, quality_score DESC
            LIMIT 1
        """, (entity_id, min_quality)).fetchone()

        if earliest_best is None:
            conn.close()
            return None

        earliest_score = earliest_best[4]  # quality_score
        earliest_shot = earliest_best[1]   # shot_id
        earliest_id_conf = earliest_best[7] if len(earliest_best) > 7 else 1.0  # id_confidence

        # Step 2: 如果最早 shot 已经是高质量，直接返回
        if earliest_score >= high_quality_threshold:
            conn.close()
            print(f"[Registry] 锚点选择: {entity_id} | 最早shot已高质量 "
                  f"(shot={earliest_shot}, score={earliest_score:.2f}, id_conf={earliest_id_conf:.2f})")
            if len(earliest_best) == 7:
                return ReferenceEntry(*earliest_best, id_confidence=1.0)
            return ReferenceEntry(*earliest_best)

        # Step 3: 最早 shot 质量中等，查找后续 shot 中是否有高质量参考
        # 找所有 shot 中最早的高质量参考图
        high_quality_anchor = conn.execute("""
            SELECT entity_id, shot_id, frame_path, crop_path, quality_score, source, created_at, id_confidence
            FROM ref_entries
            WHERE entity_id = ? AND quality_score >= ?
            ORDER BY shot_id ASC, quality_score DESC
            LIMIT 1
        """, (entity_id, high_quality_threshold)).fetchone()

        conn.close()

        if high_quality_anchor is not None:
            hq_shot = high_quality_anchor[1]
            hq_score = high_quality_anchor[4]
            hq_id_conf = high_quality_anchor[7] if len(high_quality_anchor) > 7 else 1.0
            print(f"[Registry] 锚点选择: {entity_id} | 选择后续高质量正脸 "
                  f"(shot={hq_shot}, score={hq_score:.2f}, id_conf={hq_id_conf:.2f}) 优于最早shot "
                  f"(shot={earliest_shot}, score={earliest_score:.2f})")
            if len(high_quality_anchor) == 7:
                return ReferenceEntry(*high_quality_anchor, id_confidence=1.0)
            return ReferenceEntry(*high_quality_anchor)

        # Step 4: 没有高质量参考，回退选最早 shot（防误差累积）
        print(f"[Registry] 锚点选择: {entity_id} | 无高质量参考，回退选最早shot "
              f"(shot={earliest_shot}, score={earliest_score:.2f}, id_conf={earliest_id_conf:.2f})")
        if len(earliest_best) == 7:
            return ReferenceEntry(*earliest_best, id_confidence=1.0)
        return ReferenceEntry(*earliest_best)

    def query_anchor_location(
        self,
        entity_id: str,
        min_quality: float = 0.3,
        high_quality_threshold: float = 0.7,
        quality_gap_ratio: float = 3.0,
    ) -> Optional[ReferenceEntry]:
        """
        获取 Location 实体的"锚点"参考图

        策略 "earliest_unless_much_worse"：
        ─────────────────────────────────────────────────────────
        核心原则：
          - 默认锚定最早 shot 的背景（保持场景一致性）
          - 除非后续 shot 有 **明显** 更好的背景（质量差距超过阈值）

        判断逻辑：
          1. 找最早 shot 中质量最高的背景 (earliest_best)
          2. 找全部 shot 中质量最高的背景 (global_best)
          3. 如果 earliest_best 质量已经 >= high_quality_threshold → 选它
          4. 如果 global_best / earliest_best > quality_gap_ratio → 选 global_best
          5. 否则 → 选 earliest_best（默认锚定早期）

        Args:
            entity_id:              实体 ID
            min_quality:            基础质量阈值
            high_quality_threshold: 高质量阈值（超过此分数不再考虑切换）
            quality_gap_ratio:      质量差距倍数阈值（后续背景需要好这么多倍才切换）

        Returns:
            最佳锚点参考图，或 None
        """
        conn = sqlite3.connect(self.db_path)

        # Step 1: 找最早 shot 中质量最高的背景
        earliest_best = conn.execute("""
            SELECT entity_id, shot_id, frame_path, crop_path, quality_score, source, created_at
            FROM ref_entries
            WHERE entity_id = ? AND quality_score >= ?
            ORDER BY shot_id ASC, quality_score DESC
            LIMIT 1
        """, (entity_id, min_quality)).fetchone()

        if earliest_best is None:
            conn.close()
            return None

        earliest_score = earliest_best[4]  # quality_score
        earliest_shot = earliest_best[1]   # shot_id

        # Step 2: 如果最早 shot 已经是高质量，直接返回
        if earliest_score >= high_quality_threshold:
            conn.close()
            print(f"[Registry] Location 锚点: {entity_id} | 最早shot已高质量 "
                  f"(shot={earliest_shot}, score={earliest_score:.2f})")
            return ReferenceEntry(*earliest_best)

        # Step 3: 找全部 shot 中质量最高的背景
        global_best = conn.execute("""
            SELECT entity_id, shot_id, frame_path, crop_path, quality_score, source, created_at
            FROM ref_entries
            WHERE entity_id = ? AND quality_score >= ?
            ORDER BY quality_score DESC, shot_id ASC
            LIMIT 1
        """, (entity_id, min_quality)).fetchone()

        conn.close()

        if global_best is None:
            return ReferenceEntry(*earliest_best)

        global_score = global_best[4]
        global_shot = global_best[1]

        # Step 4: 判断质量差距是否超过阈值
        # 避免除零，且如果 earliest_score 很低，gap 会很大
        if earliest_score > 0 and global_score / earliest_score >= quality_gap_ratio:
            print(f"[Registry] Location 锚点: {entity_id} | 选择后续高质量背景 "
                  f"(shot={global_shot}, score={global_score:.2f}) "
                  f"因为比最早shot (shot={earliest_shot}, score={earliest_score:.2f}) "
                  f"好 {global_score/earliest_score:.1f}x (阈值={quality_gap_ratio}x)")
            return ReferenceEntry(*global_best)

        # Step 5: 质量差距不够大，回退选最早 shot
        print(f"[Registry] Location 锚点: {entity_id} | 锚定最早shot "
              f"(shot={earliest_shot}, score={earliest_score:.2f}) "
              f"虽然有更好的 (shot={global_shot}, score={global_score:.2f}) "
              f"但差距只有 {global_score/max(earliest_score, 0.01):.1f}x < {quality_gap_ratio}x")
        return ReferenceEntry(*earliest_best)

    def has_references(self, entity_id: str) -> bool:
        """检查某实体是否有历史参考"""
        conn = sqlite3.connect(self.db_path)
        count = conn.execute(
            "SELECT COUNT(*) FROM ref_entries WHERE entity_id = ?", (entity_id,)
        ).fetchone()[0]
        conn.close()
        return count > 0

    def get_all_entities(self) -> List[str]:
        """返回所有已注册的实体 ID 列表"""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT DISTINCT entity_id FROM ref_entries").fetchall()
        conn.close()
        return [r[0] for r in rows]

    def get_registered_foreground_entities(self, exclude_types: List[str] = None) -> List[str]:
        """
        获取已注册的前景实体 ID 列表（用于 location grounding 时排除）

        Args:
            exclude_types: 要排除的实体类型前缀列表，默认 ["loc_"]
                           因为 entity_id 格式为 char_xxx / obj_xxx / loc_xxx

        Returns:
            非 location 类型的已注册实体 ID 列表
        """
        if exclude_types is None:
            exclude_types = ["loc_"]

        all_entities = self.get_all_entities()
        foreground_entities = []

        for entity_id in all_entities:
            is_excluded = any(entity_id.startswith(prefix) for prefix in exclude_types)
            if not is_excluded:
                foreground_entities.append(entity_id)

        return foreground_entities

    def stats(self) -> dict:
        """返回 registry 统计信息"""
        conn = sqlite3.connect(self.db_path)
        total = conn.execute("SELECT COUNT(*) FROM ref_entries").fetchone()[0]
        entities = conn.execute("SELECT COUNT(DISTINCT entity_id) FROM ref_entries").fetchone()[0]
        conn.close()
        return {"total_references": total, "total_entities": entities, "db_path": self.db_path}

    # ── 冷启动参考图（T2I 生成）─────────────────────────────────────────────
    def bootstrap_entity(
        self,
        entity_id: str,
        entity_description: str,
        output_dir: str,
        t2i_fn=None,
    ) -> Optional[ReferenceEntry]:
        """
        当实体无历史时，调用 T2I 生成参考图（冷启动）

        Args:
            entity_id:          实体 ID
            entity_description: 实体文本描述
            output_dir:         参考图保存目录
            t2i_fn:             T2I 生成函数，签名: (prompt: str) -> PIL.Image
                                若为 None 则尝试使用 DALL-E 3 或 SDXL
        """
        if self.has_references(entity_id):
            print(f"[Registry] {entity_id} 已有参考，跳过冷启动")
            return None

        print(f"[Registry] 冷启动: 为 {entity_id} 生成参考图...")
        os.makedirs(output_dir, exist_ok=True)

        prompt = (
            f"a photo of {entity_description}, "
            "front view, clear face, standing, white background, "
            "photorealistic, high quality, detailed"
        )

        if t2i_fn is not None:
            pil_image = t2i_fn(prompt)
        else:
            pil_image = self._default_t2i(prompt)

        if pil_image is None:
            print(f"[Registry] 冷启动失败: {entity_id}")
            return None

        crop_path = os.path.join(output_dir, f"{entity_id}_bootstrap.png")
        pil_image.save(crop_path)

        entry = ReferenceEntry(
            entity_id=entity_id,
            shot_id=0,
            frame_path="",
            crop_path=crop_path,
            quality_score=0.7,  # 生成图默认中等质量
            source="bootstrapped",
        )
        self.register(entity_id, entry)
        return entry

    def _default_t2i(self, prompt: str):
        """默认 T2I：尝试用 diffusers SDXL（本地）或 DALL-E 3（远程）"""
        # 优先尝试本地 SDXL
        try:
            from diffusers import StableDiffusionXLPipeline
            import torch
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
            ).to("cuda")
            image = pipe(prompt, num_inference_steps=30).images[0]
            return image
        except Exception:
            pass

        # fallback: DALL-E 3
        try:
            import openai
            import requests
            from PIL import Image
            from io import BytesIO
            client = openai.OpenAI()
            resp = client.images.generate(model="dall-e-3", prompt=prompt, size="1024x1024")
            img_url = resp.data[0].url
            img_data = requests.get(img_url).content
            return Image.open(BytesIO(img_data))
        except Exception as e:
            print(f"[Registry] T2I fallback 失败: {e}")
            return None


# ── 快速测试 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = EntityRegistry(os.path.join(tmpdir, "test.db"))

        # 注册几条测试记录
        for i in range(3):
            entry = ReferenceEntry(
                entity_id="char_alex",
                shot_id=i + 1,
                frame_path=f"/shots/shot{i+1}/frame_0100.jpg",
                crop_path=f"/crops/char_alex_shot{i+1}.jpg",
                quality_score=0.5 + i * 0.1,
                source="grounding",
            )
            registry.register("char_alex", entry)

        # 查询
        refs = registry.query("char_alex", top_k=2)
        print(f"查询结果 (top 2):")
        for r in refs:
            print(f"  shot={r.shot_id}  score={r.quality_score}  path={r.crop_path}")

        print(f"\nRegistry stats: {registry.stats()}")
