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
                created_at  TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entity ON ref_entries(entity_id)")
        conn.commit()
        conn.close()

    # ── 写入 ─────────────────────────────────────────────────────────────────
    def register(self, entity_id: str, entry: ReferenceEntry):
        """注册一条新参考记录"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO ref_entries
            (entity_id, shot_id, frame_path, crop_path, quality_score, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            entity_id,
            entry.shot_id,
            entry.frame_path or "",
            entry.crop_path,
            entry.quality_score,
            entry.source,
            entry.created_at,
        ))
        conn.commit()
        conn.close()
        print(f"[Registry] 注册 {entity_id} | shot={entry.shot_id} | score={entry.quality_score:.3f}")

    def register_batch(self, entity_id: str, entries: List[ReferenceEntry]):
        for e in entries:
            self.register(entity_id, e)

    # ── 查询 ─────────────────────────────────────────────────────────────────
    def query(
        self,
        entity_id: str,
        top_k: int = 3,
        min_quality: float = 0.3,
        prefer_recent: bool = True,
    ) -> List[ReferenceEntry]:
        """
        查询某实体的最优参考帧

        Args:
            entity_id:     实体 ID
            top_k:         返回条数
            min_quality:   最低质量分过滤
            prefer_recent: True=按 shot_id 降序（最近优先），False=按质量降序
        """
        conn = sqlite3.connect(self.db_path)
        order = "shot_id DESC, quality_score DESC" if prefer_recent else "quality_score DESC"
        rows = conn.execute(f"""
            SELECT entity_id, shot_id, frame_path, crop_path, quality_score, source, created_at
            FROM ref_entries
            WHERE entity_id = ? AND quality_score >= ?
            ORDER BY {order}
            LIMIT ?
        """, (entity_id, min_quality, top_k)).fetchall()
        conn.close()

        return [ReferenceEntry(*row) for row in rows]

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
