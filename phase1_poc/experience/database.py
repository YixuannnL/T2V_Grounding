"""
experience/database.py

经验数据库：使用 SQLite 持久化存储生成经验

核心概念：
1. SceneFingerprint - 场景指纹，用于相似场景匹配
2. GenerationExperience - 单次生成的完整经验记录
3. ExperienceDatabase - SQLite 存储和查询接口

设计原则：
- 轻量级：SQLite 单文件数据库，无需额外服务
- 快速查询：基于场景指纹的相似度匹配
- 跨 session：经验在不同运行间持久保留
- 隐私友好：只存储统计信息，不存储原始视频/图片
"""

import os
import json
import sqlite3
import hashlib
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum


@dataclass
class SceneFingerprint:
    """
    场景指纹：用于快速匹配相似场景

    指纹设计考量：
    - 足够抽象：不包含具体实体名称，便于跨项目复用
    - 结构化：便于精确匹配和相似度计算
    - 可哈希：可以作为查询键
    """
    shot_type: str                    # "closeup" / "medium" / "wide" / "establishing"
    entity_types: List[str]           # ["character", "character", "object"] - 按类型列出
    entity_count: int                 # 总实体数
    has_character: bool               # 是否有角色
    has_multiple_characters: bool     # 是否有多个角色
    character_count: int              # 角色数量
    has_object: bool                  # 是否有物体
    has_location: bool                # 是否有场景/地点
    is_body_part_closeup: bool        # 是否是身体部位特写（手、脚等）
    has_interaction: bool             # 是否有交互动作

    def to_hash(self) -> str:
        """生成指纹哈希（用于精确匹配）"""
        # 排序 entity_types 确保一致性
        sorted_types = sorted(self.entity_types)
        key = f"{self.shot_type}|{','.join(sorted_types)}|{self.character_count}|{self.has_interaction}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def similarity(self, other: "SceneFingerprint") -> float:
        """
        计算与另一个指纹的相似度

        Returns:
            0-1 的相似度分数
        """
        score = 0.0
        total_weight = 0.0

        # 镜头类型（权重高）
        if self.shot_type == other.shot_type:
            score += 0.3
        elif self._shot_type_distance(self.shot_type, other.shot_type) == 1:
            score += 0.15
        total_weight += 0.3

        # 角色数量（权重高）
        if self.character_count == other.character_count:
            score += 0.25
        elif abs(self.character_count - other.character_count) == 1:
            score += 0.1
        total_weight += 0.25

        # 是否有多角色
        if self.has_multiple_characters == other.has_multiple_characters:
            score += 0.15
        total_weight += 0.15

        # 实体类型匹配
        type_overlap = len(set(self.entity_types) & set(other.entity_types))
        type_total = max(len(self.entity_types), len(other.entity_types), 1)
        score += 0.15 * (type_overlap / type_total)
        total_weight += 0.15

        # 特殊标记
        if self.is_body_part_closeup == other.is_body_part_closeup:
            score += 0.1
        total_weight += 0.1

        if self.has_interaction == other.has_interaction:
            score += 0.05
        total_weight += 0.05

        return score / total_weight if total_weight > 0 else 0.0

    def _shot_type_distance(self, a: str, b: str) -> int:
        """镜头类型之间的距离"""
        order = ["closeup", "medium", "wide", "establishing"]
        if a not in order or b not in order:
            return 99
        return abs(order.index(a) - order.index(b))


@dataclass
class GenerationExperience:
    """
    单次生成经验记录

    记录一次生成的完整上下文和结果，用于未来相似场景的参考。
    """
    # 标识
    experience_id: str                    # 唯一标识
    timestamp: str                        # 记录时间

    # 场景指纹
    fingerprint: SceneFingerprint

    # 生成配置
    generation_mode: str                  # "t2v" / "phantom"
    ip_adapter_scale: float               # 参考图权重
    num_inference_steps: int              # 推理步数
    guide_scale_text: float               # 文本引导强度

    # 重试信息
    total_attempts: int                   # 总尝试次数
    final_attempt_params: Dict[str, Any]  # 最终成功的参数

    # 问题与解决
    encountered_issues: List[str]         # 遇到的问题类别列表
    successful_strategy: Optional[str]    # 成功的重试策略
    failed_strategies: List[str]          # 失败的重试策略

    # 结果
    final_score: float                    # 最终 critique 分数
    success: bool                         # 是否成功（通过 critique）

    # 建议（从经验中提炼）
    lessons_learned: List[str]            # 学到的经验教训

    # 元数据
    project_hint: str = ""                # 项目提示（可选，如 "cinematic", "anime"）
    additional_metadata: Dict = field(default_factory=dict)


# ── SQLite Schema ────────────────────────────────────────────────────────────

SCHEMA = """
-- 主经验表
CREATE TABLE IF NOT EXISTS experiences (
    experience_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    fingerprint_hash TEXT NOT NULL,
    fingerprint_json TEXT NOT NULL,
    generation_mode TEXT NOT NULL,
    ip_adapter_scale REAL,
    num_inference_steps INTEGER,
    guide_scale_text REAL,
    total_attempts INTEGER,
    final_attempt_params TEXT,
    encountered_issues TEXT,
    successful_strategy TEXT,
    failed_strategies TEXT,
    final_score REAL,
    success INTEGER,
    lessons_learned TEXT,
    project_hint TEXT,
    additional_metadata TEXT
);

-- 指纹哈希索引（加速精确匹配）
CREATE INDEX IF NOT EXISTS idx_fingerprint_hash
ON experiences (fingerprint_hash);

-- 成功经验索引
CREATE INDEX IF NOT EXISTS idx_success
ON experiences (success);

-- 时间索引
CREATE INDEX IF NOT EXISTS idx_timestamp
ON experiences (timestamp DESC);

-- 场景类型索引
CREATE INDEX IF NOT EXISTS idx_shot_type
ON experiences (json_extract(fingerprint_json, '$.shot_type'));

-- 统计表：记录各类问题的解决成功率
CREATE TABLE IF NOT EXISTS issue_stats (
    issue_category TEXT PRIMARY KEY,
    total_occurrences INTEGER DEFAULT 0,
    resolved_count INTEGER DEFAULT 0,
    avg_attempts_to_resolve REAL DEFAULT 0,
    most_effective_strategy TEXT,
    last_updated TEXT
);
"""


class ExperienceDatabase:
    """
    经验数据库

    使用 SQLite 持久化存储生成经验，支持：
    - 基于场景指纹的相似经验查询
    - 经验统计和分析
    - 跨 session 持久化

    Usage:
        db = ExperienceDatabase("./experience.db")

        # 记录经验
        db.record_experience(experience)

        # 查询相似经验
        similar = db.find_similar_experiences(fingerprint, top_k=5)

        # 获取问题统计
        stats = db.get_issue_stats("identity")
    """

    def __init__(self, db_path: str = "./experience.db", verbose: bool = True):
        self.db_path = db_path
        self.verbose = verbose

        # 确保目录存在
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)

        # 初始化数据库
        self._init_db()

        if self.verbose:
            count = self.get_total_count()
            print(f"[ExperienceDB] 已加载 {count} 条历史经验 from {db_path}")

    def _init_db(self):
        """初始化数据库 schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── 写入操作 ────────────────────────────────────────────────────────────

    def record_experience(self, exp: GenerationExperience):
        """
        记录一条生成经验

        Args:
            exp: GenerationExperience 对象
        """
        fingerprint_hash = exp.fingerprint.to_hash()
        fingerprint_json = json.dumps(asdict(exp.fingerprint))

        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO experiences (
                    experience_id, timestamp, fingerprint_hash, fingerprint_json,
                    generation_mode, ip_adapter_scale, num_inference_steps, guide_scale_text,
                    total_attempts, final_attempt_params,
                    encountered_issues, successful_strategy, failed_strategies,
                    final_score, success, lessons_learned,
                    project_hint, additional_metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                exp.experience_id,
                exp.timestamp,
                fingerprint_hash,
                fingerprint_json,
                exp.generation_mode,
                exp.ip_adapter_scale,
                exp.num_inference_steps,
                exp.guide_scale_text,
                exp.total_attempts,
                json.dumps(exp.final_attempt_params),
                json.dumps(exp.encountered_issues),
                exp.successful_strategy,
                json.dumps(exp.failed_strategies),
                exp.final_score,
                1 if exp.success else 0,
                json.dumps(exp.lessons_learned),
                exp.project_hint,
                json.dumps(exp.additional_metadata),
            ))

            # 更新问题统计
            self._update_issue_stats(conn, exp)

            conn.commit()

        if self.verbose:
            status = "✅" if exp.success else "❌"
            print(f"[ExperienceDB] {status} 记录经验: {exp.experience_id} "
                  f"(score={exp.final_score:.2f}, attempts={exp.total_attempts})")

    def _update_issue_stats(self, conn: sqlite3.Connection, exp: GenerationExperience):
        """更新问题统计"""
        for issue_cat in exp.encountered_issues:
            # 获取当前统计
            row = conn.execute(
                "SELECT * FROM issue_stats WHERE issue_category = ?",
                (issue_cat,)
            ).fetchone()

            if row:
                # 更新现有记录
                total = row["total_occurrences"] + 1
                resolved = row["resolved_count"] + (1 if exp.success else 0)
                avg_attempts = (row["avg_attempts_to_resolve"] * row["total_occurrences"] +
                               exp.total_attempts) / total

                # 如果成功且策略更有效，更新最有效策略
                most_effective = row["most_effective_strategy"]
                if exp.success and exp.successful_strategy:
                    if not most_effective or exp.total_attempts <= (row["avg_attempts_to_resolve"] or 99):
                        most_effective = exp.successful_strategy

                conn.execute("""
                    UPDATE issue_stats SET
                        total_occurrences = ?,
                        resolved_count = ?,
                        avg_attempts_to_resolve = ?,
                        most_effective_strategy = ?,
                        last_updated = ?
                    WHERE issue_category = ?
                """, (total, resolved, avg_attempts, most_effective,
                      datetime.now().isoformat(), issue_cat))
            else:
                # 插入新记录
                conn.execute("""
                    INSERT INTO issue_stats (
                        issue_category, total_occurrences, resolved_count,
                        avg_attempts_to_resolve, most_effective_strategy, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    issue_cat,
                    1,
                    1 if exp.success else 0,
                    exp.total_attempts,
                    exp.successful_strategy if exp.success else None,
                    datetime.now().isoformat(),
                ))

    # ── 查询操作 ────────────────────────────────────────────────────────────

    def find_similar_experiences(
        self,
        fingerprint: SceneFingerprint,
        top_k: int = 5,
        min_similarity: float = 0.6,
        success_only: bool = False,
    ) -> List[Tuple[GenerationExperience, float]]:
        """
        查找相似场景的经验

        Args:
            fingerprint: 当前场景指纹
            top_k: 返回最相似的 k 条
            min_similarity: 最低相似度阈值
            success_only: 是否只返回成功的经验

        Returns:
            [(experience, similarity_score), ...] 按相似度降序排列
        """
        with self._get_conn() as conn:
            # 先尝试精确匹配
            fp_hash = fingerprint.to_hash()
            exact_matches = conn.execute("""
                SELECT * FROM experiences
                WHERE fingerprint_hash = ?
                ORDER BY success DESC, final_score DESC, timestamp DESC
                LIMIT ?
            """, (fp_hash, top_k)).fetchall()

            if exact_matches:
                results = []
                for row in exact_matches:
                    if success_only and not row["success"]:
                        continue
                    exp = self._row_to_experience(row)
                    results.append((exp, 1.0))  # 精确匹配，相似度 = 1.0
                if results:
                    if self.verbose:
                        print(f"[ExperienceDB] 找到 {len(results)} 条精确匹配经验")
                    return results[:top_k]

            # 没有精确匹配，进行相似度查询
            # 为了效率，先筛选出相同 shot_type 的记录
            query = """
                SELECT * FROM experiences
                WHERE json_extract(fingerprint_json, '$.shot_type') = ?
            """
            params = [fingerprint.shot_type]

            if success_only:
                query += " AND success = 1"

            query += " ORDER BY final_score DESC, timestamp DESC LIMIT 100"

            candidates = conn.execute(query, params).fetchall()

            # 计算相似度并排序
            scored_results = []
            for row in candidates:
                exp = self._row_to_experience(row)
                sim = fingerprint.similarity(exp.fingerprint)
                if sim >= min_similarity:
                    scored_results.append((exp, sim))

            # 按相似度排序
            scored_results.sort(key=lambda x: x[1], reverse=True)

            if self.verbose and scored_results:
                print(f"[ExperienceDB] 找到 {len(scored_results)} 条相似经验 "
                      f"(相似度 >= {min_similarity:.2f})")

            return scored_results[:top_k]

    def get_issue_stats(self, issue_category: str) -> Optional[Dict]:
        """
        获取特定问题类别的统计信息

        Returns:
            {
                "total_occurrences": int,
                "resolved_count": int,
                "resolution_rate": float,
                "avg_attempts": float,
                "most_effective_strategy": str,
            }
        """
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM issue_stats WHERE issue_category = ?",
                (issue_category,)
            ).fetchone()

            if not row:
                return None

            total = row["total_occurrences"] or 1
            resolved = row["resolved_count"] or 0

            return {
                "total_occurrences": total,
                "resolved_count": resolved,
                "resolution_rate": resolved / total,
                "avg_attempts": row["avg_attempts_to_resolve"] or 0,
                "most_effective_strategy": row["most_effective_strategy"],
            }

    def get_all_issue_stats(self) -> Dict[str, Dict]:
        """获取所有问题类别的统计"""
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM issue_stats").fetchall()
            result = {}
            for row in rows:
                total = row["total_occurrences"] or 1
                resolved = row["resolved_count"] or 0
                result[row["issue_category"]] = {
                    "total_occurrences": total,
                    "resolved_count": resolved,
                    "resolution_rate": resolved / total,
                    "avg_attempts": row["avg_attempts_to_resolve"] or 0,
                    "most_effective_strategy": row["most_effective_strategy"],
                }
            return result

    def get_total_count(self) -> int:
        """获取总经验数"""
        with self._get_conn() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM experiences").fetchone()
            return row["cnt"] if row else 0

    def get_success_rate(self) -> float:
        """获取整体成功率"""
        with self._get_conn() as conn:
            row = conn.execute("""
                SELECT COUNT(*) as total, SUM(success) as success_count
                FROM experiences
            """).fetchone()
            if row and row["total"] > 0:
                return (row["success_count"] or 0) / row["total"]
            return 0.0

    # ── 辅助方法 ────────────────────────────────────────────────────────────

    def _row_to_experience(self, row: sqlite3.Row) -> GenerationExperience:
        """将数据库行转换为 GenerationExperience 对象"""
        fp_data = json.loads(row["fingerprint_json"])
        fingerprint = SceneFingerprint(**fp_data)

        return GenerationExperience(
            experience_id=row["experience_id"],
            timestamp=row["timestamp"],
            fingerprint=fingerprint,
            generation_mode=row["generation_mode"],
            ip_adapter_scale=row["ip_adapter_scale"] or 0.6,
            num_inference_steps=row["num_inference_steps"] or 50,
            guide_scale_text=row["guide_scale_text"] or 7.5,
            total_attempts=row["total_attempts"] or 1,
            final_attempt_params=json.loads(row["final_attempt_params"] or "{}"),
            encountered_issues=json.loads(row["encountered_issues"] or "[]"),
            successful_strategy=row["successful_strategy"],
            failed_strategies=json.loads(row["failed_strategies"] or "[]"),
            final_score=row["final_score"] or 0.0,
            success=bool(row["success"]),
            lessons_learned=json.loads(row["lessons_learned"] or "[]"),
            project_hint=row["project_hint"] or "",
            additional_metadata=json.loads(row["additional_metadata"] or "{}"),
        )


# ── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    import uuid

    # 使用临时数据库测试
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        test_db_path = f.name

    db = ExperienceDatabase(test_db_path, verbose=True)

    # 创建测试指纹
    fp1 = SceneFingerprint(
        shot_type="medium",
        entity_types=["character", "character"],
        entity_count=2,
        has_character=True,
        has_multiple_characters=True,
        character_count=2,
        has_object=False,
        has_location=True,
        is_body_part_closeup=False,
        has_interaction=True,
    )

    # 创建测试经验
    exp1 = GenerationExperience(
        experience_id=str(uuid.uuid4()),
        timestamp=datetime.now().isoformat(),
        fingerprint=fp1,
        generation_mode="phantom",
        ip_adapter_scale=0.75,
        num_inference_steps=50,
        guide_scale_text=7.5,
        total_attempts=2,
        final_attempt_params={"ip_adapter_scale": 0.75, "seed": 12345},
        encountered_issues=["identity"],
        successful_strategy="increase_ip_scale",
        failed_strategies=["change_seed"],
        final_score=0.82,
        success=True,
        lessons_learned=["双人场景需要更高的 ip_adapter_scale"],
    )

    # 记录经验
    db.record_experience(exp1)

    # 查询相似经验
    fp2 = SceneFingerprint(
        shot_type="medium",
        entity_types=["character", "character", "object"],
        entity_count=3,
        has_character=True,
        has_multiple_characters=True,
        character_count=2,
        has_object=True,
        has_location=True,
        is_body_part_closeup=False,
        has_interaction=True,
    )

    print("\n查询相似经验:")
    similar = db.find_similar_experiences(fp2, top_k=3)
    for exp, sim in similar:
        print(f"  相似度: {sim:.2f} | score: {exp.final_score:.2f} | success: {exp.success}")

    # 获取统计
    print("\n问题统计:")
    stats = db.get_all_issue_stats()
    for cat, stat in stats.items():
        print(f"  {cat}: 解决率={stat['resolution_rate']:.1%}, "
              f"平均尝试={stat['avg_attempts']:.1f}")

    # 清理
    os.unlink(test_db_path)
