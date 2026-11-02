"""
Database access and persistence manager for Decoration Photo Filtering & Management.
Manages SQLite schemas, category rules, image records, batch tracking, and duplicate indexing.
"""

import sqlite3
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager
from datetime import datetime
import imagehash

from database.models import CategoryRule, ImageMetadata, UploadBatch

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "decoration_app.db")


class DatabaseManager:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self):
        with self.get_connection() as conn:
            # 1. Categories table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS categories (
                    name TEXT PRIMARY KEY,
                    description TEXT,
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # 2. Upload Batches table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS upload_batches (
                    batch_id TEXT PRIMARY KEY,
                    source_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    total_images INTEGER DEFAULT 0,
                    approved_count INTEGER DEFAULT 0,
                    rejected_count INTEGER DEFAULT 0,
                    review_count INTEGER DEFAULT 0,
                    duplicate_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            """)

            # 3. Images table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sha256 TEXT NOT NULL,
                    phash TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    file_format TEXT NOT NULL,
                    category TEXT NOT NULL,
                    upload_batch_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    rejection_reasons TEXT,
                    blur_score REAL DEFAULT 0.0,
                    detected_angle TEXT DEFAULT 'UNKNOWN',
                    angle_confidence REAL DEFAULT 0.0,
                    human_detected INTEGER DEFAULT 0,
                    human_count INTEGER DEFAULT 0,
                    watermark_detected INTEGER DEFAULT 0,
                    watermark_confidence REAL DEFAULT 0.0,
                    detected_text TEXT DEFAULT '',
                    detected_numbers TEXT,
                    duplicate_of_id INTEGER,
                    duplicate_similarity REAL DEFAULT 0.0,
                    source_path TEXT NOT NULL,
                    output_path TEXT,
                    first_seen_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (upload_batch_id) REFERENCES upload_batches(batch_id)
                )
            """)

            # Indexes for fast lookup
            conn.execute("CREATE INDEX IF NOT EXISTS idx_images_sha256 ON images(sha256);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_images_phash ON images(phash);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_images_category ON images(category);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_images_batch ON images(upload_batch_id);")

            # 4. Manual Reviews Audit table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS manual_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id INTEGER NOT NULL,
                    previous_status TEXT NOT NULL,
                    new_status TEXT NOT NULL,
                    reason TEXT,
                    action_timestamp TEXT NOT NULL,
                    FOREIGN KEY (image_id) REFERENCES images(id)
                )
            """)

        self._seed_default_categories()

    def _seed_default_categories(self):
        defaults = [
            CategoryRule(
                name="First Birthday Decoration",
                description="1st Birthday decorations with balloons, cake table, and banners.",
                required_text=["Happy Birthday"],
                required_numbers=["1", "ONE", "First"],
                human_detection_enabled=True,
                human_action="REJECT",
                blur_detection_enabled=True,
                blur_threshold=100.0,
                watermark_detection_enabled=True,
                angle_filter_enabled=True,
                preferred_angles=["FRONT"],
            ),
            CategoryRule(
                name="Second Birthday Decoration",
                description="2nd Birthday celebration decorations.",
                required_text=["Happy Birthday"],
                required_numbers=["2", "TWO", "Second"],
                human_detection_enabled=True,
                human_action="REJECT",
                blur_detection_enabled=True,
                blur_threshold=100.0,
                watermark_detection_enabled=True,
                angle_filter_enabled=True,
                preferred_angles=["FRONT"],
            ),
            CategoryRule(
                name="Wedding Decoration",
                description="Stage, mandap, and entrance floral and lighting setups.",
                required_text=[],
                required_numbers=[],
                human_detection_enabled=True,
                human_action="REJECT",
                blur_detection_enabled=True,
                blur_threshold=100.0,
                watermark_detection_enabled=True,
                angle_filter_enabled=True,
                preferred_angles=["FRONT"],
            ),
            CategoryRule(
                name="Anniversary Decoration",
                description="Silver, Golden, and annual wedding anniversary setups.",
                required_text=["Anniversary"],
                required_numbers=[],
                human_detection_enabled=True,
                human_action="REJECT",
                blur_detection_enabled=True,
                blur_threshold=100.0,
                watermark_detection_enabled=True,
                angle_filter_enabled=True,
                preferred_angles=["FRONT"],
            ),
            CategoryRule(
                name="Baby Shower Decoration",
                description="Cradle, balloon arch, and baby welcome decorations.",
                required_text=["Baby Shower", "Baby"],
                required_numbers=[],
                human_detection_enabled=True,
                human_action="REJECT",
                blur_detection_enabled=True,
                blur_threshold=100.0,
                watermark_detection_enabled=True,
                angle_filter_enabled=True,
                preferred_angles=["FRONT"],
            ),
            CategoryRule(
                name="Haldi Decoration",
                description="Traditional yellow floral and marigold themes for Haldi ceremonies.",
                required_text=["Haldi"],
                required_numbers=[],
                human_detection_enabled=True,
                human_action="REJECT",
                blur_detection_enabled=True,
                blur_threshold=100.0,
                watermark_detection_enabled=True,
                angle_filter_enabled=True,
                preferred_angles=["FRONT"],
            ),
            CategoryRule(
                name="Engagement Decoration",
                description="Ring ceremony backdrops and romantic stage arrangements.",
                required_text=["Engagement", "Engaged"],
                required_numbers=[],
                human_detection_enabled=True,
                human_action="REJECT",
                blur_detection_enabled=True,
                blur_threshold=100.0,
                watermark_detection_enabled=True,
                angle_filter_enabled=True,
                preferred_angles=["FRONT"],
            ),
        ]

        for cat in defaults:
            self.save_category(cat, overwrite_if_exists=False)

    # ---------------- Category Management ----------------

    def save_category(self, cat: CategoryRule, overwrite_if_exists: bool = True) -> bool:
        now = datetime.utcnow().isoformat()
        with self.get_connection() as conn:
            existing = conn.execute("SELECT name FROM categories WHERE name = ?", (cat.name,)).fetchone()
            if existing and not overwrite_if_exists:
                return False
            if existing:
                conn.execute("""
                    UPDATE categories
                    SET description = ?, config_json = ?, updated_at = ?
                    WHERE name = ?
                """, (cat.description, cat.to_json(), now, cat.name))
            else:
                conn.execute("""
                    INSERT INTO categories (name, description, config_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (cat.name, cat.description, cat.to_json(), cat.created_at or now, now))
        return True

    def get_category(self, name: str) -> Optional[CategoryRule]:
        with self.get_connection() as conn:
            row = conn.execute("SELECT config_json FROM categories WHERE name = ?", (name,)).fetchone()
            if row:
                return CategoryRule.from_json(row["config_json"])
        return None

    def list_categories(self) -> List[CategoryRule]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT config_json FROM categories ORDER BY name ASC").fetchall()
            return [CategoryRule.from_json(r["config_json"]) for r in rows]

    def delete_category(self, name: str) -> bool:
        with self.get_connection() as conn:
            res = conn.execute("DELETE FROM categories WHERE name = ?", (name,))
            return res.rowcount > 0

    # ---------------- Upload Batch Management ----------------

    def create_batch(self, batch: UploadBatch):
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO upload_batches (
                    batch_id, source_name, category, total_images,
                    approved_count, rejected_count, review_count, duplicate_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                batch.batch_id, batch.source_name, batch.category, batch.total_images,
                batch.approved_count, batch.rejected_count, batch.review_count,
                batch.duplicate_count, batch.created_at
            ))

    def update_batch_counts(self, batch_id: str):
        with self.get_connection() as conn:
            counts = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'APPROVED' THEN 1 ELSE 0 END) as approved,
                    SUM(CASE WHEN status = 'REJECTED' THEN 1 ELSE 0 END) as rejected,
                    SUM(CASE WHEN status = 'REVIEW_REQUIRED' THEN 1 ELSE 0 END) as review,
                    SUM(CASE WHEN status = 'DUPLICATE' THEN 1 ELSE 0 END) as duplicate
                FROM images
                WHERE upload_batch_id = ?
            """, (batch_id,)).fetchone()

            if counts:
                conn.execute("""
                    UPDATE upload_batches
                    SET total_images = ?, approved_count = ?, rejected_count = ?,
                        review_count = ?, duplicate_count = ?
                    WHERE batch_id = ?
                """, (
                    counts["total"] or 0,
                    counts["approved"] or 0,
                    counts["rejected"] or 0,
                    counts["review"] or 0,
                    counts["duplicate"] or 0,
                    batch_id
                ))

    def list_batches(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM upload_batches ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]

    # ---------------- Duplicate Detection Queries ----------------

    def find_exact_duplicate_by_sha256(self, sha256_hash: str, global_scope: bool = True, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            query = "SELECT * FROM images WHERE sha256 = ?"
            params = [sha256_hash]
            if not global_scope and category:
                query += " AND category = ?"
                params.append(category)
            query += " ORDER BY id ASC LIMIT 1"
            row = conn.execute(query, params).fetchone()
            return dict(row) if row else None

    def find_perceptual_duplicates(self, target_phash: str, max_hamming_distance: int = 6, global_scope: bool = True, category: Optional[str] = None) -> List[Tuple[Dict[str, Any], int]]:
        """
        Calculates Hamming distance against indexed perceptual hashes in SQLite.
        Returns matching records and their distance.
        """
        try:
            target_hash = imagehash.hex_to_hash(target_phash)
        except Exception:
            return []

        matches = []
        with self.get_connection() as conn:
            query = "SELECT * FROM images WHERE phash IS NOT NULL AND phash != ''"
            params = []
            if not global_scope and category:
                query += " AND category = ?"
                params.append(category)

            cursor = conn.execute(query, params)
            for row in cursor:
                row_phash_str = row["phash"]
                try:
                    db_hash = imagehash.hex_to_hash(row_phash_str)
                    dist = target_hash - db_hash
                    if dist <= max_hamming_distance:
                        matches.append((dict(row), dist))
                except Exception:
                    continue

        # Sort by best match (smallest distance)
        matches.sort(key=lambda x: x[1])
        return matches

    # ---------------- Image Record Operations ----------------

    def insert_image(self, record: ImageMetadata) -> int:
        now = datetime.utcnow().isoformat()
        with self.get_connection() as conn:
            # Ensure upload_batches entry exists to satisfy foreign key constraint
            if record.upload_batch_id:
                conn.execute("""
                    INSERT OR IGNORE INTO upload_batches (batch_id, source_name, category, created_at)
                    VALUES (?, ?, ?, ?)
                """, (record.upload_batch_id, "Direct Ingestion", record.category, now))

            cursor = conn.execute("""
                INSERT INTO images (
                    sha256, phash, original_filename, file_size_bytes, width, height,
                    file_format, category, upload_batch_id, status, rejection_reasons,
                    blur_score, detected_angle, angle_confidence, human_detected, human_count,
                    watermark_detected, watermark_confidence, detected_text, detected_numbers,
                    duplicate_of_id, duplicate_similarity, source_path, output_path,
                    first_seen_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.sha256,
                record.phash,
                record.original_filename,
                record.file_size_bytes,
                record.width,
                record.height,
                record.file_format,
                record.category,
                record.upload_batch_id,
                record.status,
                json.dumps(record.rejection_reasons),
                record.blur_score,
                record.detected_angle,
                record.angle_confidence,
                1 if record.human_detected else 0,
                record.human_count,
                1 if record.watermark_detected else 0,
                record.watermark_confidence,
                record.detected_text,
                json.dumps(record.detected_numbers),
                record.duplicate_of_id,
                record.duplicate_similarity,
                record.source_path,
                record.output_path,
                record.first_seen_at or now,
                record.updated_at or now
            ))
            record.id = cursor.lastrowid
            return record.id

    def update_image_status(self, image_id: int, new_status: str, new_output_path: Optional[str] = None, reason: str = "") -> bool:
        now = datetime.utcnow().isoformat()
        with self.get_connection() as conn:
            curr = conn.execute("SELECT status, upload_batch_id FROM images WHERE id = ?", (image_id,)).fetchone()
            if not curr:
                return False
            prev_status = curr["status"]
            batch_id = curr["upload_batch_id"]

            if new_output_path is not None:
                conn.execute("""
                    UPDATE images
                    SET status = ?, output_path = ?, updated_at = ?
                    WHERE id = ?
                """, (new_status, new_output_path, now, image_id))
            else:
                conn.execute("""
                    UPDATE images
                    SET status = ?, updated_at = ?
                    WHERE id = ?
                """, (new_status, now, image_id))

            # Record manual audit action
            conn.execute("""
                INSERT INTO manual_reviews (image_id, previous_status, new_status, reason, action_timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (image_id, prev_status, new_status, reason, now))

        self.update_batch_counts(batch_id)
        return True

    def get_image_by_id(self, image_id: int) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
            if row:
                d = dict(row)
                d["rejection_reasons"] = json.loads(d["rejection_reasons"]) if d["rejection_reasons"] else []
                d["detected_numbers"] = json.loads(d["detected_numbers"]) if d["detected_numbers"] else []
                return d
        return None

    def query_images(self, 
                     category: Optional[str] = None, 
                     status: Optional[str] = None, 
                     batch_id: Optional[str] = None,
                     limit: int = 1000, 
                     offset: int = 0) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            clauses = []
            params = []
            if category and category != "All":
                clauses.append("category = ?")
                params.append(category)
            if status and status != "All":
                clauses.append("status = ?")
                params.append(status)
            if batch_id and batch_id != "All":
                clauses.append("upload_batch_id = ?")
                params.append(batch_id)

            where_str = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            query = f"SELECT * FROM images {where_str} ORDER BY id DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            rows = conn.execute(query, params).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["rejection_reasons"] = json.loads(d["rejection_reasons"]) if d["rejection_reasons"] else []
                d["detected_numbers"] = json.loads(d["detected_numbers"]) if d["detected_numbers"] else []
                result.append(d)
            return result

    def get_statistics(self, category: Optional[str] = None) -> Dict[str, Any]:
        with self.get_connection() as conn:
            where_sql = "WHERE category = ?" if (category and category != "All") else ""
            params = [category] if (category and category != "All") else []

            stats = conn.execute(f"""
                SELECT
                    COUNT(*) as total_images,
                    SUM(CASE WHEN status = 'APPROVED' THEN 1 ELSE 0 END) as approved,
                    SUM(CASE WHEN status = 'REJECTED' THEN 1 ELSE 0 END) as rejected,
                    SUM(CASE WHEN status = 'REVIEW_REQUIRED' THEN 1 ELSE 0 END) as review_required,
                    SUM(CASE WHEN status = 'DUPLICATE' THEN 1 ELSE 0 END) as duplicates
                FROM images
                {where_sql}
            """, params).fetchone()

            total = stats["total_images"] or 0
            approved = stats["approved"] or 0
            rejected = stats["rejected"] or 0
            review = stats["review_required"] or 0
            duplicates = stats["duplicates"] or 0
            processed = approved + rejected + review + duplicates
            success_rate = round((approved / processed * 100.0), 1) if processed > 0 else 0.0

            return {
                "total_images": total,
                "total_processed": processed,
                "approved": approved,
                "rejected": rejected,
                "review_required": review,
                "duplicates": duplicates,
                "success_rate": success_rate
            }
