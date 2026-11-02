"""
Database Models and Schemas for Decoration Photo Filtering & Management Application.
Defines typed schemas and data structures for categories, images, batches, and duplicate records.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import json


@dataclass
class CategoryRule:
    name: str
    description: str = ""
    required_text: List[str] = field(default_factory=list)
    required_numbers: List[str] = field(default_factory=list)
    human_detection_enabled: bool = True
    human_action: str = "REJECT"  # 'REJECT' or 'REVIEW'
    blur_detection_enabled: bool = True
    blur_threshold: float = 100.0  # Laplacian variance threshold
    watermark_detection_enabled: bool = True
    angle_filter_enabled: bool = True
    preferred_angles: List[str] = field(default_factory=lambda: ["FRONT"])
    min_width: int = 400
    min_height: int = 400
    min_file_size_kb: float = 10.0
    max_file_size_kb: float = 50000.0
    custom_keywords: List[str] = field(default_factory=list)
    custom_ai_prompt: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data_str: str) -> "CategoryRule":
        data = json.loads(data_str)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CategoryRule":
        return cls(**data)


@dataclass
class ImageMetadata:
    original_filename: str
    file_size_bytes: int
    width: int
    height: int
    file_format: str
    sha256: str
    phash: str
    source_path: str
    category: str
    upload_batch_id: str
    id: Optional[int] = None
    status: str = "PENDING"  # APPROVED, REVIEW_REQUIRED, REJECTED, DUPLICATE
    rejection_reasons: List[str] = field(default_factory=list)
    blur_score: float = 0.0
    detected_angle: str = "UNKNOWN"
    angle_confidence: float = 0.0
    human_detected: bool = False
    human_count: int = 0
    watermark_detected: bool = False
    watermark_confidence: float = 0.0
    detected_text: str = ""
    detected_numbers: List[str] = field(default_factory=list)
    duplicate_of_id: Optional[int] = None
    duplicate_similarity: float = 0.0
    output_path: Optional[str] = None
    first_seen_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["rejection_reasons"] = json.dumps(self.rejection_reasons)
        d["detected_numbers"] = json.dumps(self.detected_numbers)
        return d


@dataclass
class UploadBatch:
    batch_id: str
    source_name: str
    category: str
    total_images: int = 0
    approved_count: int = 0
    rejected_count: int = 0
    review_count: int = 0
    duplicate_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
