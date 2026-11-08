"""Adım 2 inceleme kuyruğu sıralama ve filtreleme modülü (ReviewQueueSorter).

Bu modül, insan inceleme kuyruğundaki (Adım 2) kareleri:
- Tarihe göre (`date_desc`, `date_asc`, `date`)
- Güven skoru yüksekten düşüğe (`conf_desc`)
- Güven skoru düşükten yükseğe (`conf_asc`)
- Şüpheli / zayıf kutu sayısına göre (`weak_count_desc`, `weak_count_asc`)
- Toplam kutu sayısına göre (`box_count_desc`, `box_count_asc`, `total_boxes_desc`)
sıralar ve onaylanmış (`human_approved=True` veya `done="ok"`) kareleri
bekleyen inceleme kuyruğundan filtreler.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

QueueItem = Union[Dict[str, Any], Any]


class SortOrder(str, Enum):
    """Desteklenen kuyruk sıralama modları."""

    DATE_DESC = "date_desc"
    DATE_ASC = "date_asc"
    DATE = "date"
    CONF_DESC = "conf_desc"
    CONF_ASC = "conf_asc"
    WEAK_COUNT_DESC = "weak_count_desc"
    WEAK_COUNT_ASC = "weak_count_asc"
    BOX_COUNT_DESC = "box_count_desc"
    BOX_COUNT_ASC = "box_count_asc"
    TOTAL_BOXES_DESC = "total_boxes_desc"
    TOTAL_BOXES_ASC = "total_boxes_asc"


def _get_field(item: QueueItem, key: str, default: Any = None) -> Any:
    """Sözlük veya nesne üzerinden alan değerini güvenli biçimde okur."""
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _extract_boxes(item: QueueItem) -> List[Any]:
    """Kare içindeki kutu/tespit listesini döndürür."""
    for key in ("boxes", "detections", "annotations", "predictions", "labels"):
        val = _get_field(item, key)
        if isinstance(val, list):
            return val
    return []


class ReviewQueueSorter:
    """Adım 2 inceleme kuyruğunu filtreleyen ve çoklu kritere göre sıralayan sınıf."""

    SUPPORTED_SORT_MODES: Tuple[str, ...] = (
        SortOrder.DATE_DESC.value,
        SortOrder.DATE_ASC.value,
        SortOrder.DATE.value,
        SortOrder.CONF_DESC.value,
        SortOrder.CONF_ASC.value,
        SortOrder.WEAK_COUNT_DESC.value,
        SortOrder.WEAK_COUNT_ASC.value,
        SortOrder.BOX_COUNT_DESC.value,
        SortOrder.BOX_COUNT_ASC.value,
        SortOrder.TOTAL_BOXES_DESC.value,
        SortOrder.TOTAL_BOXES_ASC.value,
    )

    def __init__(
        self,
        weak_conf_threshold: float = 0.60,
        default_sort_by: str = SortOrder.WEAK_COUNT_DESC.value,
    ) -> None:
        """
        Args:
            weak_conf_threshold: Kutu bazında şüpheli/zayıf kabul edilecek üst güven eşiği.
            default_sort_by: Varsayılan sıralama kriteri.
        """
        self.weak_conf_threshold = float(weak_conf_threshold)
        self.default_sort_by = default_sort_by

    @staticmethod
    def is_approved(item: QueueItem) -> bool:
        """Karenin insan tarafından onaylanıp onaylanmadığını kontrol eder.

        `human_approved is True` veya `done == "ok"` (büyük/küçük harf duyarsız)
        durumlarında True döner.
        """
        human_approved = _get_field(item, "human_approved", False)
        if human_approved is True or str(human_approved).strip().lower() == "true":
            return True

        done_val = _get_field(item, "done", None)
        if isinstance(done_val, str) and done_val.strip().lower() == "ok":
            return True
        if done_val is True:
            return True

        meta = _get_field(item, "metadata") or _get_field(item, "meta")
        if isinstance(meta, dict):
            if meta.get("human_approved") is True:
                return True
            done_meta = meta.get("done")
            if isinstance(done_meta, str) and done_meta.strip().lower() == "ok":
                return True

        return False

    def filter_pending(self, items: Iterable[QueueItem]) -> List[QueueItem]:
        """Onaylanmış (`human_approved=True` / `done="ok"`) kareleri bekleyen kuyruktan çıkarır."""
        return [item for item in items if not self.is_approved(item)]

    def filter_approved(self, items: Iterable[QueueItem]) -> List[QueueItem]:
        """Sadece onaylanmış (`human_approved=True` / `done="ok"`) kareleri döndürür."""
        return [item for item in items if self.is_approved(item)]

    def extract_timestamp(self, item: QueueItem) -> float:
        """Tarih/zaman bilgisini sıralanabilir float epoch değerine çevirir."""
        for key in (
            "timestamp",
            "created_at",
            "updated_at",
            "date",
            "mtime",
            "captured_at",
            "frame_time",
        ):
            val = _get_field(item, key)
            if val is None:
                continue
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, datetime):
                return val.timestamp()
            if isinstance(val, str):
                text = val.strip()
                if not text:
                    continue
                try:
                    return float(text)
                except ValueError:
                    pass
                try:
                    normalized = text.replace("Z", "+00:00")
                    return datetime.fromisoformat(normalized).timestamp()
                except ValueError:
                    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y%m%d_%H%M%S"):
                        try:
                            return datetime.strptime(text, fmt).timestamp()
                        except ValueError:
                            continue
        return 0.0

    def extract_confidence(self, item: QueueItem, mode: str = "avg") -> float:
        """Kareye ait güven skorunu çıkarır.

        Önce kare üzerindeki doğrudan güven alanlarını kontrol eder, yoksa kutuların
        güven skorlarından hesaplar.
        """
        direct_keys = (
            ("confidence", "conf", "score", "avg_conf", "mean_conf", "min_conf", "max_conf")
            if mode != "min"
            else ("min_conf", "confidence", "conf", "score", "avg_conf", "mean_conf", "max_conf")
        )
        for key in direct_keys:
            val = _get_field(item, key)
            if isinstance(val, (int, float)):
                return float(val)

        boxes = _extract_boxes(item)
        confs: List[float] = []
        for box in boxes:
            for bkey in ("conf", "confidence", "score"):
                bval = _get_field(box, bkey)
                if isinstance(bval, (int, float)):
                    confs.append(float(bval))
                    break

        if not confs:
            return 0.0
        if mode == "min":
            return min(confs)
        if mode == "max":
            return max(confs)
        return sum(confs) / len(confs)

    def extract_weak_count(self, item: QueueItem) -> int:
        """Şüpheli / düşük güvenli (`weak`) kutu sayısını döndürür."""
        for key in (
            "weak_count",
            "suspicious_count",
            "low_conf_count",
            "weak_box_count",
            "uncertain_count",
        ):
            val = _get_field(item, key)
            if isinstance(val, (int, float)):
                return int(val)

        boxes = _extract_boxes(item)
        weak_count = 0
        for box in boxes:
            is_weak = _get_field(box, "is_weak")
            is_suspicious = _get_field(box, "suspicious") or _get_field(box, "weak")
            status = _get_field(box, "status") or _get_field(box, "tier")
            if is_weak is True or is_suspicious is True:
                weak_count += 1
                continue
            if isinstance(status, str) and status.strip().lower() in ("weak", "suspicious", "review", "uncertain"):
                weak_count += 1
                continue

            for bkey in ("conf", "confidence", "score"):
                bval = _get_field(box, bkey)
                if isinstance(bval, (int, float)):
                    if float(bval) < self.weak_conf_threshold:
                        weak_count += 1
                    break

        return weak_count

    def extract_total_boxes(self, item: QueueItem) -> int:
        """Karedeki toplam kutu sayısını döndürür."""
        for key in ("total_boxes", "box_count", "num_boxes", "detection_count", "count"):
            val = _get_field(item, key)
            if isinstance(val, (int, float)):
                return int(val)
        return len(_extract_boxes(item))

    def _get_tie_breaker(self, item: QueueItem) -> str:
        """Sıralamada eşitlik durumunda deterministik sıra için kimlik döndürür."""
        for key in ("frame_id", "id", "filename", "image_path", "path", "name"):
            val = _get_field(item, key)
            if val is not None:
                return str(val)
        return ""

    def sort_queue(
        self,
        items: Iterable[QueueItem],
        sort_by: Optional[str] = None,
        only_pending: bool = True,
    ) -> List[QueueItem]:
        """İnceleme kuyruğunu seçilen kritere göre sıralar.

        Args:
            items: Sıralanacak kare koleksiyonu.
            sort_by: Sıralama modu (`date_desc`, `date_asc`, `conf_desc`, `conf_asc`,
                `weak_count_desc`, `weak_count_asc`, `box_count_desc`, `box_count_asc`).
            only_pending: True ise onaylanmış (`human_approved=True` / `done="ok"`)
                kareleri sıralama öncesinde filtreler.

        Returns:
            Sıralanmış (ve isteğe bağlı filtrelenmiş) kare listesi.
        """
        queue = self.filter_pending(items) if only_pending else list(items)
        mode = (sort_by or self.default_sort_by or SortOrder.WEAK_COUNT_DESC.value).strip().lower()

        if mode in (SortOrder.DATE_DESC.value, SortOrder.DATE.value, "newest"):
            return sorted(
                queue,
                key=lambda x: (-self.extract_timestamp(x), self._get_tie_breaker(x)),
            )

        if mode in (SortOrder.DATE_ASC.value, "oldest"):
            return sorted(
                queue,
                key=lambda x: (self.extract_timestamp(x), self._get_tie_breaker(x)),
            )

        if mode == SortOrder.CONF_DESC.value:
            return sorted(
                queue,
                key=lambda x: (-self.extract_confidence(x), self._get_tie_breaker(x)),
            )

        if mode == SortOrder.CONF_ASC.value:
            return sorted(
                queue,
                key=lambda x: (self.extract_confidence(x, mode="min"), self._get_tie_breaker(x)),
            )

        if mode in (SortOrder.WEAK_COUNT_DESC.value, "suspicious_desc", "weak_desc"):
            return sorted(
                queue,
                key=lambda x: (
                    -self.extract_weak_count(x),
                    self.extract_confidence(x, mode="min"),
                    self._get_tie_breaker(x),
                ),
            )

        if mode in (SortOrder.WEAK_COUNT_ASC.value, "suspicious_asc", "weak_asc"):
            return sorted(
                queue,
                key=lambda x: (
                    self.extract_weak_count(x),
                    -self.extract_confidence(x),
                    self._get_tie_breaker(x),
                ),
            )

        if mode in (
            SortOrder.BOX_COUNT_DESC.value,
            SortOrder.TOTAL_BOXES_DESC.value,
            "boxes_desc",
            "total_boxes",
            "box_count",
        ):
            return sorted(
                queue,
                key=lambda x: (-self.extract_total_boxes(x), self._get_tie_breaker(x)),
            )

        if mode in (
            SortOrder.BOX_COUNT_ASC.value,
            SortOrder.TOTAL_BOXES_ASC.value,
            "boxes_asc",
        ):
            return sorted(
                queue,
                key=lambda x: (self.extract_total_boxes(x), self._get_tie_breaker(x)),
            )

        raise ValueError(
            f"Desteklenmeyen sıralama modu: '{sort_by}'. "
            f"Desteklenen modlar: {', '.join(self.SUPPORTED_SORT_MODES)}"
        )

    def summarize_queue(self, items: Sequence[QueueItem]) -> Dict[str, int]:
        """Kuyruktaki toplam, bekleyen ve onaylanmış kare istatistiklerini döndürür."""
        total = len(items)
        approved = sum(1 for item in items if self.is_approved(item))
        pending = total - approved
        total_weak_boxes = sum(
            self.extract_weak_count(item) for item in items if not self.is_approved(item)
        )
        return {
            "total_frames": total,
            "pending_frames": pending,
            "approved_frames": approved,
            "pending_weak_boxes": total_weak_boxes,
        }


__all__ = [
    "ReviewQueueSorter",
    "SortOrder",
]
