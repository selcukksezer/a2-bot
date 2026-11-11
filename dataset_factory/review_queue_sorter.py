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
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union

QueueItem = Union[Dict[str, Any], Any]


_WEAK_STATUSES = frozenset({"weak", "suspicious", "review", "uncertain"})


def _first_conf_value(get: Callable[[str], Any]) -> Optional[float]:
    """First numeric confidence among ``conf``/``confidence``/``score``.

    ``get`` is a bound ``dict.get`` (or an equivalent field reader). Mirrors
    the old extractors' key order and ``isinstance(..., (int, float))``
    check exactly — a non-numeric first key falls through to the next key.
    """
    val = get("conf")
    if not isinstance(val, (int, float)):
        val = get("confidence")
        if not isinstance(val, (int, float)):
            val = get("score")
            if not isinstance(val, (int, float)):
                return None
    return float(val)


def _first_direct_value(get: Callable[[str], Any], keys: Tuple[str, ...]) -> Optional[float]:
    """First numeric direct field in ``keys`` order (old extractor precedence)."""
    for key in keys:
        val = get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _box_confidences(boxes: List[Any]) -> List[float]:
    """Per-box confidences (first numeric ``conf``/``confidence``/``score``).

    Dict boxes — the common case — read ``conf`` directly and only fall
    through to the other keys when it is missing/non-numeric, exactly like
    the old :meth:`ReviewQueueSorter.extract_confidence` loop (which paid
    one :func:`_get_field` call per key instead).
    """
    confs: List[float] = []
    append = confs.append
    for box in boxes:
        if type(box) is dict:
            c = box.get("conf")
            if not isinstance(c, (int, float)):
                c = _first_conf_value(box.get)
            if c is not None:
                append(float(c))
        else:
            c = _first_conf_value(lambda key: _get_field(box, key))
            if c is not None:
                append(c)
    return confs


class _FrameMetrics(NamedTuple):
    """Single-pass per-frame aggregates (see ``_frame_metrics``)."""

    weak_count: int
    min_conf: float
    avg_conf: float
    max_conf: float
    total_boxes: int


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
        # An empty list must not shadow a populated list under a later key:
        # `boxes=[]` with `detections=[...]` means the detections exist.
        if isinstance(val, list) and val:
            return val
        # S247 (v19/P3-8): yalnızca `list` kabul ediliyordu — tuple
        # veya tek-kutu dict sessizce düşüyordu.
        if isinstance(val, tuple) and val:
            return list(val)
        if isinstance(val, dict) and val:
            # Tek kutu dict'i (cx/class_id taşır) tek öğeli liste;
            # kutu eşlemesi (id -> kutu) değer listesi sayılır.
            if any(k in val for k in ("cx", "cy", "class_id", "x", "y", "bbox")):
                return [val]
            return list(val.values())
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
        """Tarih/zaman bilgisini sıralanabilir float epoch değerine çevirir.

        Tarihi çözülemeyen kare `float("inf")` döner: bilinmeyen tarih
        epoch 0 (1970) değildir ve sıralamada en sona düşer.
        """
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
        return float("inf")

    def _direct_confidences(self, item: QueueItem) -> Tuple[Optional[float], Optional[float]]:
        """Direct per-frame confidence fields for the ``min`` and ``avg`` modes.

        One pass over the direct keys (instead of one pass per mode): the
        two modes only differ in key priority, so the present numeric
        values are collected once and then resolved per mode order.
        ``max`` shares the non-``min`` order with ``avg``.

        Dict items (the overwhelmingly common case: JSON-loaded queue
        entries) are read with the bound ``dict.get`` — no per-key Python
        function call; object-style items fall back to :func:`_get_field`.
        """
        if isinstance(item, dict):
            get = item.get
            return (
                _first_direct_value(
                    get, ("min_conf", "confidence", "conf", "score", "avg_conf", "mean_conf", "max_conf")
                ),
                _first_direct_value(
                    get, ("confidence", "conf", "score", "avg_conf", "mean_conf", "min_conf", "max_conf")
                ),
            )
        get_field = lambda key: _get_field(item, key)  # noqa: E731
        return (
            _first_direct_value(
                get_field, ("min_conf", "confidence", "conf", "score", "avg_conf", "mean_conf", "max_conf")
            ),
            _first_direct_value(
                get_field, ("confidence", "conf", "score", "avg_conf", "mean_conf", "min_conf", "max_conf")
            ),
        )

    def _frame_metrics(
        self,
        item: QueueItem,
        need_weak: bool = True,
        need_min: bool = False,
        need_avg: bool = False,
        need_max: bool = False,
        need_total: bool = True,
    ) -> _FrameMetrics:
        """Compute the requested per-frame aggregates in ONE box scan.

        Previously ``sort_queue`` (``weak_count_desc``) called
        :meth:`extract_weak_count` and then :meth:`extract_confidence`, and
        :meth:`summarize_queue` re-scanned boxes per pending frame — each
        extractor walked the whole box list separately (2–3 full scans per
        frame). All requested values are derived here from a single pass
        over the boxes, with the same direct-field precedence each
        extractor had: a direct ``weak_count``/``total_boxes``/confidence
        field wins over the box-derived value, exactly as before.

        Only the aggregates the caller asked for are computed: a
        ``box_count`` sort never touches per-box confidence/weak fields
        (it only needs ``len(boxes)``), and a confidence sort never
        evaluates weak flags. Unrequested fields are returned as ``0``/``0.0``
        and must not be read by the caller.

        Dict items/boxes (JSON-loaded queue entries) are read with bound
        ``dict.get`` instead of the per-key :func:`_get_field` call — the
        old extractors paid one Python call per field lookup, which
        dominated the cost at ~25 boxes/frame. Object-style items/boxes
        keep the exact old :func:`_get_field` path.
        """
        # Bound reader: dict.get for dict items, _get_field otherwise.
        if isinstance(item, dict):
            get = item.get
        else:
            get = lambda key: _get_field(item, key)  # noqa: E731

        weak_count: Optional[int] = None
        if need_weak:
            weak_count = _first_direct_value(
                get,
                ("weak_count", "suspicious_count", "low_conf_count", "weak_box_count", "uncertain_count"),
            )
            if weak_count is not None:
                weak_count = int(weak_count)

        total_boxes: Optional[int] = None
        if need_total:
            total_boxes = _first_direct_value(
                get, ("total_boxes", "box_count", "num_boxes", "detection_count", "count")
            )
            if total_boxes is not None:
                total_boxes = int(total_boxes)

        min_direct: Optional[float] = None
        avg_direct: Optional[float] = None
        if need_min or need_avg or need_max:
            min_direct, avg_direct = self._direct_confidences(item)

        need_box_weak = need_weak and weak_count is None
        need_box_total = need_total and total_boxes is None
        need_box_conf = (need_min and min_direct is None) or (
            (need_avg or need_max) and avg_direct is None
        )
        if not (need_box_weak or need_box_total or need_box_conf):
            return _FrameMetrics(
                weak_count=weak_count if weak_count is not None else 0,
                min_conf=min_direct if min_direct is not None else 0.0,
                avg_conf=avg_direct if avg_direct is not None else 0.0,
                max_conf=avg_direct if avg_direct is not None else 0.0,
                total_boxes=total_boxes if total_boxes is not None else 0,
            )

        boxes = _extract_boxes(item) if (need_box_weak or need_box_total or need_box_conf) else []
        box_weak = 0
        conf_sum = 0.0
        conf_n = 0
        box_min: Optional[float] = None
        box_max: Optional[float] = None
        weak_threshold = self.weak_conf_threshold
        # When only the box COUNT is needed (box_count sorts), there is no
        # per-box work at all: len(boxes) above is the answer, and walking
        # the list would be pure overhead.
        if need_box_weak or need_box_conf:
            if need_box_conf and not need_box_weak:
                # Confidence-only request (extract_confidence / conf sorts):
                # collect once and let C-level min/max/sum do the reduce —
                # exactly the old extract_confidence shape, so the float
                # result is bit-identical too. A running Python accumulation
                # here measured ~2x slower than this (and 1 ulp off on avg).
                confs = _box_confidences(boxes)
                if confs:
                    if need_min:
                        box_min = min(confs)
                    if need_max:
                        box_max = max(confs)
                    if need_avg:
                        conf_sum = sum(confs)
                        conf_n = len(confs)
                # Weak counts are not requested on this path.
            else:
                for box in boxes:
                    if type(box) is dict:
                        bget = box.get
                        conf_val: Optional[float] = None
                        conf_known = False
                        if need_box_weak:
                            if bget("is_weak") is True or (bget("suspicious") or bget("weak")) is True:
                                box_weak += 1
                            else:
                                status = bget("status") or bget("tier")
                                if isinstance(status, str) and status.strip().lower() in _WEAK_STATUSES:
                                    box_weak += 1
                                else:
                                    conf_val = _first_conf_value(bget)
                                    conf_known = True
                                    if conf_val is not None and conf_val < weak_threshold:
                                        box_weak += 1
                        if need_box_conf:
                            if not conf_known:
                                conf_val = _first_conf_value(bget)
                            if conf_val is not None:
                                conf_sum += conf_val
                                conf_n += 1
                                if box_min is None or conf_val < box_min:
                                    box_min = conf_val
                                if box_max is None or conf_val > box_max:
                                    box_max = conf_val
                    else:
                        # Object-style box: exact old _get_field logic.
                        if need_box_weak:
                            is_weak = _get_field(box, "is_weak")
                            is_suspicious = _get_field(box, "suspicious") or _get_field(box, "weak")
                            status = _get_field(box, "status") or _get_field(box, "tier")
                            if is_weak is True or is_suspicious is True:
                                box_weak += 1
                            elif isinstance(status, str) and status.strip().lower() in _WEAK_STATUSES:
                                box_weak += 1
                            else:
                                bval = _first_conf_value(lambda key: _get_field(box, key))
                                if bval is not None and bval < weak_threshold:
                                    box_weak += 1
                        if need_box_conf:
                            c = _first_conf_value(lambda key: _get_field(box, key))
                            if c is not None:
                                conf_sum += c
                                conf_n += 1
                                if box_min is None or c < box_min:
                                    box_min = c
                                if box_max is None or c > box_max:
                                    box_max = c

        return _FrameMetrics(
            weak_count=weak_count if weak_count is not None else box_weak,
            min_conf=min_direct if min_direct is not None else (box_min if box_min is not None else 0.0),
            avg_conf=avg_direct if avg_direct is not None else (conf_sum / conf_n if conf_n else 0.0),
            max_conf=avg_direct if avg_direct is not None else (box_max if box_max is not None else 0.0),
            total_boxes=total_boxes if total_boxes is not None else len(boxes),
        )

    def extract_confidence(self, item: QueueItem, mode: str = "avg") -> float:
        """Kareye ait güven skorunu çıkarır.

        Önce kare üzerindeki doğrudan güven alanlarını kontrol eder, yoksa kutuların
        güven skorlarından hesaplar.

        Lean path (does NOT route through :meth:`_frame_metrics`): for a
        single aggregate the aggregate engine's call overhead + running
        accumulation measured ~1.3x slower than the old tight loop, so this
        keeps the old shape — one box-conf list + C-level ``min``/``max``/
        ``sum`` — with the dict fast path from :func:`_box_confidences`.
        Results are bit-identical to the old implementation.
        """
        min_direct, avg_direct = self._direct_confidences(item)
        if mode == "min":
            if min_direct is not None:
                return min_direct
        elif avg_direct is not None:
            return avg_direct
        confs = _box_confidences(_extract_boxes(item))
        if not confs:
            return 0.0
        if mode == "min":
            return min(confs)
        if mode == "max":
            return max(confs)
        return sum(confs) / len(confs)

    def extract_weak_count(self, item: QueueItem) -> int:
        """Şüpheli / düşük güvenli (`weak`) kutu sayısını döndürür."""
        return self._frame_metrics(item, need_total=False).weak_count

    def extract_total_boxes(self, item: QueueItem) -> int:
        """Karedeki toplam kutu sayısını döndürür.

        Lean path: a direct field or ``len(boxes)`` needs no aggregate
        machinery — routing this through :meth:`_frame_metrics` measured
        ~1.4x slower than the old direct implementation.
        """
        if isinstance(item, dict):
            total = _first_direct_value(
                item.get, ("total_boxes", "box_count", "num_boxes", "detection_count", "count")
            )
        else:
            total = _first_direct_value(
                lambda key: _get_field(item, key),
                ("total_boxes", "box_count", "num_boxes", "detection_count", "count"),
            )
        if total is not None:
            return int(total)
        return len(_extract_boxes(item))

    def _queue_metrics(
        self, item: QueueItem, want: Iterable[str] = ("weak", "min", "avg", "total")
    ) -> Dict[str, Any]:
        """Kareye ait sıralama metriklerini tek kutu taramasıyla hesaplar.

        `extract_weak_count` / `extract_confidence` / `extract_total_boxes`
        ile birebir aynı değerleri döndürür: doğrudan alan kısayolları aynı
        sırayla kontrol edilir, kutu taraması gerekiyorsa zayıf sayısı ile
        min/avg güven tek döngüde toplanır. `sort_queue` eskiden zayıf
        modlarda kutuları iki kez tarıyordu (bir kez zayıf sayı, bir kez
        güven için); bu yardımcı iki taramayı bire indirir.
        """
        want_set = set(want)
        out: Dict[str, Any] = {}
        if "weak" in want_set:
            for key in (
                "weak_count",
                "suspicious_count",
                "low_conf_count",
                "weak_box_count",
                "uncertain_count",
            ):
                val = _get_field(item, key)
                if isinstance(val, (int, float)):
                    out["weak"] = int(val)
                    break
        if "min" in want_set:
            for key in (
                "min_conf",
                "confidence",
                "conf",
                "score",
                "avg_conf",
                "mean_conf",
                "max_conf",
            ):
                val = _get_field(item, key)
                if isinstance(val, (int, float)):
                    out["min"] = float(val)
                    break
        if "avg" in want_set:
            for key in (
                "confidence",
                "conf",
                "score",
                "avg_conf",
                "mean_conf",
                "min_conf",
                "max_conf",
            ):
                val = _get_field(item, key)
                if isinstance(val, (int, float)):
                    out["avg"] = float(val)
                    break
        if "total" in want_set:
            for key in ("total_boxes", "box_count", "num_boxes", "detection_count", "count"):
                val = _get_field(item, key)
                if isinstance(val, (int, float)):
                    out["total"] = int(val)
                    break

        do_weak = "weak" in want_set and "weak" not in out
        do_conf = ("min" in want_set and "min" not in out) or (
            "avg" in want_set and "avg" not in out
        )
        need_boxes = do_weak or do_conf or ("total" in want_set and "total" not in out)
        if need_boxes:
            boxes = _extract_boxes(item)
            confs: List[float] = []
            weak_count = 0
            for box in boxes:
                bconf: Optional[float] = None
                if do_weak or do_conf:
                    for bkey in ("conf", "confidence", "score"):
                        bval = _get_field(box, bkey)
                        if isinstance(bval, (int, float)):
                            bconf = float(bval)
                            break
                if do_conf and bconf is not None:
                    confs.append(bconf)
                if do_weak:
                    is_weak = _get_field(box, "is_weak")
                    is_suspicious = _get_field(box, "suspicious") or _get_field(box, "weak")
                    status = _get_field(box, "status") or _get_field(box, "tier")
                    if is_weak is True or is_suspicious is True:
                        weak_count += 1
                        continue
                    if isinstance(status, str) and status.strip().lower() in (
                        "weak",
                        "suspicious",
                        "review",
                        "uncertain",
                    ):
                        weak_count += 1
                        continue
                    if bconf is not None and bconf < self.weak_conf_threshold:
                        weak_count += 1
            if do_weak:
                out["weak"] = weak_count
            if "min" in want_set and "min" not in out:
                out["min"] = min(confs) if confs else 0.0
            if "avg" in want_set and "avg" not in out:
                out["avg"] = (sum(confs) / len(confs)) if confs else 0.0
            if "total" in want_set and "total" not in out:
                out["total"] = len(boxes)
        return out

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

        def _date_key(item: QueueItem, descending: bool) -> float:
            # Unknown dates (inf) sort last in BOTH directions.
            ts = self.extract_timestamp(item)
            if ts == float("inf"):
                return float("inf")
            return -ts if descending else ts

        # Weak-count modes need TWO aggregates per frame (weak count +
        # a confidence tie-breaker), so they decorate once with a single
        # box scan per frame (``_frame_metrics``). Conf/box modes need only
        # ONE cheap aggregate each, and decorating measured slower than the
        # old key-lambda shape there (tuple + append per item beats nothing
        # when the metric itself is just ``len(boxes)`` or one conf list),
        # so they keep the old ``sorted(..., key=...)`` shape with the lean
        # extractors. Date modes need no box metrics at all.
        needs_metrics = mode in (
            SortOrder.WEAK_COUNT_DESC.value,
            "suspicious_desc",
            "weak_desc",
            SortOrder.WEAK_COUNT_ASC.value,
            "suspicious_asc",
            "weak_asc",
        )
        if mode in (
            SortOrder.WEAK_COUNT_DESC.value,
            "suspicious_desc",
            "weak_desc",
        ):
            metric_kwargs: Dict[str, bool] = {"need_min": True}
        elif mode in (
            SortOrder.WEAK_COUNT_ASC.value,
            "suspicious_asc",
            "weak_asc",
        ):
            metric_kwargs = {"need_avg": True}
        else:
            metric_kwargs = {}
        decorated = [
            (
                item,
                self._frame_metrics(item, **metric_kwargs) if needs_metrics else None,
                self._get_tie_breaker(item),
            )
            for item in queue
        ]

        if mode in (SortOrder.DATE_DESC.value, SortOrder.DATE.value, "newest"):
            return [
                item
                for item, _m, _tie in sorted(
                    decorated,
                    key=lambda d: (_date_key(d[0], True), d[2]),
                )
            ]

        if mode in (SortOrder.DATE_ASC.value, "oldest"):
            return [
                item
                for item, _m, _tie in sorted(
                    decorated,
                    key=lambda d: (_date_key(d[0], False), d[2]),
                )
            ]

        if mode == SortOrder.CONF_DESC.value:
            # Asymmetric with conf_asc by design (test_w62 locks it):
            # CONF_DESC ranks by the frame's AVERAGE box confidence while
            # CONF_ASC ranks by the MINIMUM, so the two orders are NOT
            # exact reverses of each other. D51 (23a2b14) made desc rank
            # by minimum ("symmetric"); fix operation R20b restores the
            # average per the w62 contract.
            # Single-aggregate mode: direct key lambda with the lean
            # extractor (see the note at the top of sort_queue).
            return sorted(
                queue,
                key=lambda x: (-self.extract_confidence(x, mode="avg"), self._get_tie_breaker(x)),
            )

        if mode == SortOrder.CONF_ASC.value:
            return sorted(
                queue,
                key=lambda x: (self.extract_confidence(x, mode="min"), self._get_tie_breaker(x)),
            )

        if mode in (SortOrder.WEAK_COUNT_DESC.value, "suspicious_desc", "weak_desc"):
            return [
                item
                for item, _m, _tie in sorted(
                    decorated,
                    key=lambda d: (-d[1].weak_count, d[1].min_conf, d[2]),
                )
            ]

        if mode in (SortOrder.WEAK_COUNT_ASC.value, "suspicious_asc", "weak_asc"):
            return [
                item
                for item, _m, _tie in sorted(
                    decorated,
                    key=lambda d: (d[1].weak_count, -d[1].avg_conf, d[2]),
                )
            ]

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
        approved = 0
        pending_weak_boxes = 0
        for item in items:
            # Single pass: previously this counted approved in one generator
            # pass and then re-ran is_approved + extract_weak_count per
            # pending frame in a second pass.
            if self.is_approved(item):
                approved += 1
            else:
                pending_weak_boxes += self._frame_metrics(
                    item, need_total=False
                ).weak_count
        return {
            "total_frames": total,
            "pending_frames": total - approved,
            "approved_frames": approved,
            "pending_weak_boxes": pending_weak_boxes,
        }


__all__ = [
    "ReviewQueueSorter",
    "SortOrder",
]
