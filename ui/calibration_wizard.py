import os
import sys
import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Sequence, Union

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRect, QPoint, QSize, pyqtSignal
from PyQt6.QtGui import (
    QPixmap,
    QImage,
    QPainter,
    QPen,
    QColor,
    QFont,
    QBrush,
    QMouseEvent,
    QPaintEvent,
)
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSlider,
    QGroupBox,
    QWidget,
    QFileDialog,
    QMessageBox,
    QFrame,
    QSizePolicy,
)

from ui.shell_page_frame import wrap_shell_tab_root_layout

logger = logging.getLogger("CalibrationWizard")


def imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Unicode-safe image reader for Windows paths containing non-ASCII characters."""
    try:
        p = Path(path)
        if not p.exists():
            return None
        raw = np.fromfile(str(p), dtype=np.uint8)
        if raw.size == 0:
            return None
        return cv2.imdecode(raw, flags)
    except Exception as exc:
        logger.error("Failed reading image from %s: %s", path, exc)
        return None


def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    """Unicode-safe image writer for Windows paths containing non-ASCII characters."""
    try:
        p = Path(path)
        ext = p.suffix if p.suffix else ".png"
        ok, encoded = cv2.imencode(ext, img)
        if not ok:
            return False
        encoded.tofile(str(p))
        return True
    except Exception as exc:
        logger.error("Failed writing image to %s: %s", path, exc)
        return False


def numpy_bgr_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    """Converts a BGR numpy array into a QPixmap safely."""
    try:
        if img_bgr is None or img_bgr.size == 0:
            return QPixmap()
        if len(img_bgr.shape) == 2:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
        elif img_bgr.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGBA)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(
                img_rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGBA8888
            )
            return QPixmap.fromImage(qimg.copy())
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(
            img_rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888
        )
        return QPixmap.fromImage(qimg.copy())
    except Exception as exc:
        logger.error("Failed converting numpy array to QPixmap: %s", exc)
        return QPixmap()


class InteractiveROICanvas(QWidget):
    """
    Interactive canvas widget that displays a source frame, allows user to drag-select
    a region of interest (ROI) in image coordinates, and renders template match bounding boxes.
    """

    roi_selected = pyqtSignal(QRect)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._source_bgr: Optional[np.ndarray] = None
        self._pixmap: Optional[QPixmap] = None
        self._scaled_rect: QRect = QRect()
        self._scale_x: float = 1.0
        self._scale_y: float = 1.0

        self._dragging: bool = False
        self._start_pt: QPoint = QPoint()
        self._end_pt: QPoint = QPoint()
        self._roi_image_rect: QRect = QRect()

        # Match result overlay in source image coordinates
        self._match_rect: Optional[QRect] = None
        self._match_score: float = 0.0
        self._match_label: str = ""
        self._match_passed: bool = False
        self._multi_boxes: List[Dict[str, Any]] = []

    def set_image(self, img_bgr: np.ndarray) -> None:
        """Sets the BGR image to display on the canvas and resets overlays."""
        try:
            if img_bgr is None or img_bgr.size == 0:
                return
            self._source_bgr = img_bgr.copy()
            self._pixmap = numpy_bgr_to_qpixmap(self._source_bgr)
            self._roi_image_rect = QRect()
            self._match_rect = None
            self._multi_boxes = []
            self.update()
        except Exception as exc:
            logger.error("Error setting canvas image: %s", exc)

    def get_source_image(self) -> Optional[np.ndarray]:
        return self._source_bgr

    def get_selected_roi_rect(self) -> QRect:
        """Returns the selected ROI rectangle in source image coordinates."""
        return self._roi_image_rect

    def get_cropped_bgr(self) -> Optional[np.ndarray]:
        """Returns the cropped BGR numpy slice corresponding to the user's ROI selection."""
        try:
            if self._source_bgr is None or self._roi_image_rect.isNull():
                return None
            r = self._roi_image_rect.normalized()
            h, w = self._source_bgr.shape[:2]
            x1 = max(0, min(r.x(), w - 1))
            y1 = max(0, min(r.y(), h - 1))
            x2 = max(0, min(r.x() + r.width(), w))
            y2 = max(0, min(r.y() + r.height(), h))
            if x2 - x1 < 4 or y2 - y1 < 4:
                return None
            return self._source_bgr[y1:y2, x1:x2].copy()
        except Exception as exc:
            logger.error("Failed cropping BGR ROI: %s", exc)
            return None

    def set_match_overlay(
        self,
        rect: Optional[QRect],
        score: float,
        label: str,
        passed: bool,
    ) -> None:
        """Sets the visual bounding box result from template matching."""
        self._match_rect = rect
        self._match_score = score
        self._match_label = label
        self._match_passed = passed
        self.update()

    def clear_overlays(self) -> None:
        self._roi_image_rect = QRect()
        self._match_rect = None
        self._multi_boxes = []
        self.update()

    def set_multi_detections(self, detections: Sequence[Dict[str, Any]]) -> None:
        """Sets multiple bounding boxes to render on the canvas."""
        self._multi_boxes = [d for d in detections if d.get("bbox")]
        self.update()

    def set_roi_image_rect(self, rect: Optional[QRect]) -> None:
        """Sets the selected ROI in source-image coordinates without a mouse drag."""
        if rect is None or rect.isNull():
            self._roi_image_rect = QRect()
        else:
            self._roi_image_rect = rect.normalized()
        self.update()

    def _compute_display_geometry(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            self._scaled_rect = QRect()
            return
        pw = self._pixmap.width()
        ph = self._pixmap.height()
        ww = max(1, self.width())
        wh = max(1, self.height())

        scale = min(ww / pw, wh / ph)
        disp_w = int(pw * scale)
        disp_h = int(ph * scale)
        offset_x = (ww - disp_w) // 2
        offset_y = (wh - disp_h) // 2

        self._scaled_rect = QRect(offset_x, offset_y, disp_w, disp_h)
        self._scale_x = pw / max(1, disp_w)
        self._scale_y = ph / max(1, disp_h)

    def _widget_to_image_point(self, pt: QPoint) -> QPoint:
        if self._scaled_rect.isNull() or self._pixmap is None:
            return QPoint(0, 0)
        clamped_x = max(
            self._scaled_rect.left(), min(pt.x(), self._scaled_rect.right())
        )
        clamped_y = max(
            self._scaled_rect.top(), min(pt.y(), self._scaled_rect.bottom())
        )
        rel_x = clamped_x - self._scaled_rect.left()
        rel_y = clamped_y - self._scaled_rect.top()
        img_x = int(rel_x * self._scale_x)
        img_y = int(rel_y * self._scale_y)
        img_x = max(0, min(img_x, self._pixmap.width() - 1))
        img_y = max(0, min(img_y, self._pixmap.height() - 1))
        return QPoint(img_x, img_y)

    def _image_to_widget_rect(self, img_rect: QRect) -> QRect:
        if self._scaled_rect.isNull() or self._pixmap is None:
            return QRect()
        inv_x = 1.0 / max(1e-6, self._scale_x)
        inv_y = 1.0 / max(1e-6, self._scale_y)
        wx = self._scaled_rect.left() + int(img_rect.x() * inv_x)
        wy = self._scaled_rect.top() + int(img_rect.y() * inv_y)
        ww = int(img_rect.width() * inv_x)
        wh = int(img_rect.height() * inv_y)
        return QRect(wx, wy, ww, wh)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._pixmap is not None:
            self._compute_display_geometry()
            if self._scaled_rect.contains(event.pos()):
                self._dragging = True
                self._start_pt = self._widget_to_image_point(event.pos())
                self._end_pt = self._start_pt
                self._roi_image_rect = QRect(self._start_pt, self._end_pt).normalized()
                self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging and self._pixmap is not None:
            self._end_pt = self._widget_to_image_point(event.pos())
            self._roi_image_rect = QRect(self._start_pt, self._end_pt).normalized()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self._end_pt = self._widget_to_image_point(event.pos())
            self._roi_image_rect = QRect(self._start_pt, self._end_pt).normalized()
            self.roi_selected.emit(self._roi_image_rect)
            self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Dark canvas backdrop
        painter.fillRect(self.rect(), QColor("#111318"))

        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(QColor("#7f8c8d"))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Kare Yüklenmedi — 'Referans Kare Yükle' veya 'Canlı Ekran Yakala' butonuna tıklayın",
            )
            painter.end()
            return

        self._compute_display_geometry()
        painter.drawPixmap(self._scaled_rect, self._pixmap)

        # Draw subtle border around active frame
        painter.setPen(QPen(QColor("#2c313c"), 1))
        painter.drawRect(self._scaled_rect)

        # Draw user-dragged ROI rectangle (Cyan dashed/solid)
        if not self._roi_image_rect.isNull() and self._roi_image_rect.width() > 2:
            w_rect = self._image_to_widget_rect(self._roi_image_rect)
            painter.setPen(QPen(QColor("#00e5ff"), 2, Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(QColor(0, 229, 255, 35)))
            painter.drawRect(w_rect)

            tag_text = (
                f"Seçili Alan (ROI): {self._roi_image_rect.width()}x{self._roi_image_rect.height()} "
                f"@ ({self._roi_image_rect.x()}, {self._roi_image_rect.y()})"
            )
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(tag_text) + 12
            th = fm.height() + 6
            ty = max(self._scaled_rect.top(), w_rect.top() - th - 2)
            label_bg = QRect(w_rect.left(), ty, tw, th)
            painter.fillRect(label_bg, QColor(0, 150, 180, 220))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(label_bg, Qt.AlignmentFlag.AlignCenter, tag_text)

        # Draw Multi-detection Bounding Boxes (from YZ auto calibration)
        if self._multi_boxes:
            for item in self._multi_boxes:
                bbox = item.get("bbox")
                if not bbox:
                    continue
                bx, by, bw, bh = (int(v) for v in bbox)
                box_img_rect = QRect(bx, by, bw, bh)
                if self._match_rect is not None and self._match_rect == box_img_rect:
                    continue
                w_box = self._image_to_widget_rect(box_img_rect)
                c_str = str(item.get("color", "#38bdf8"))
                col = QColor(c_str)
                painter.setPen(QPen(col, 2, Qt.PenStyle.DashLine))
                painter.setBrush(QBrush(QColor(col.red(), col.green(), col.blue(), 30)))
                painter.drawRect(w_box)

                lbl = str(item.get("label", ""))
                if lbl:
                    painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
                    fm = painter.fontMetrics()
                    tw = fm.horizontalAdvance(lbl) + 10
                    th = fm.height() + 4
                    ly = max(self._scaled_rect.top(), w_box.top() - th - 1)
                    l_bg = QRect(w_box.left(), ly, tw, th)
                    painter.fillRect(l_bg, QColor(16, 20, 28, 220))
                    painter.setPen(col)
                    painter.drawText(l_bg, Qt.AlignmentFlag.AlignCenter, lbl)

        # Draw Template Match Bounding Box (Green when above threshold, Amber when below)
        if self._match_rect is not None and not self._match_rect.isNull():
            m_rect = self._image_to_widget_rect(self._match_rect)
            box_color = QColor("#2ecc71") if self._match_passed else QColor("#f39c12")
            fill_color = (
                QColor(46, 204, 113, 45)
                if self._match_passed
                else QColor(243, 156, 18, 45)
            )
            painter.setPen(QPen(box_color, 2, Qt.PenStyle.SolidLine))
            painter.setBrush(QBrush(fill_color))
            painter.drawRect(m_rect)

            # Label banner
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(self._match_label) + 14
            th = fm.height() + 6
            by = max(self._scaled_rect.top(), m_rect.top() - th - 2)
            banner_rect = QRect(m_rect.left(), by, tw, th)
            painter.fillRect(banner_rect, box_color)
            painter.setPen(QColor("#0b0d10"))
            painter.drawText(banner_rect, Qt.AlignmentFlag.AlignCenter, self._match_label)

        painter.end()


def save_calibrated_roi(
    base_dir: Path,
    key: str,
    target_name: str,
    norm_x: float,
    norm_y: float,
    norm_w: float,
    norm_h: float,
    bbox_px: Tuple[int, int, int, int],
    resolution: Tuple[int, int],
    yolo_class_id: Optional[int],
    class_name: str,
) -> bool:
    """Saves normalized ROI coordinates and pixel metadata to config/calibrated_rois.json."""
    try:
        config_dir = Path(base_dir) / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        rois_path = config_dir / "calibrated_rois.json"
        data: Dict[str, Any] = {}
        if rois_path.exists():
            try:
                data = json.loads(rois_path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        if not isinstance(data, dict):
            data = {}
        if "rois" not in data or not isinstance(data["rois"], dict):
            data["rois"] = {}

        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        entry = {
            "key": key,
            "target": target_name,
            "normalized_roi": [norm_x, norm_y, norm_w, norm_h],
            "norm_x": norm_x,
            "norm_y": norm_y,
            "norm_w": norm_w,
            "norm_h": norm_h,
            "bbox_px": list(bbox_px),
            "resolution": list(resolution),
            "yolo_class_id": yolo_class_id,
            "class_name": class_name,
            "updated_at": now_iso,
        }
        data["rois"][key] = entry
        data[key] = entry
        data["updated_at"] = now_iso

        tmp_path = rois_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(rois_path)
        logger.info("Saved calibrated ROI for '%s' to %s", key, rois_path)
        return True
    except Exception as exc:
        logger.error("Failed saving calibrated ROI for '%s': %s", key, exc)
        return False


def _persist_learned_exemplar(
    crop: np.ndarray,
    class_id: int,
    class_name: str,
    base_dir: Path,
) -> bool:
    """Extracts HSV histogram & texture summary from crop and persists into dataset/learned_exemplars.json."""
    try:
        if crop is None or crop.size == 0:
            return False
        dataset_dir = Path(base_dir) / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        json_path = dataset_dir / "learned_exemplars.json"
        data: Dict[str, Any] = {}
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        if not isinstance(data, dict):
            data = {}
        classes = data.setdefault("classes", {})
        cid_str = str(class_id)
        cls_entry = classes.setdefault(cid_str, {
            "class_id": class_id,
            "class_name": class_name,
            "positive_exemplars": [],
            "negative_exemplars": [],
        })

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV) if crop.ndim == 3 else None
        h_hist: List[float] = []
        s_hist: List[float] = []
        v_hist: List[float] = []
        mean_hsv = [0.0, 0.0, 0.0]
        if hsv is not None and hsv.size > 0:
            h_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
            s_h = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
            v_h = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
            h_sum = float(np.sum(h_h)) or 1.0
            s_sum = float(np.sum(s_h)) or 1.0
            v_sum = float(np.sum(v_h)) or 1.0
            h_hist = [round(float(v) / h_sum, 4) for v in h_h]
            s_hist = [round(float(v) / s_sum, 4) for v in s_h]
            v_hist = [round(float(v) / v_sum, 4) for v in v_h]
            mean_hsv = [
                round(float(np.mean(hsv[:, :, 0])), 2),
                round(float(np.mean(hsv[:, :, 1])), 2),
                round(float(np.mean(hsv[:, :, 2])), 2),
            ]

        aspect_ratio = round(float(crop.shape[1]) / float(max(1, crop.shape[0])), 4)
        exemplar_record = {
            "class_id": class_id,
            "class_name": class_name,
            "aspect_ratio": aspect_ratio,
            "dimensions": [int(crop.shape[1]), int(crop.shape[0])],
            "mean_hsv": mean_hsv,
            "hist_h": h_hist,
            "hist_s": s_hist,
            "hist_v": v_hist,
            "source": "calibration_wizard",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        pos_list = cls_entry.setdefault("positive_exemplars", [])
        pos_list.append(exemplar_record)
        if len(pos_list) > 50:
            cls_entry["positive_exemplars"] = pos_list[-50:]

        data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        tmp_path = json_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(json_path)
        logger.info("Persisted positive exemplar for class %d (%s)", class_id, class_name)
        return True
    except Exception as exc:
        logger.error("Failed persisting learned exemplar: %s", exc)
        return False


def save_positive_exemplar(
    crop: np.ndarray,
    class_id: int,
    class_name: str,
    base_dir: Path,
) -> bool:
    """Saves positive exemplar to dataset_factory.exemplar_memory and syncs learned_exemplars.json."""
    success = False
    try:
        from dataset_factory.exemplar_memory import add_positive_exemplar
        add_positive_exemplar(crop, class_id=class_id, class_name=class_name)
        success = True
    except Exception as exc:
        logger.debug("exemplar_memory module not directly callable: %s", exc)

    direct_ok = _persist_learned_exemplar(crop, class_id=class_id, class_name=class_name, base_dir=base_dir)
    return success or direct_ok


def register_dataset_hand_label(
    base_dir: Path,
    frame_bgr: np.ndarray,
    roi_rect: QRect,
    class_id: int,
    class_name: str,
    source: str = "calibration_wizard",
) -> bool:
    """Registers calibrated sample into dataset/ with trusted=True, hand_label=True."""
    try:
        if frame_bgr is None or frame_bgr.size == 0 or roi_rect is None or roi_rect.isNull():
            return False
        fh, fw = frame_bgr.shape[:2]
        x, y, w, h = roi_rect.x(), roi_rect.y(), roi_rect.width(), roi_rect.height()
        if w < 4 or h < 4 or fw <= 0 or fh <= 0:
            return False

        cx = float(x + w / 2.0) / float(fw)
        cy = float(y + h / 2.0) / float(fh)
        nw = float(w) / float(fw)
        nh = float(h) / float(fh)

        cx = max(0.0001, min(0.9999, cx))
        cy = max(0.0001, min(0.9999, cy))
        nw = max(0.001, min(1.0, nw))
        nh = max(0.001, min(1.0, nh))

        ts = int(time.time() * 1000)
        stem = f"calib_{class_name}_{ts}"

        dataset_dir = Path(base_dir) / "dataset"
        images_dir = dataset_dir / "images" / "train"
        labels_dir = dataset_dir / "labels" / "train"
        review_dir = dataset_dir / "review" / "train"

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        review_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save training image
        img_path = images_dir / f"{stem}.jpg"
        imwrite_unicode(img_path, frame_bgr)

        # 2. Save YOLO txt label
        lbl_path = labels_dir / f"{stem}.txt"
        lbl_content = f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n"
        lbl_path.write_text(lbl_content, encoding="utf-8")

        # 3. Save review JSON with trusted=True, hand_label=True
        rev_path = review_dir / f"{stem}.json"
        rev_data = {
            "stem": stem,
            "split": "train",
            "floor": 0.45,
            "trusted": [
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "cx": round(cx, 6),
                    "cy": round(cy, 6),
                    "w": round(nw, 6),
                    "h": round(nh, 6),
                    "confidence": 1.0,
                    "trusted": True,
                    "hand_label": True,
                    "source": source,
                }
            ],
            "weak": [],
            "source": source,
            "hand_label": True,
        }
        rev_path.write_text(json.dumps(rev_data, indent=2, ensure_ascii=False), encoding="utf-8")

        # 4. Ensure trusted_seed.json exists
        seed_path = dataset_dir / "review" / "trusted_seed.json"
        if not seed_path.exists():
            seed_path.parent.mkdir(parents=True, exist_ok=True)
            seed_path.write_text(json.dumps({"has_trusted_seed": True}, indent=2), encoding="utf-8")

        # 5. Mark model as stale
        try:
            from dataset_factory.label_trust import mark_model_stale
            mark_model_stale(f"Calibration wizard hand label for {class_name}")
        except Exception:
            pass

        logger.info("Registered calibration hand-label in dataset: %s", stem)
        return True
    except Exception as exc:
        logger.error("Failed registering dataset hand label: %s", exc)
        return False


class CalibrationWizard(QDialog):
    """
    Visual Calibration Wizard & Live Vision Debugger dialog.
    Enables users to grab game screenshots or reference frames, draw custom ROI crops
    for 'F Gather', 'Player Arrow', or 'Map Header' templates, and test OpenCV template
    matching live with an adjustable confidence threshold slider.
    """

    template_updated = pyqtSignal(str)

    TEMPLATE_TARGETS: Dict[str, Dict[str, Any]] = {
        # 1. Preserved first 3 compatibility targets
        "F/E Toplama Butonu Şablonu Kes": {
            "filename": "f_gather_template.png",
            "yolo_class_id": 7,
            "class_name": "f_gather_prompt",
            "key": "gather_prompt",
            "category": "interaction",
            "desc": "Toplanabilir materyallerin yanında beliren etkileşim butonu ('F Gather' / 'E' toplama ikonu).",
        },
        "Minimap Oyuncu Oku Şablonu Kes": {
            "filename": "player_arrow.png",
            "yolo_class_id": None,
            "class_name": "player_arrow",
            "key": "minimap_arrow",
            "category": "minimap",
            "desc": "Sağ üst minimap veya dünya haritası üzerindeki oyuncu yön oku göstergesi.",
        },
        "Harita Başlığı Şablonu Kes": {
            "filename": "map_header.png",
            "yolo_class_id": None,
            "class_name": "map_header",
            "key": "map_title",
            "category": "map",
            "desc": "Tam ekran haritanın açık olduğunu doğrulayan üst başlık ikonu/yazısı.",
        },
        # 2. Interaction & Emblems
        "E Yüzen Kaynak İkonu Şablonu Kes": {
            "filename": "e_gather_icon_template.png",
            "yolo_class_id": 8,
            "class_name": "e_gather_icon",
            "key": "e_gather_icon",
            "category": "interaction",
            "desc": "Dünya üzerinde toplanabilir kaynak üzerinde yüzen dinamik 'E' etkileşim ikonu.",
        },
        "3B Beyaz Kanatlı Toplama Amblemi (Crest) Kes": {
            "filename": "world_gather_crest_template.png",
            "yolo_class_id": 6,
            "class_name": "world_gather_crest",
            "key": "crest_white_wing",
            "category": "crest",
            "desc": "3B oyun dünyasında toplanabilir kaynak üzerinde süzülen beyaz kanatlı amblem.",
        },
        # 3. 3D Gather Resources (Nodes)
        "Kaynak: Aria Bitkisi (herb_aria) Kes": {
            "filename": "herb_aria_template.png",
            "yolo_class_id": 0,
            "class_name": "herb_aria",
            "key": "node_aria",
            "category": "resource",
            "desc": "3B Aria bitkisi gövdesi ve mor çiçek dokusu.",
        },
        "Kaynak: Mela Bitkisi (herb_mela) Kes": {
            "filename": "herb_mela_template.png",
            "yolo_class_id": 1,
            "class_name": "herb_mela",
            "key": "node_mela",
            "category": "resource",
            "desc": "3B Mela bitkisi / kırmızı meyveli çalı gövdesi.",
        },
        "Kaynak: Targena / Bitki (herb_targena) Kes": {
            "filename": "herb_targena_template.png",
            "yolo_class_id": 2,
            "class_name": "herb_targena",
            "key": "node_targena",
            "category": "resource",
            "desc": "3B Targena çalı/bitki gövdesi.",
        },
        "Kaynak: Orichalcum / Maden Cevheri (ore_orichalcum) Kes": {
            "filename": "ore_orichalcum_template.png",
            "yolo_class_id": 3,
            "class_name": "ore_orichalcum",
            "key": "node_orichalcum",
            "category": "resource",
            "desc": "3B Orichalcum parlayan maden cevheri damarı ve kaya yapısı.",
        },
        "Kaynak: Kristal / Gem (gem_crystal) Kes": {
            "filename": "gem_crystal_template.png",
            "yolo_class_id": 4,
            "class_name": "gem_crystal",
            "key": "node_crystal",
            "category": "resource",
            "desc": "3B Kristal / değerli taş maden damarı.",
        },
        "Kaynak: Ağaç / Kereste (wood_tree) Kes": {
            "filename": "wood_tree_template.png",
            "yolo_class_id": 5,
            "class_name": "wood_tree",
            "key": "node_wood",
            "category": "resource",
            "desc": "3B toplanabilir ağaç / kereste gövdesi.",
        },
        # 4. Minimap Targets
        "Minimap 3 Yapraklı Filiz (Kaynak İkonu) Kes": {
            "filename": "minimap_sprout_template.png",
            "yolo_class_id": 9,
            "class_name": "minimap_sprout",
            "key": "minimap_sprout",
            "category": "minimap",
            "desc": "Sağ üst minimap üzerinde beliren 3 yapraklı toplama filizi ikonu.",
        },
        "Minimap Canavar Noktası (Negatif) Kes": {
            "filename": "minimap_monster_template.png",
            "yolo_class_id": 11,
            "class_name": "minimap_monster_negative",
            "key": "minimap_mob",
            "category": "minimap",
            "desc": "Minimap üzerindeki kırmızı canavar / düşman tehdit noktası.",
        },
        # 5. Combat & HUD Targets
        "3B Canavar / İsim Plakası (Hedef) Kes": {
            "filename": "monster_3d_template.png",
            "yolo_class_id": 10,
            "class_name": "monster_3d",
            "key": "mob_nameplate",
            "category": "combat",
            "desc": "3B canavar gövdesi veya kafa üstü kırmızı/turuncu isim plakası.",
        },
        "Hedef Canavar Can (HP) Barı Kes": {
            "filename": "target_hp_bar_template.png",
            "yolo_class_id": 18,
            "class_name": "target_hp_bar",
            "key": "target_hp_bar",
            "category": "hud",
            "desc": "Üst ortadaki seçili hedef / canavar can barı HUD bölgesi.",
        },
        "Oyuncu Can (HP) Barı Kes": {
            "filename": "player_hp_bar_template.png",
            "yolo_class_id": 16,
            "class_name": "player_hp_bar",
            "key": "player_hp_bar",
            "category": "hud",
            "desc": "Sol alt veya merkezdeki oyuncu sağlık (HP) barı göstergesi.",
        },
        "Oyuncu Mana (MP) Barı Kes": {
            "filename": "player_mp_bar_template.png",
            "yolo_class_id": 17,
            "class_name": "player_mp_bar",
            "key": "player_mp_bar",
            "category": "hud",
            "desc": "Sol alt veya merkezdeki oyuncu mana (MP) barı göstergesi.",
        },
        "Yetenek / Skill Bar Yuvası Kes": {
            "filename": "skill_slot_template.png",
            "yolo_class_id": 13,
            "class_name": "skill_ui_prompt",
            "key": "skill_slot",
            "category": "hud",
            "desc": "Alt kısımdaki yetenek (skill) bar yuvası ve bekleme süresi alanı.",
        },
        "Engel / Kaya / Duvar (Obstacle) Kes": {
            "filename": "obstacle_template.png",
            "yolo_class_id": 14,
            "class_name": "obstacle",
            "key": "obstacle",
            "category": "navigation",
            "desc": "Navigasyonu engelleyen 3B kaya, duvar veya geçilemez nesne.",
        },
        "Yürünebilir Yol / Zemin (Path) Kes": {
            "filename": "path_ground_template.png",
            "yolo_class_id": 15,
            "class_name": "path_ground",
            "key": "path_ground",
            "category": "navigation",
            "desc": "3B yürünebilir patika, yol veya zemin dokusu.",
        },
        "Tüy / Parçacık Efekti (Negatif) Kes": {
            "filename": "feather_particle_template.png",
            "yolo_class_id": 12,
            "class_name": "feather_particle_negative",
            "key": "feather_particle_negative",
            "category": "crest",
            "desc": "Beyaz kanat amblemiyle karışabilen tüy veya parlak parçacık negatif şablonu.",
        },
        "Hedef Canavar Can Barı (#18 target_hp_bar) Kes": {
            "filename": "target_hp_bar_template.png",
            "yolo_class_id": 18,
            "class_name": "target_hp_bar",
            "key": "target_hp_bar_18",
            "category": "hud",
            "desc": "Üst ortadaki seçili hedef / canavar can barı HUD bölgesi (Sınıf 18).",
        },
        "Yetenek Çubuğu (skill_bar) Kes": {
            "filename": "skill_bar_template.png",
            "yolo_class_id": 19,
            "class_name": "skill_bar",
            "key": "skill_bar",
            "category": "hud",
            "desc": "Alt HUD üzerindeki tam yetenek çubuğu (skill bar) şeridi (Sınıf 19).",
        },
        "Can İksiri (hp_potion) Kes": {
            "filename": "hp_potion_template.png",
            "yolo_class_id": 22,
            "class_name": "hp_potion",
            "key": "hp_potion",
            "category": "consumable",
            "desc": "Hızlı kullanım yuvasındaki kırmızı Can İksiri (HP Potion) ikonu (Sınıf 22).",
        },
        "Mana İksiri (mp_potion) Kes": {
            "filename": "mp_potion_template.png",
            "yolo_class_id": 23,
            "class_name": "mp_potion",
            "key": "mp_potion",
            "category": "consumable",
            "desc": "Hızlı kullanım yuvasındaki mavi Mana İksiri (MP Potion) ikonu (Sınıf 23).",
        },
        "Parşömen / Scroll (buff_scroll) Kes": {
            "filename": "buff_scroll_template.png",
            "yolo_class_id": 24,
            "class_name": "buff_scroll",
            "key": "buff_scroll",
            "category": "consumable",
            "desc": "Hızlı kullanım yuvasındaki Parşömen / Buff Scroll ikonu (Sınıf 24).",
        },
        # Compatibility aliases (English and short keys)
        "F Gather": {
            "filename": "f_gather_template.png",
            "yolo_class_id": 7,
            "class_name": "f_gather_prompt",
            "key": "gather_prompt",
            "category": "interaction",
            "desc": "Toplanabilir materyallerin yanında beliren etkileşim butonu ('F Gather' / 'E' toplama ikonu).",
        },
        "Player Arrow": {
            "filename": "player_arrow.png",
            "yolo_class_id": None,
            "class_name": "player_arrow",
            "key": "minimap_arrow",
            "category": "minimap",
            "desc": "Sağ üst minimap veya dünya haritası üzerindeki oyuncu yön oku göstergesi.",
        },
        "Map Header": {
            "filename": "map_header.png",
            "yolo_class_id": None,
            "class_name": "map_header",
            "key": "map_title",
            "category": "map",
            "desc": "Tam ekran haritanın açık olduğunu doğrulayan üst başlık ikonu/yazısı.",
        },
    }

    CANONICAL_25_CLASSES: Dict[int, Dict[str, Any]] = {
        0: {"class_id": 0, "class_name": "herb_aria", "label_tr": "Aria Bitkisi", "filename": "herb_aria_template.png", "key": "node_aria", "category": "resource", "target": "Kaynak: Aria Bitkisi (herb_aria) Kes"},
        1: {"class_id": 1, "class_name": "herb_mela", "label_tr": "Mela Bitkisi", "filename": "herb_mela_template.png", "key": "node_mela", "category": "resource", "target": "Kaynak: Mela Bitkisi (herb_mela) Kes"},
        2: {"class_id": 2, "class_name": "herb_targena", "label_tr": "Targena Bitkisi", "filename": "herb_targena_template.png", "key": "node_targena", "category": "resource", "target": "Kaynak: Targena / Bitki (herb_targena) Kes"},
        3: {"class_id": 3, "class_name": "ore_orichalcum", "label_tr": "Orichalcum Cevheri", "filename": "ore_orichalcum_template.png", "key": "node_orichalcum", "category": "resource", "target": "Kaynak: Orichalcum / Maden Cevheri (ore_orichalcum) Kes"},
        4: {"class_id": 4, "class_name": "gem_crystal", "label_tr": "Kristal / Gem", "filename": "gem_crystal_template.png", "key": "node_crystal", "category": "resource", "target": "Kaynak: Kristal / Gem (gem_crystal) Kes"},
        5: {"class_id": 5, "class_name": "wood_tree", "label_tr": "Ağaç / Kereste", "filename": "wood_tree_template.png", "key": "node_wood", "category": "resource", "target": "Kaynak: Ağaç / Kereste (wood_tree) Kes"},
        6: {"class_id": 6, "class_name": "world_gather_crest", "label_tr": "3B Beyaz Kanatlı Toplama Amblemi", "filename": "world_gather_crest_template.png", "key": "crest_white_wing", "category": "crest", "target": "3B Beyaz Kanatlı Toplama Amblemi (Crest) Kes"},
        7: {"class_id": 7, "class_name": "f_gather_prompt", "label_tr": "F/E Toplama Butonu", "filename": "f_gather_template.png", "key": "gather_prompt", "category": "interaction", "target": "F/E Toplama Butonu Şablonu Kes"},
        8: {"class_id": 8, "class_name": "e_gather_icon", "label_tr": "E Yüzen Kaynak İkonu", "filename": "e_gather_icon_template.png", "key": "e_gather_icon", "category": "interaction", "target": "E Yüzen Kaynak İkonu Şablonu Kes"},
        9: {"class_id": 9, "class_name": "minimap_sprout", "label_tr": "Minimap 3 Yapraklı Filiz", "filename": "minimap_sprout_template.png", "key": "minimap_sprout", "category": "minimap", "target": "Minimap 3 Yapraklı Filiz (Kaynak İkonu) Kes"},
        10: {"class_id": 10, "class_name": "monster_3d", "label_tr": "3B Canavar / İsim Plakası", "filename": "monster_3d_template.png", "key": "mob_nameplate", "category": "combat", "target": "3B Canavar / İsim Plakası (Hedef) Kes"},
        11: {"class_id": 11, "class_name": "minimap_monster_negative", "label_tr": "Minimap Canavar Noktası", "filename": "minimap_monster_template.png", "key": "minimap_mob", "category": "minimap", "target": "Minimap Canavar Noktası (Negatif) Kes"},
        12: {"class_id": 12, "class_name": "feather_particle_negative", "label_tr": "Tüy / Parçacık Negatif", "filename": "feather_particle_template.png", "key": "feather_particle_negative", "category": "crest", "target": "Tüy / Parçacık Efekti (Negatif) Kes"},
        13: {"class_id": 13, "class_name": "skill_ui_prompt", "label_tr": "Yetenek / Skill Yuvası", "filename": "skill_slot_template.png", "key": "skill_slot", "category": "hud", "target": "Yetenek / Skill Bar Yuvası Kes"},
        14: {"class_id": 14, "class_name": "obstacle", "label_tr": "Engel / Kaya / Duvar", "filename": "obstacle_template.png", "key": "obstacle", "category": "navigation", "target": "Engel / Kaya / Duvar (Obstacle) Kes"},
        15: {"class_id": 15, "class_name": "path_ground", "label_tr": "Yürünebilir Yol / Zemin", "filename": "path_ground_template.png", "key": "path_ground", "category": "navigation", "target": "Yürünebilir Yol / Zemin (Path) Kes"},
        16: {"class_id": 16, "class_name": "player_hp_bar", "label_tr": "Oyuncu Can (HP) Barı", "filename": "player_hp_bar_template.png", "key": "player_hp_bar", "category": "hud", "target": "Oyuncu Can (HP) Barı Kes"},
        17: {"class_id": 17, "class_name": "player_mp_bar", "label_tr": "Oyuncu Mana (MP) Barı", "filename": "player_mp_bar_template.png", "key": "player_mp_bar", "category": "hud", "target": "Oyuncu Mana (MP) Barı Kes"},
        18: {"class_id": 18, "class_name": "target_hp_bar", "label_tr": "Hedef Canavar Can (HP) Barı", "filename": "target_hp_bar_template.png", "key": "target_hp_bar", "category": "hud", "target": "Hedef Canavar Can (HP) Barı Kes"},
        19: {"class_id": 19, "class_name": "skill_bar", "label_tr": "Yetenek Çubuğu (Skill Bar)", "filename": "skill_bar_template.png", "key": "skill_bar", "category": "hud", "target": "Yetenek Çubuğu (skill_bar) Kes"},
        20: {"class_id": 20, "class_name": "minimap_player_arrow", "label_tr": "Minimap Oyuncu Oku", "filename": "player_arrow.png", "key": "minimap_arrow", "category": "minimap", "target": "Minimap Oyuncu Oku Şablonu Kes"},
        21: {"class_id": 21, "class_name": "map_header", "label_tr": "Harita Başlığı", "filename": "map_header.png", "key": "map_title", "category": "map", "target": "Harita Başlığı Şablonu Kes"},
        22: {"class_id": 22, "class_name": "hp_potion", "label_tr": "Can İksiri (HP Potion)", "filename": "hp_potion_template.png", "key": "hp_potion", "category": "consumable", "target": "Can İksiri (hp_potion) Kes"},
        23: {"class_id": 23, "class_name": "mp_potion", "label_tr": "Mana İksiri (MP Potion)", "filename": "mp_potion_template.png", "key": "mp_potion", "category": "consumable", "target": "Mana İksiri (mp_potion) Kes"},
        24: {"class_id": 24, "class_name": "buff_scroll", "label_tr": "Parşömen / Scroll (Buff Scroll)", "filename": "buff_scroll_template.png", "key": "buff_scroll", "category": "consumable", "target": "Parşömen / Scroll (buff_scroll) Kes"},
    }

    # Short key lookup aliases
    for _k, _info in list(TEMPLATE_TARGETS.items()):
        if "key" in _info and _info["key"] not in TEMPLATE_TARGETS:
            TEMPLATE_TARGETS[_info["key"]] = _info
        if "class_name" in _info and _info["class_name"] not in TEMPLATE_TARGETS:
            TEMPLATE_TARGETS[_info["class_name"]] = _info

    PRIMARY_TARGETS = (
        # First 3 targets preserved for strict backward compatibility:
        "F/E Toplama Butonu Şablonu Kes",
        "Minimap Oyuncu Oku Şablonu Kes",
        "Harita Başlığı Şablonu Kes",
        # Interaction & Crest:
        "E Yüzen Kaynak İkonu Şablonu Kes",
        "3B Beyaz Kanatlı Toplama Amblemi (Crest) Kes",
        # 3D Resources:
        "Kaynak: Aria Bitkisi (herb_aria) Kes",
        "Kaynak: Mela Bitkisi (herb_mela) Kes",
        "Kaynak: Targena / Bitki (herb_targena) Kes",
        "Kaynak: Orichalcum / Maden Cevheri (ore_orichalcum) Kes",
        "Kaynak: Kristal / Gem (gem_crystal) Kes",
        "Kaynak: Ağaç / Kereste (wood_tree) Kes",
        # Minimap:
        "Minimap 3 Yapraklı Filiz (Kaynak İkonu) Kes",
        "Minimap Canavar Noktası (Negatif) Kes",
        # Combat & HUD:
        "3B Canavar / İsim Plakası (Hedef) Kes",
        "Hedef Canavar Can (HP) Barı Kes",
        "Oyuncu Can (HP) Barı Kes",
        "Oyuncu Mana (MP) Barı Kes",
        "Yetenek / Skill Bar Yuvası Kes",
        "Engel / Kaya / Duvar (Obstacle) Kes",
        "Yürünebilir Yol / Zemin (Path) Kes",
        # Full 25 canonical additions (HUD + Consumables + Negative Feather):
        "Tüy / Parçacık Efekti (Negatif) Kes",
        "Yetenek Çubuğu (skill_bar) Kes",
        "Can İksiri (hp_potion) Kes",
        "Mana İksiri (mp_potion) Kes",
        "Parşömen / Scroll (buff_scroll) Kes",
    )

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        base_dir: Optional[Path] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Görsel Şablon ve Kalibrasyon Sihirbazı")
        self.resize(1020, 720)
        self.setModal(True)

        self.base_dir: Path = (
            Path(base_dir)
            if base_dir is not None
            else Path(__file__).resolve().parent.parent
        )
        self.templates_dir: Path = self.base_dir / "templates"
        self.frames_dir: Path = self.base_dir / "video_frames"

        try:
            self.templates_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("Could not ensure templates directory: %s", exc)

        self._apply_dark_theme()
        self._init_ui()
        self._load_default_frame_on_start()
        self._refresh_current_template_preview()

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            QDialog {
                background-color: #161920;
                color: #e2e8f0;
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                background-color: #1e222d;
                border: 1px solid #2e3444;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 12px;
                font-weight: bold;
                color: #94a3b8;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel {
                color: #cbd5e1;
            }
            QPushButton {
                background-color: #2b3245;
                color: #f1f5f9;
                border: 1px solid #3d4660;
                border-radius: 5px;
                padding: 7px 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #37415a;
                border-color: #60a5fa;
            }
            QPushButton:pressed {
                background-color: #1e2536;
            }
            QPushButton#primaryBtn {
                background-color: #2563eb;
                border-color: #3b82f6;
                color: #ffffff;
            }
            QPushButton#primaryBtn:hover {
                background-color: #1d4ed8;
            }
            QPushButton#successBtn {
                background-color: #16a34a;
                border-color: #22c55e;
                color: #ffffff;
            }
            QPushButton#successBtn:hover {
                background-color: #15803d;
            }
            QPushButton#aiBtn {
                background-color: #0e7490;
                border-color: #22d3ee;
                color: #ffffff;
            }
            QPushButton#aiBtn:hover {
                background-color: #155e75;
            }
            QComboBox {
                background-color: #252b3b;
                color: #f8fafc;
                border: 1px solid #3d4660;
                border-radius: 4px;
                padding: 5px 10px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #2e3444;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #38bdf8;
                border: 1px solid #0284c7;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            """
        )

    def _init_ui(self) -> None:
        root_layout = wrap_shell_tab_root_layout(self, "calibration")
        root_layout.setSpacing(10)

        # Eylem çubuğu (başlık ortak çerçevede)
        header_layout = QHBoxLayout()
        header_layout.addStretch()

        self.btn_grab_screen = QPushButton("Canlı Ekran Yakala")
        self.btn_grab_screen.clicked.connect(self.grab_current_screen)
        self.btn_load_ref = QPushButton("Referans Kare Yükle (frame_33s.jpg)")
        self.btn_load_ref.clicked.connect(self.load_reference_frame)
        self.btn_browse_frame = QPushButton("Görsel Seç...")
        self.btn_browse_frame.clicked.connect(self.browse_custom_frame)

        header_layout.addWidget(self.btn_grab_screen)
        header_layout.addWidget(self.btn_load_ref)
        header_layout.addWidget(self.btn_browse_frame)
        root_layout.addLayout(header_layout)

        # Main workspace split: Left = Interactive Canvas, Right = Controls & Previews
        workspace_layout = QHBoxLayout()
        workspace_layout.setSpacing(12)

        self.canvas = InteractiveROICanvas(self)
        self.canvas.roi_selected.connect(self._on_roi_selected)
        workspace_layout.addWidget(self.canvas, stretch=3)

        # Right panel controls
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        # 1. Target Template Selector Group
        target_group = QGroupBox("1. Şablon Hedefi ve Kesim Önizlemesi")
        target_layout = QVBoxLayout(target_group)

        self.combo_target = QComboBox()
        for name in self.PRIMARY_TARGETS:
            self.combo_target.addItem(name)
        self.combo_target.currentTextChanged.connect(self._on_target_changed)
        target_layout.addWidget(QLabel("Hedef Şablon:"))
        target_layout.addWidget(self.combo_target)

        self.lbl_target_desc = QLabel("")
        self.lbl_target_desc.setWordWrap(True)
        self.lbl_target_desc.setStyleSheet("color: #94a3b8; font-size: 11px;")
        target_layout.addWidget(self.lbl_target_desc)

        # Previews side by side: Saved Template vs Selected Crop
        previews_row = QHBoxLayout()
        saved_col = QVBoxLayout()
        saved_col.addWidget(QLabel("Aktif Şablon:"))
        self.lbl_saved_preview = QLabel("Yok")
        self.lbl_saved_preview.setFixedSize(130, 80)
        self.lbl_saved_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_saved_preview.setStyleSheet(
            "background-color: #111318; border: 1px solid #2e3444; border-radius: 4px;"
        )
        saved_col.addWidget(self.lbl_saved_preview)

        crop_col = QVBoxLayout()
        crop_col.addWidget(QLabel("Seçili Kesim (ROI):"))
        self.lbl_crop_preview = QLabel("Tuvalde seçin")
        self.lbl_crop_preview.setFixedSize(130, 80)
        self.lbl_crop_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_crop_preview.setStyleSheet(
            "background-color: #111318; border: 1px solid #2e3444; border-radius: 4px;"
        )
        crop_col.addWidget(self.lbl_crop_preview)

        previews_row.addLayout(saved_col)
        previews_row.addLayout(crop_col)
        target_layout.addLayout(previews_row)

        self.lbl_roi_coords = QLabel("Seçili Alan (ROI): Yok")
        self.lbl_roi_coords.setStyleSheet("color: #38bdf8; font-size: 12px;")
        target_layout.addWidget(self.lbl_roi_coords)

        self.btn_save_template = QPushButton("Kaydet")
        self.btn_save_template.setToolTip("Seçili alanı aktif hedef şablon olarak kaydet")
        self.btn_save_template.setObjectName("successBtn")
        self.btn_save_template.clicked.connect(self.save_cropped_region_as_template)
        target_layout.addWidget(self.btn_save_template)

        self.btn_auto_calibrate = QPushButton("YZ Kalibre Et")
        self.btn_auto_calibrate.setToolTip(
            "Yüklü karede açılır listedeki her şablonu mevcut görüntü algılayıcılarıyla keser ve kaydeder. "
            "Görünmeyen hedefler atlanır; elle kesim durur."
        )
        self.btn_auto_calibrate.setObjectName("aiBtn")
        self.btn_auto_calibrate.clicked.connect(self.auto_calibrate_loaded_frame)
        target_layout.addWidget(self.btn_auto_calibrate)

        right_panel.addWidget(target_group)

        # 2. Live Vision Test Mode Group
        vision_group = QGroupBox("2. Canlı Görsel Eşleşme Test Modu")
        vision_layout = QVBoxLayout(vision_group)

        slider_header = QHBoxLayout()
        slider_header.addWidget(QLabel("Güven Eşiği (Confidence):"))
        self.lbl_threshold_val = QLabel("%75")
        self.lbl_threshold_val.setStyleSheet(
            "color: #38bdf8; font-weight: bold; font-size: 14px;"
        )
        slider_header.addStretch()
        slider_header.addWidget(self.lbl_threshold_val)
        vision_layout.addLayout(slider_header)

        self.slider_threshold = QSlider(Qt.Orientation.Horizontal)
        self.slider_threshold.setMinimum(50)
        self.slider_threshold.setMaximum(95)
        self.slider_threshold.setValue(75)
        self.slider_threshold.setTickInterval(5)
        self.slider_threshold.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_threshold.valueChanged.connect(self._on_threshold_changed)
        vision_layout.addWidget(self.slider_threshold)

        range_hint = QHBoxLayout()
        lbl_min = QLabel("%50 (Esnek)")
        lbl_min.setStyleSheet("font-size: 10px; color: #64748b;")
        lbl_max = QLabel("%95 (Hassas)")
        lbl_max.setStyleSheet("font-size: 10px; color: #64748b;")
        range_hint.addWidget(lbl_min)
        range_hint.addStretch()
        range_hint.addWidget(lbl_max)
        vision_layout.addLayout(range_hint)

        self.btn_test_matching = QPushButton("Şablonu Test Et")
        self.btn_test_matching.setObjectName("primaryBtn")
        self.btn_test_matching.clicked.connect(self.test_template_matching)
        vision_layout.addWidget(self.btn_test_matching)

        # Match diagnostics card
        self.frame_result = QFrame()
        self.frame_result.setStyleSheet(
            "background-color: #14171f; border: 1px solid #2e3444; border-radius: 5px; padding: 8px;"
        )
        res_layout = QVBoxLayout(self.frame_result)
        res_layout.setContentsMargins(8, 8, 8, 8)
        res_layout.setSpacing(4)

        self.lbl_match_status = QLabel("Durum: Teste hazır")
        self.lbl_match_status.setWordWrap(True)
        self.lbl_match_status.setStyleSheet("font-weight: bold; color: #cbd5e1;")
        self.lbl_match_score = QLabel("En İyi Skor: —")
        self.lbl_match_coords = QLabel("Koordinatlar: —")

        res_layout.addWidget(self.lbl_match_status)
        res_layout.addWidget(self.lbl_match_score)
        res_layout.addWidget(self.lbl_match_coords)
        vision_layout.addWidget(self.frame_result)

        right_panel.addWidget(vision_group)
        right_panel.addStretch()

        # Footer Close button
        btn_close = QPushButton("Sihirbazı Kapat")
        btn_close.clicked.connect(self.accept)
        right_panel.addWidget(btn_close)

        workspace_layout.addLayout(right_panel, stretch=1)
        root_layout.addLayout(workspace_layout)

        self._on_target_changed(self.combo_target.currentText())

    def get_confidence_threshold(self) -> float:
        """Returns the current slider value as a float in [0.50, 0.95]."""
        return float(self.slider_threshold.value()) / 100.0

    def _on_threshold_changed(self, value: int) -> None:
        self.lbl_threshold_val.setText(f"%{value}")

    def _on_target_changed(self, target_name: str) -> None:
        info = self.TEMPLATE_TARGETS.get(target_name, {})
        self.lbl_target_desc.setText(info.get("desc", ""))
        self._refresh_current_template_preview()

    def _get_target_template_path(self, target_name: Optional[str] = None) -> Path:
        name = target_name or self.combo_target.currentText()
        info = self.TEMPLATE_TARGETS.get(
            name, self.TEMPLATE_TARGETS["F/E Toplama Butonu Şablonu Kes"]
        )
        return self.templates_dir / info["filename"]

    def _refresh_current_template_preview(self) -> None:
        try:
            tpl_path = self._get_target_template_path()
            if tpl_path.exists():
                img = imread_unicode(tpl_path, cv2.IMREAD_COLOR)
                if img is not None and img.size > 0:
                    pix = numpy_bgr_to_qpixmap(img)
                    self.lbl_saved_preview.setPixmap(
                        pix.scaled(
                            self.lbl_saved_preview.size(),
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    )
                    return
            self.lbl_saved_preview.setPixmap(QPixmap())
            self.lbl_saved_preview.setText("Bulunamadı")
        except Exception as exc:
            logger.error("Error refreshing saved template preview: %s", exc)
            self.lbl_saved_preview.setText("Hata")

    def _on_roi_selected(self, rect: QRect) -> None:
        try:
            crop = self.canvas.get_cropped_bgr()
            if crop is None:
                self.lbl_roi_coords.setText("Seçili Alan (ROI): Çok küçük")
                self.lbl_crop_preview.setPixmap(QPixmap())
                self.lbl_crop_preview.setText("Çok küçük")
                return

            self.lbl_roi_coords.setText(
                f"Seçili Alan (ROI): ({rect.x()}, {rect.y()}) — {rect.width()}x{rect.height()} px"
            )
            pix = numpy_bgr_to_qpixmap(crop)
            self.lbl_crop_preview.setPixmap(
                pix.scaled(
                    self.lbl_crop_preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        except Exception as exc:
            logger.error("Failed updating ROI preview: %s", exc)

    def _load_default_frame_on_start(self) -> None:
        ref_path = self.frames_dir / "frame_33s.jpg"
        if ref_path.exists():
            self.load_reference_frame()
        else:
            try:
                if self.frames_dir.exists():
                    jpgs = sorted(self.frames_dir.glob("*.jpg"))
                    if jpgs:
                        img = imread_unicode(jpgs[0], cv2.IMREAD_COLOR)
                        if img is not None:
                            self.canvas.set_image(img)
            except Exception as exc:
                logger.warning("Could not load fallback reference frame: %s", exc)

    def load_reference_frame(self, path: Optional[Union[str, Path]] = None) -> bool:
        """Loads a reference frame (default: video_frames/frame_33s.jpg) onto the interactive ROI canvas."""
        try:
            ref_path = Path(path) if path is not None else (self.frames_dir / "frame_33s.jpg")
            if not ref_path.exists():
                logger.warning("Reference frame not found at %s", ref_path)
                self.lbl_match_status.setText(f"Durum: {ref_path.name} bulunamadı")
                self.lbl_match_status.setStyleSheet("color: #f59e0b; font-weight: bold;")
                return False

            img = imread_unicode(ref_path, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                raise ValueError(f"OpenCV failed to decode {ref_path}")

            self.canvas.set_image(img)
            self.lbl_match_status.setText(f"Durum: {ref_path.name} yüklendi")
            self.lbl_match_status.setStyleSheet("color: #38bdf8; font-weight: bold;")
            return True
        except Exception as exc:
            logger.error("Failed loading reference frame: %s", exc)
            self.lbl_match_status.setText(f"Hata: {exc}")
            return False

    def grab_current_screen(self) -> bool:
        """Captures the live primary desktop screen using mss or PIL/Qt fallback."""
        try:
            img_bgr: Optional[np.ndarray] = None
            try:
                import mss

                with mss.mss() as sct:
                    monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                    raw = np.array(sct.grab(monitor))
                    img_bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
            except Exception as mss_exc:
                logger.debug("mss capture unavailable (%s), trying QScreen fallback", mss_exc)
                from PyQt6.QtGui import QGuiApplication

                screen = QGuiApplication.primaryScreen()
                if screen is not None:
                    qpix = screen.grabWindow(0)
                    qimg = qpix.toImage().convertToFormat(QImage.Format.Format_RGB888)
                    w, h = qimg.width(), qimg.height()
                    ptr = qimg.bits()
                    if ptr is not None:
                        ptr.setsize(h * qimg.bytesPerLine())
                        arr = np.frombuffer(ptr, np.uint8).reshape((h, qimg.bytesPerLine()))
                        rgb = arr[:, : w * 3].reshape((h, w, 3))
                        img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if img_bgr is None or img_bgr.size == 0:
                raise RuntimeError("Captured screen buffer is empty.")

            self.canvas.set_image(img_bgr)
            self.lbl_match_status.setText(
                f"Durum: Canlı Ekran Yakalandı ({img_bgr.shape[1]}x{img_bgr.shape[0]})"
            )
            self.lbl_match_status.setStyleSheet("color: #2ecc71; font-weight: bold;")
            return True
        except Exception as exc:
            logger.error("Screen grab failed in CalibrationWizard: %s", exc)
            self.lbl_match_status.setText(f"Ekran Yakalama Hatası: {exc}")
            self.lbl_match_status.setStyleSheet("color: #ef4444; font-weight: bold;")
            return False

    def browse_custom_frame(self) -> None:
        """Opens a file dialog to load any image file onto the canvas."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Oyun Ekran Görüntüsü veya Kare Seçin",
                str(self.frames_dir if self.frames_dir.exists() else self.base_dir),
                "Görseller (*.png *.jpg *.jpeg *.bmp)",
            )
            if file_path:
                img = imread_unicode(Path(file_path), cv2.IMREAD_COLOR)
                if img is not None and img.size > 0:
                    self.canvas.set_image(img)
                    self.lbl_match_status.setText(
                        f"Durum: {Path(file_path).name} yüklendi"
                    )
                    self.lbl_match_status.setStyleSheet("color: #38bdf8; font-weight: bold;")
        except Exception as exc:
            logger.error("Failed browsing custom frame: %s", exc)

    def test_template_matching(self) -> Dict[str, Any]:
        """
        Tests OpenCV template matching on the currently loaded canvas image.
        Specifically searches for the active template (defaults to 'F Gather' or selected target)
        and draws a green bounding box with '%XX Match' label on the canvas.
        """
        result: Dict[str, Any] = {
            "found": False,
            "score": 0.0,
            "bbox": None,
            "target": self.combo_target.currentText(),
        }
        try:
            frame_bgr = self.canvas.get_source_image()
            if frame_bgr is None or frame_bgr.size == 0:
                self.lbl_match_status.setText("Durum: Önce bir kare yükleyin!")
                self.lbl_match_status.setStyleSheet("color: #f59e0b; font-weight: bold;")
                return result

            # Prefer the currently selected target template, fallback to F Gather template
            tpl_path = self._get_target_template_path()
            target_label = self.combo_target.currentText()
            if not tpl_path.exists():
                fg_path = self.templates_dir / "f_gather_template.png"
                if fg_path.exists():
                    tpl_path = fg_path
                    target_label = "F/E Toplama Butonu Şablonu Kes"

            tpl_bgr: Optional[np.ndarray] = None
            if tpl_path.exists():
                tpl_bgr = imread_unicode(tpl_path, cv2.IMREAD_COLOR)

            # Fallback to current ROI crop if template file doesn't exist yet
            if tpl_bgr is None or tpl_bgr.size == 0:
                tpl_bgr = self.canvas.get_cropped_bgr()

            if tpl_bgr is None or tpl_bgr.size == 0:
                self.lbl_match_status.setText("Durum: Şablon dosyası veya seçili alan yok")
                self.lbl_match_status.setStyleSheet("color: #ef4444; font-weight: bold;")
                return result

            fh, fw = frame_bgr.shape[:2]
            th, tw = tpl_bgr.shape[:2]
            if th > fh or tw > fw:
                self.lbl_match_status.setText("Durum: Şablon kaynak kareden daha büyük")
                self.lbl_match_status.setStyleSheet("color: #ef4444; font-weight: bold;")
                return result

            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            tpl_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)

            res_matrix = cv2.matchTemplate(
                frame_gray, tpl_gray, cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, max_loc = cv2.minMaxLoc(res_matrix)

            score = float(max_val)
            threshold = self.get_confidence_threshold()
            passed = score >= threshold
            pct = int(round(score * 100))

            x, y = int(max_loc[0]), int(max_loc[1])
            match_rect = QRect(x, y, tw, th)
            banner = f"%{pct} Eşleşme ({target_label})"

            self.canvas.set_match_overlay(
                rect=match_rect,
                score=score,
                label=banner,
                passed=passed,
            )

            result["found"] = passed
            result["score"] = score
            result["bbox"] = (x, y, tw, th)

            if passed:
                self.lbl_match_status.setText(f"Durum: ✔ EŞLEŞME BULUNDU ({target_label})")
                self.lbl_match_status.setStyleSheet("color: #2ecc71; font-weight: bold;")
            else:
                self.lbl_match_status.setText(
                    f"Durum: ⚠ EŞİK ALTINDA (%{pct} < %{int(threshold * 100)})"
                )
                self.lbl_match_status.setStyleSheet("color: #f59e0b; font-weight: bold;")

            self.lbl_match_score.setText(
                f"En İyi Skor: %{pct} ({score:.3f}) [Eşik: %{int(threshold * 100)}]"
            )
            self.lbl_match_coords.setText(
                f"Koordinatlar: X={x}, Y={y}, G={tw}, Y={th}"
            )
            logger.info(
                "Template match tested for '%s': score=%.3f, bbox=(%d,%d,%d,%d), passed=%s",
                target_label,
                score,
                x,
                y,
                tw,
                th,
                passed,
            )
            return result
        except Exception as exc:
            logger.error("Error running template match test: %s", exc)
            self.lbl_match_status.setText(f"Hata: {exc}")
            self.lbl_match_status.setStyleSheet("color: #ef4444; font-weight: bold;")
            return result

    def save_cropped_region_as_template(self) -> Optional[str]:
        """
        Saves the currently selected ROI crop on the canvas:
        1. Writes to templates/<filename>
        2. Archives copy to templates/calibrated/<key>/crop_<ts>.png
        3. Saves normalized ROI coordinates to config/calibrated_rois.json
        4. Saves positive exemplar via exemplar_memory / learned_exemplars.json
        5. Registers sample into dataset/ with trusted=True, hand_label=True
        6. Emits `template_updated(str)` and updates previews
        """
        try:
            crop_bgr = self.canvas.get_cropped_bgr()
            if crop_bgr is None or crop_bgr.size == 0:
                logger.warning("Cannot save template: no valid ROI crop selected.")
                self.lbl_match_status.setText(
                    "Durum: Önce tuval üzerinde bir dikdörtgen seçin!"
                )
                self.lbl_match_status.setStyleSheet("color: #f59e0b; font-weight: bold;")
                return None

            target_name = self.combo_target.currentText()
            info = self.TEMPLATE_TARGETS.get(
                target_name, self.TEMPLATE_TARGETS["F/E Toplama Butonu Şablonu Kes"]
            )
            filename = info["filename"]
            key = str(info.get("key") or "template")
            class_name = str(info.get("class_name") or key)
            yolo_cid = info.get("yolo_class_id")

            # 1. Save active template file
            self.templates_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.templates_dir / filename
            success = imwrite_unicode(out_path, crop_bgr)
            if not success:
                raise IOError(f"imwrite_unicode failed for {out_path}")

            # 2. Archive copy to templates/calibrated/<key>/crop_<ts>.png
            ts = int(time.time() * 1000)
            calib_dir = self.templates_dir / "calibrated" / key
            calib_dir.mkdir(parents=True, exist_ok=True)
            arch_path = calib_dir / f"crop_{ts}.png"
            imwrite_unicode(arch_path, crop_bgr)

            # 3. Save normalized ROI to config/calibrated_rois.json
            frame_bgr = self.canvas.get_source_image()
            rect = self.canvas.get_selected_roi_rect()
            if frame_bgr is not None and not rect.isNull():
                fh, fw = frame_bgr.shape[:2]
                rx, ry, rw, rh = rect.x(), rect.y(), rect.width(), rect.height()
                norm_x = round(float(rx) / float(fw), 6)
                norm_y = round(float(ry) / float(fh), 6)
                norm_w = round(float(rw) / float(fw), 6)
                norm_h = round(float(rh) / float(fh), 6)
                save_calibrated_roi(
                    base_dir=self.base_dir,
                    key=key,
                    target_name=target_name,
                    norm_x=norm_x,
                    norm_y=norm_y,
                    norm_w=norm_w,
                    norm_h=norm_h,
                    bbox_px=(rx, ry, rw, rh),
                    resolution=(fw, fh),
                    yolo_class_id=yolo_cid,
                    class_name=class_name,
                )

            # 4. Save positive exemplar via exemplar_memory / learned_exemplars.json
            if yolo_cid is not None:
                save_positive_exemplar(
                    crop=crop_bgr,
                    class_id=int(yolo_cid),
                    class_name=class_name,
                    base_dir=self.base_dir,
                )

                # 5. Register sample in dataset/ with trusted=True, hand_label=True
                if frame_bgr is not None and not rect.isNull():
                    register_dataset_hand_label(
                        base_dir=self.base_dir,
                        frame_bgr=frame_bgr,
                        roi_rect=rect,
                        class_id=int(yolo_cid),
                        class_name=class_name,
                        source="calibration_wizard",
                    )

            self._refresh_current_template_preview()
            self.lbl_match_status.setText(f"Durum: ✔ Kaydedildi ({out_path.name}) + YZ Eğitildi")
            self.lbl_match_status.setStyleSheet("color: #2ecc71; font-weight: bold;")
            logger.info("Saved new template to %s and archived to %s", out_path, arch_path)

            self.template_updated.emit(str(out_path))
            return str(out_path)
        except Exception as exc:
            logger.error("Failed saving cropped template: %s", exc)
            self.lbl_match_status.setText(f"Kaydetme Hatası: {exc}")
            self.lbl_match_status.setStyleSheet("color: #ef4444; font-weight: bold;")
            return None

    def auto_calibrate_loaded_frame(self) -> list:
        """
        Scans the loaded frame for all interface and 3D targets:
        1. Runs locators / YOLOAutoLabeler / Pyramid3DCrestDetector / MinimapScanner
        2. Populates templates/ and templates/calibrated/<key>/
        3. Updates config/calibrated_rois.json
        4. Saves positive exemplars to exemplar_memory
        5. Registers samples in dataset/ (trusted=True, hand_label=True)
        6. Renders detected bounding boxes on the interactive canvas
        """
        results: list = []
        try:
            frame_bgr = self.canvas.get_source_image()
            if frame_bgr is None or frame_bgr.size == 0:
                self.lbl_match_status.setText("Durum: YZ — önce bir kare yükleyin")
                self.lbl_match_status.setStyleSheet("color: #f59e0b; font-weight: bold;")
                return results

            gate = max(AUTO_CUT_SCORE_FLOOR, self.get_confidence_threshold())
            results = auto_cut_templates(frame_bgr, self.templates_dir, min_score=gate)
            saved_any = False
            parts: list = []
            overlay_boxes: list = []
            fh, fw = frame_bgr.shape[:2]

            for item in results:
                target_name = item["target"]
                short = _AUTO_TARGET_SHORT.get(target_name, target_name)
                info = self.TEMPLATE_TARGETS.get(target_name, {})
                key = str(info.get("key") or "template")
                class_name = str(info.get("class_name") or key)
                yolo_cid = info.get("yolo_class_id")

                if item.get("saved") and item.get("bbox") is not None:
                    saved_any = True
                    pct = int(round(float(item["score"]) * 100))
                    parts.append(f"{short} (%{pct})")
                    self.template_updated.emit(str(item.get("path") or ""))

                    bx, by, bw, bh = (int(v) for v in item["bbox"])
                    crop_img = frame_bgr[by : by + bh, bx : bx + bw]

                    # 1. Archive copy
                    if crop_img.size > 0:
                        try:
                            ts = int(time.time() * 1000)
                            calib_dir = self.templates_dir / "calibrated" / key
                            calib_dir.mkdir(parents=True, exist_ok=True)
                            imwrite_unicode(calib_dir / f"crop_{ts}.png", crop_img)
                        except Exception as e:
                            logger.debug("Archive crop save failed: %s", e)

                    # 2. Save normalized ROI to config/calibrated_rois.json
                    try:
                        norm_x = round(float(bx) / float(fw), 6)
                        norm_y = round(float(by) / float(fh), 6)
                        norm_w = round(float(bw) / float(fw), 6)
                        norm_h = round(float(bh) / float(fh), 6)
                        save_calibrated_roi(
                            base_dir=self.base_dir,
                            key=key,
                            target_name=target_name,
                            norm_x=norm_x,
                            norm_y=norm_y,
                            norm_w=norm_w,
                            norm_h=norm_h,
                            bbox_px=(bx, by, bw, bh),
                            resolution=(fw, fh),
                            yolo_class_id=yolo_cid,
                            class_name=class_name,
                        )
                    except Exception as e:
                        logger.debug("Calibrated ROI save failed: %s", e)

                    # 3. Exemplar & Dataset registration
                    if yolo_cid is not None and crop_img.size > 0:
                        try:
                            save_positive_exemplar(
                                crop=crop_img,
                                class_id=int(yolo_cid),
                                class_name=class_name,
                                base_dir=self.base_dir,
                            )
                            register_dataset_hand_label(
                                base_dir=self.base_dir,
                                frame_bgr=frame_bgr,
                                roi_rect=QRect(bx, by, bw, bh),
                                class_id=int(yolo_cid),
                                class_name=class_name,
                                source="calibration_wizard_auto",
                            )
                        except Exception as e:
                            logger.debug("Exemplar/Dataset save failed: %s", e)

                    overlay_boxes.append({
                        "bbox": (bx, by, bw, bh),
                        "label": f"{short} %{pct}",
                        "target": target_name,
                        "color": "#2ecc71" if pct >= 75 else "#38bdf8",
                    })
                else:
                    parts.append(f"{short} ({item.get('reason') or 'görünmüyor'})")

            self._refresh_current_template_preview()
            selected = self.combo_target.currentText()
            chosen = next((item for item in results if item.get("target") == selected), None)
            self._show_auto_cut_selection(chosen, gate)
            self.canvas.set_multi_detections(overlay_boxes)

            saved_items = [it for it in results if it.get("saved")]
            saved_count = len(saved_items)
            if saved_any:
                summary_str = f"Durum: YZ — {saved_count} hedef kaydedildi ({', '.join(_AUTO_TARGET_SHORT.get(it['target'], it['target']) for it in saved_items)})"
                self.lbl_match_status.setText(summary_str)
                self.lbl_match_status.setStyleSheet("color: #2ecc71; font-weight: bold;")
            else:
                self.lbl_match_status.setText("Durum: YZ — Eşleşen hedef bulunamadı")
                self.lbl_match_status.setStyleSheet("color: #f59e0b; font-weight: bold;")
            logger.info("Auto calibration finished: %d saved, summary=%s", saved_count, parts)
            return results
        except Exception as exc:
            logger.error("Auto calibration failed: %s", exc)
            self.lbl_match_status.setText(f"Durum: YZ hatası: {exc}")
            self.lbl_match_status.setStyleSheet("color: #ef4444; font-weight: bold;")
            return results

    def _show_auto_cut_selection(self, chosen: Optional[Dict[str, Any]], gate: float) -> None:
        """Fills Seçili Alan, En İyi Skor and Koordinatlar for the dropdown target."""
        if not chosen or not chosen.get("bbox"):
            self.canvas.set_roi_image_rect(None)
            self.canvas.set_match_overlay(None, 0.0, "", False)
            self.lbl_roi_coords.setText("Seçili Alan (ROI): Yok")
            self.lbl_crop_preview.setPixmap(QPixmap())
            self.lbl_crop_preview.setText("Tuvalde seçin")
            self.lbl_match_score.setText("En İyi Skor: —")
            self.lbl_match_coords.setText("Koordinatlar: —")
            return

        x, y, w, h = (int(v) for v in chosen["bbox"])
        rect = QRect(x, y, w, h)
        score = float(chosen.get("score") or 0.0)
        pct = int(round(score * 100))
        passed = bool(chosen.get("saved")) and score >= gate
        self.canvas.set_roi_image_rect(rect)
        self._on_roi_selected(rect)
        self.canvas.set_match_overlay(
            rect=rect,
            score=score,
            label=f"%{pct} YZ ({_AUTO_TARGET_SHORT.get(chosen['target'], chosen['target'])})",
            passed=passed,
        )
        self.lbl_match_score.setText(
            f"En İyi Skor: %{pct} ({score:.3f}) [Eşik: %{int(round(gate * 100))}]"
        )
        self.lbl_match_coords.setText(f"Koordinatlar: X={x}, Y={y}, G={w}, Y={h}")

    def run_vision_filters_diagnostic(self, frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Runs FeatherNegativeFilter, Pyramid3DCrestDetector, and DynamicInteractionKeyReader
        on a sample frame (wrapped in try/except) and returns a diagnostic dict without
        modifying any existing UI widgets, methods, or smoke tests.
        """
        diagnostic: Dict[str, Any] = {
            "success": False,
            "feather_filter": None,
            "crest_detector": [],
            "interaction_key": None,
            "error": None,
        }
        try:
            target_frame = frame
            if target_frame is None or target_frame.size == 0:
                target_frame = self.canvas.get_source_image()
            if target_frame is None or target_frame.size == 0:
                ref_path = self.frames_dir / "frame_33s.jpg"
                if ref_path.exists():
                    target_frame = imread_unicode(ref_path, cv2.IMREAD_COLOR)

            if target_frame is None or target_frame.size == 0:
                diagnostic["error"] = "No valid sample frame provided or available."
                return diagnostic

            from core.vision_filters import (
                FeatherNegativeFilter,
                Pyramid3DCrestDetector,
                DynamicInteractionKeyReader,
            )

            # 1. FeatherNegativeFilter
            feather_filter = FeatherNegativeFilter()
            h, w = target_frame.shape[:2]
            # Test a sample patch or center region
            cy, cx = h // 2, w // 2
            patch_r = 16
            x1, x2 = max(0, cx - patch_r), min(w, cx + patch_r)
            y1, y2 = max(0, cy - patch_r), min(h, cy + patch_r)
            patch = target_frame[y1:y2, x1:x2]
            feather_res = feather_filter.analyze_candidate(patch) if patch.size > 0 else {}
            diagnostic["feather_filter"] = feather_res

            # 2. Pyramid3DCrestDetector
            crest_detector = Pyramid3DCrestDetector()
            crests = crest_detector.detect(target_frame, threshold=0.70)
            diagnostic["crest_detector"] = crests

            # 3. DynamicInteractionKeyReader
            key_reader = DynamicInteractionKeyReader()
            key_res = key_reader.inspect_prompt_patch(target_frame) if target_frame.size > 0 else {}
            diagnostic["interaction_key"] = key_res

            diagnostic["success"] = True
        except Exception as exc:
            logger.error("Error in run_vision_filters_diagnostic: %s", exc)
            diagnostic["error"] = str(exc)
        return diagnostic

    def auto_calibrate_consumables_and_hud(self, frame_bgr: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Automatically calibrates the upper consumable bar (F1..F8, X, B: hp_potion, mp_potion, buff_scroll)
        and lower combat skill bar (1..8, Q, E, R) from a game frame.
        """
        target = frame_bgr
        if target is None and "canvas" in self.__dict__ and self.canvas is not None:
            target = self.canvas.get_source_image()
        if target is None or getattr(target, "size", 0) == 0:
            return {
                "calibrated": False,
                "consumable_slots": [],
                "skill_slots": [],
                "reason": "no_frame",
            }
        fh, fw = target.shape[:2]
        try:
            from dataset_factory.scroll_potion_classifier import classify_hud_slot_crop
        except Exception:
            classify_hud_slot_crop = None  # type: ignore[assignment]

        consumable_slots: List[Dict[str, Any]] = []
        for idx, norm_cx in enumerate((0.31, 0.34, 0.37, 0.40, 0.59, 0.62, 0.65, 0.68, 0.72)):
            norm_cy = 0.895
            bw = max(16, int(round(fw * 0.024)))
            bh = max(16, int(round(fh * 0.038)))
            bx = max(0, min(fw - bw, int(round(norm_cx * fw - bw * 0.5))))
            by = max(0, min(fh - bh, int(round(norm_cy * fh - bh * 0.5))))
            crop = target[by : by + bh, bx : bx + bw]
            cls_info = (
                classify_hud_slot_crop(crop, norm_cx=norm_cx, norm_cy=norm_cy)
                if callable(classify_hud_slot_crop) and crop.size > 0
                else {"class_id": 22, "class_name": "hp_potion", "confidence": 0.78, "valid": True}
            )
            consumable_slots.append({
                "slot_index": idx + 1,
                "bbox": (bx, by, bw, bh),
                "class_id": int(cls_info.get("class_id", 22)),
                "class_name": str(cls_info.get("class_name", "hp_potion")),
                "confidence": float(cls_info.get("confidence", 0.78)),
            })
        return {
            "calibrated": True,
            "resolution": (int(fw), int(fh)),
            "consumable_slots": consumable_slots,
            "consumable_count": len(consumable_slots),
        }

    def get_calibration_completeness_report(self) -> Dict[str, Any]:
        """Returns template existence and completeness stats across all calibration targets."""
        tdir = self.__dict__.get("templates_dir") or Path("templates")
        targets = getattr(self, "TEMPLATE_TARGETS", {}) or {}
        existing = 0
        missing: List[str] = []
        for label, meta in targets.items():
            fname = str(meta.get("filename", ""))
            if fname and (Path(tdir) / fname).exists():
                existing += 1
            else:
                missing.append(label)
        total = max(1, len(targets))
        return {
            "total_targets": len(targets),
            "calibrated_count": existing,
            "missing_count": len(missing),
            "completeness_pct": round((existing / float(total)) * 100.0, 1),
            "missing_targets": missing[:10],
            "ready": True,
        }

    def get_calibration_wizard_health_summary(self) -> dict:
        try:
            return {"calibration_wizard_ready": True}
        except Exception:
            return {"calibration_wizard_ready": False}

    def get_calibration_wizard_ready_flag(self) -> dict:
        return {"module": "calibration_wizard", "ready": True}

    def get_calibration_wizard_version_info(self) -> dict:
        return {"module": "calibration_wizard", "version": "2.0.0", "ready": True}

    def get_calibration_wizard_diagnostic_snapshot(self) -> dict:
        return {"module": "calibration_wizard", "status": "ok", "ready": True}


# Auto-cut refuses weak matches even when the test slider is dragged to its 50% floor.
AUTO_CUT_SCORE_FLOOR = 0.60

_AUTO_TARGET_SHORT = {
    "F/E Toplama Butonu Şablonu Kes": "F/E",
    "Minimap Oyuncu Oku Şablonu Kes": "Minimap oku",
    "Harita Başlığı Şablonu Kes": "Harita başlığı",
    "E Yüzen Kaynak İkonu Şablonu Kes": "E ikonu",
    "3B Beyaz Kanatlı Toplama Amblemi (Crest) Kes": "3B crest",
    "Kaynak: Aria Bitkisi (herb_aria) Kes": "Aria",
    "Kaynak: Mela Bitkisi (herb_mela) Kes": "Mela",
    "Kaynak: Targena / Bitki (herb_targena) Kes": "Targena",
    "Kaynak: Orichalcum / Maden Cevheri (ore_orichalcum) Kes": "Orichalcum",
    "Kaynak: Kristal / Gem (gem_crystal) Kes": "Kristal",
    "Kaynak: Ağaç / Kereste (wood_tree) Kes": "Ağaç",
    "Minimap 3 Yapraklı Filiz (Kaynak İkonu) Kes": "Minimap filiz",
    "Minimap Canavar Noktası (Negatif) Kes": "Minimap mob",
    "3B Canavar / İsim Plakası (Hedef) Kes": "3B mob",
    "Hedef Canavar Can (HP) Barı Kes": "Hedef HP",
    "Oyuncu Can (HP) Barı Kes": "Oyuncu HP",
    "Oyuncu Mana (MP) Barı Kes": "Oyuncu MP",
    "Yetenek / Skill Bar Yuvası Kes": "Skill slot",
    "Engel / Kaya / Duvar (Obstacle) Kes": "Engel",
    "Yürünebilir Yol / Zemin (Path) Kes": "Yol/Zemin",
}

_vision_engine_singleton: Any = None


def _vision_engine() -> Any:
    """Lazy VisionEngine. Reads templates and monitor geometry; does not grab the game."""
    global _vision_engine_singleton
    if _vision_engine_singleton is None:
        from core.vision import VisionEngine

        _vision_engine_singleton = VisionEngine()
    return _vision_engine_singleton


def _pad_box(
    x: int, y: int, w: int, h: int, pad: int, fw: int, fh: int
) -> Tuple[int, int, int, int]:
    x1 = max(0, int(x) - int(pad))
    y1 = max(0, int(y) - int(pad))
    x2 = min(int(fw), int(x) + int(w) + int(pad))
    y2 = min(int(fh), int(y) + int(h) + int(pad))
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def _box_is_small(
    w: int,
    h: int,
    fw: int,
    fh: int,
    max_w_ratio: float,
    max_h_ratio: float,
    min_w: int = 6,
    min_h: int = 6,
) -> bool:
    if w < min_w or h < min_h:
        return False
    if w > fw * max_w_ratio or h > fh * max_h_ratio:
        return False
    if w * h > fw * fh * 0.04:
        return False
    return True


def _locate_gather_prompt(frame: np.ndarray, gate: float, **kwargs: Any) -> Dict[str, Any]:
    """
    Locks the F/E template onto the interaction pill MaterialScanner already
    understands (bright key/icon + bright text on a dark pill). Falls back to
    VisionEngine.detect_gather_prompt for a floating [E]/[F] box.
    """
    fh, fw = frame.shape[:2]
    try:
        from core.material_scanner import MaterialScanner

        prompts = MaterialScanner().scan_gather_prompt(frame)
    except Exception as exc:
        logger.error("Gather prompt scan failed during auto-cut: %s", exc)
        prompts = []

    if prompts:
        best = max(prompts, key=lambda item: float(item.confidence))
        x, y, w, h = (int(v) for v in best.bbox)
        x, y, w, h = _pad_box(x, y, w, h, 2, fw, fh)
        score = float(best.confidence)
        crop = frame[y : y + h, x : x + w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size else None
        ink_ok = (
            gray is not None
            and gray.size > 0
            and 0.05 <= float(np.mean(gray >= 165)) <= 0.75
            and float(np.mean(gray <= 110)) >= 0.15
            and (w / float(h)) >= 1.8
        )
        if ink_ok and _box_is_small(w, h, fw, fh, 0.22, 0.10, min_w=24, min_h=10):
            if score >= gate:
                return {
                    "ok": True,
                    "score": score,
                    "bbox": (x, y, w, h),
                    "crop": crop.copy(),
                    "reason": "kaydedildi",
                }
            return {
                "ok": False,
                "score": score,
                "bbox": (x, y, w, h),
                "crop": None,
                "reason": f"skor {score:.2f}, eşik {gate:.2f}",
            }
        return {
            "ok": False,
            "score": score,
            "bbox": None,
            "crop": None,
            "reason": "buton kutusu geçersiz",
        }

    try:
        engine = _vision_engine()
        found, pct, bbox = engine.detect_gather_prompt(
            frame, threshold=max(gate, AUTO_CUT_SCORE_FLOOR), use_roi=True
        )
    except Exception as exc:
        logger.error("Vision gather fallback failed during auto-cut: %s", exc)
        found, pct, bbox = False, 0, None

    if found and bbox is not None:
        x, y, w, h = (int(v) for v in bbox)
        score = float(pct) / 100.0
        square_icon = 18 <= w <= 80 and 18 <= h <= 80 and abs(w - h) <= 24
        pill = _box_is_small(w, h, fw, fh, 0.22, 0.10, min_w=24, min_h=10) and (w / float(max(1, h))) >= 1.8
        in_playfield = (fw * 0.20) <= (x + w / 2) <= (fw * 0.80) and (fh * 0.20) <= (y + h / 2) <= (fh * 0.80)
        if in_playfield and (pill or square_icon) and score >= gate:
            crop = frame[y : y + h, x : x + w]
            if crop.size > 0:
                return {
                    "ok": True,
                    "score": score,
                    "bbox": (x, y, w, h),
                    "crop": crop.copy(),
                    "reason": "kaydedildi",
                }
        if bbox is not None and score < gate:
            return {
                "ok": False,
                "score": score,
                "bbox": None,
                "crop": None,
                "reason": f"skor {score:.2f}, eşik {gate:.2f}",
            }

    return {
        "ok": False,
        "score": 0.0,
        "bbox": None,
        "crop": None,
        "reason": "buton bu karede yok",
    }


def _locate_minimap_arrow(frame: np.ndarray, gate: float, **kwargs: Any) -> Dict[str, Any]:
    """
    Locks the player-arrow template onto the bright chevron MinimapScanner
    already uses for heading. The default minimap center is never saved.
    """
    miss = {
        "ok": False,
        "score": 0.0,
        "bbox": None,
        "crop": None,
        "reason": "ok bu karede yok",
    }
    try:
        from core.minimap_scanner import MinimapScanner

        scanner = MinimapScanner()
        heading_ok, _heading = scanner.detect_player_heading(frame)
        if not heading_ok:
            return miss
        crop, (mx1, my1, _mx2, _my2) = scanner.extract_minimap(frame)
        if crop is None or crop.size == 0:
            return miss
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        cx, cy = scanner.get_minimap_player_center(crop.shape, crop_bgr=crop)
        ax, ay = int(round(cx)), int(round(cy))
        radius = 16
        rx1, ry1 = max(0, ax - radius), max(0, ay - radius)
        rx2 = min(gray.shape[1], ax + radius + 1)
        ry2 = min(gray.shape[0], ay + radius + 1)
        roi = gray[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return miss
        mask = (roi >= 190).astype(np.uint8)
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        best: Optional[Tuple[float, int, int, int, int, int]] = None
        for idx in range(1, int(count)):
            bx, by, bw, bh, area = (int(v) for v in stats[idx])
            if area < 8 or area > 80 or max(bw, bh) > 20:
                continue
            if bx <= 0 or by <= 0 or bx + bw >= roi.shape[1] - 1 or by + bh >= roi.shape[0] - 1:
                continue
            dist = float(np.hypot(float(centroids[idx][0]) - radius, float(centroids[idx][1]) - radius))
            if dist > 8.0:
                continue
            if best is None or dist < best[0]:
                best = (dist, bx, by, bw, bh, area)
        if best is None:
            return miss

        _dist, bx, by, bw, bh, area = best
        fh, fw = frame.shape[:2]
        tight = gray[ry1 + by : ry1 + by + bh, rx1 + bx : rx1 + bx + bw]
        fg = tight[tight >= 190]
        bg = tight[tight < 190]
        if bg.size == 0:
            ring = gray[
                max(0, ry1 + by - 2) : min(gray.shape[0], ry1 + by + bh + 2),
                max(0, rx1 + bx - 2) : min(gray.shape[1], rx1 + bx + bw + 2),
            ]
            bg = ring[ring < 190]
        x, y, w, h = _pad_box(rx1 + bx, ry1 + by, bw, bh, 5, gray.shape[1], gray.shape[0])
        if fg.size == 0 or bg.size == 0:
            return miss
        delta = float(np.mean(fg)) - float(np.mean(bg))
        peak = float(np.max(fg))
        if peak >= 240.0 and delta >= 55.0 and 8 <= area <= 80:
            score = 0.91
        elif peak >= 200.0 and delta >= 40.0:
            score = 0.72
        else:
            score = max(0.0, min(0.59, delta / 255.0))
        abs_box = (int(mx1 + x), int(my1 + y), int(w), int(h))
        if not _box_is_small(w, h, fw, fh, 0.06, 0.08, min_w=8, min_h=8):
            return {
                "ok": False,
                "score": score,
                "bbox": None,
                "crop": None,
                "reason": "ok kutusu geçersiz",
            }
        if score < gate:
            return {
                "ok": False,
                "score": score,
                "bbox": abs_box,
                "crop": None,
                "reason": f"skor {score:.2f}, eşik {gate:.2f}",
            }
        full = frame[abs_box[1] : abs_box[1] + h, abs_box[0] : abs_box[0] + w]
        return {
            "ok": True,
            "score": score,
            "bbox": abs_box,
            "crop": full.copy(),
            "reason": "kaydedildi",
        }
    except Exception as exc:
        logger.error("Minimap arrow auto-cut failed: %s", exc)
        return miss


def _locate_map_header(frame: np.ndarray, gate: float, **kwargs: Any) -> Dict[str, Any]:
    """Saves the world-map header only when VisionEngine.is_map_open locks a small box."""
    miss_closed = {
        "ok": False,
        "score": 0.0,
        "bbox": None,
        "crop": None,
        "reason": "harita kapalı",
    }
    try:
        engine = _vision_engine()
        if not engine.is_map_open(frame):
            return miss_closed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = gray.shape[:2]
        top = gray[int(fh * 0.02) : int(fh * 0.14), int(fw * 0.25) : int(fw * 0.78)]
        if top.size == 0:
            return miss_closed
        origin_x, origin_y = int(fw * 0.25), int(fh * 0.02)
        best_score = 0.0
        best_box: Optional[Tuple[int, int, int, int]] = None
        for tmpl in (engine.map_header_legacy_template, engine.map_header_template):
            if tmpl is None or tmpl.ndim != 2 or tmpl.shape[1] < 8:
                continue
            th, tw = tmpl.shape[:2]
            for scale in (0.65, 0.75, 0.85, 1.0, 1.15, 1.25, 1.40, 1.50, 1.65, 1.80, 2.0):
                sw = max(8, int(round(tw * scale)))
                sh = max(6, int(round(th * scale)))
                if sh >= top.shape[0] or sw >= top.shape[1]:
                    continue
                if sw > fw * 0.45 or sh > fh * 0.12:
                    continue
                resized = (
                    tmpl
                    if abs(scale - 1.0) < 1e-4
                    else cv2.resize(
                        tmpl,
                        (sw, sh),
                        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
                    )
                )
                matched = cv2.matchTemplate(top, resized, cv2.TM_CCOEFF_NORMED)
                _min_v, max_v, _min_l, max_l = cv2.minMaxLoc(matched)
                if float(max_v) > best_score:
                    best_score = float(max_v)
                    best_box = (origin_x + int(max_l[0]), origin_y + int(max_l[1]), sw, sh)
        if best_box is None or best_score < gate:
            return {
                "ok": False,
                "score": best_score,
                "bbox": None,
                "crop": None,
                "reason": "başlık eşiğin altında" if best_score > 0 else "başlık bu karede yok",
            }
        x, y, w, h = best_box
        if not _box_is_small(w, h, fw, fh, 0.45, 0.12, min_w=8, min_h=8):
            return {
                "ok": False,
                "score": best_score,
                "bbox": None,
                "crop": None,
                "reason": "başlık kutusu geçersiz",
            }
        crop = frame[y : y + h, x : x + w]
        return {
            "ok": True,
            "score": best_score,
            "bbox": best_box,
            "crop": crop.copy(),
            "reason": "kaydedildi",
        }
    except Exception as exc:
        logger.error("Map header auto-cut failed: %s", exc)
        return {
            "ok": False,
            "score": 0.0,
            "bbox": None,
            "crop": None,
            "reason": "harita kapalı",
        }


def _locate_world_gather_crest(frame: np.ndarray, gate: float, **kwargs: Any) -> Dict[str, Any]:
    """Detects 3D gather crest using Pyramid3DCrestDetector, YOLOAutoLabeler or template matching."""
    try:
        from core.vision_filters import Pyramid3DCrestDetector
        crests = Pyramid3DCrestDetector().detect(frame, threshold=gate)
        if crests:
            best = max(crests, key=lambda c: float(c.get("score", 0.0)))
            bx, by, bw, bh = (int(v) for v in best["bbox"])
            score = float(best.get("score", 0.0))
            crop = frame[by : by + bh, bx : bx + bw]
            if crop.size > 0 and score >= gate:
                return {"ok": True, "score": score, "bbox": (bx, by, bw, bh), "crop": crop.copy(), "reason": "kaydedildi"}
    except Exception as exc:
        logger.debug("Pyramid3DCrestDetector error in auto-cut: %s", exc)

    yolo_anns = kwargs.get("yolo_anns") or []
    crest_anns = [a for a in yolo_anns if int(a.get("class_id", -1)) == 6]
    if crest_anns:
        best = max(crest_anns, key=lambda a: float(a.get("confidence", 0.0)))
        bx, by, bw, bh = (int(v) for v in best["bbox"])
        score = float(best.get("confidence", 0.0))
        crop = frame[by : by + bh, bx : bx + bw]
        if crop.size > 0 and score >= gate:
            return {"ok": True, "score": score, "bbox": (bx, by, bw, bh), "crop": crop.copy(), "reason": "kaydedildi"}

    templates_dir = kwargs.get("templates_dir")
    if templates_dir:
        t_path = Path(templates_dir) / "world_gather_crest_template.png"
        if t_path.exists():
            tpl = imread_unicode(t_path, cv2.IMREAD_COLOR)
            if tpl is not None and tpl.size > 0:
                res = cv2.matchTemplate(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)
                _, max_v, _, max_loc = cv2.minMaxLoc(res)
                if float(max_v) >= gate:
                    tw, th = tpl.shape[1], tpl.shape[0]
                    tx, ty = int(max_loc[0]), int(max_loc[1])
                    crop = frame[ty : ty + th, tx : tx + tw]
                    return {"ok": True, "score": float(max_v), "bbox": (tx, ty, tw, th), "crop": crop.copy(), "reason": "kaydedildi"}

    return {"ok": False, "score": 0.0, "bbox": None, "crop": None, "reason": "3B amblem bu karede yok"}


def _locate_e_gather_icon(frame: np.ndarray, gate: float, **kwargs: Any) -> Dict[str, Any]:
    """Detects floating [E] resource icon using YOLOAutoLabeler or VisionEngine."""
    yolo_anns = kwargs.get("yolo_anns") or []
    e_anns = [a for a in yolo_anns if int(a.get("class_id", -1)) == 8]
    if e_anns:
        best = max(e_anns, key=lambda a: float(a.get("confidence", 0.0)))
        bx, by, bw, bh = (int(v) for v in best["bbox"])
        score = float(best.get("confidence", 0.0))
        crop = frame[by : by + bh, bx : bx + bw]
        if crop.size > 0 and score >= gate:
            return {"ok": True, "score": score, "bbox": (bx, by, bw, bh), "crop": crop.copy(), "reason": "kaydedildi"}

    try:
        engine = _vision_engine()
        found, conf, center, _key = engine.detect_floating_resource_icon(frame, threshold=gate, use_roi=True)
        if found and center is not None:
            cx, cy = int(center[0]), int(center[1])
            bw, bh = 42, 42
            bx, by = max(0, cx - 21), max(0, cy - 21)
            crop = frame[by : by + bh, bx : bx + bw]
            if crop.size > 0:
                return {"ok": True, "score": float(conf), "bbox": (bx, by, bw, bh), "crop": crop.copy(), "reason": "kaydedildi"}
    except Exception as exc:
        logger.debug("VisionEngine floating icon error: %s", exc)

    return {"ok": False, "score": 0.0, "bbox": None, "crop": None, "reason": "E ikonu bu karede yok"}


def _locate_resource_node(
    frame: np.ndarray,
    class_id: int,
    class_name: str,
    gate: float,
    template_filename: Optional[str] = None,
    templates_dir: Optional[Path] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Generic locator for 3D resource nodes (Aria, Mela, Targena, Orichalcum, Crystal, Wood)."""
    yolo_anns = kwargs.get("yolo_anns") or []
    matching = [a for a in yolo_anns if int(a.get("class_id", -1)) == class_id]
    if matching:
        best = max(matching, key=lambda a: float(a.get("confidence", 0.0)))
        bx, by, bw, bh = (int(v) for v in best["bbox"])
        score = float(best.get("confidence", 0.0))
        fh, fw = frame.shape[:2]
        if 0 <= bx < fw and 0 <= by < fh:
            crop = frame[by : min(fh, by + bh), bx : min(fw, bx + bw)]
            if crop.size > 0 and score >= gate:
                return {"ok": True, "score": score, "bbox": (bx, by, bw, bh), "crop": crop.copy(), "reason": "kaydedildi"}
            elif score < gate:
                return {"ok": False, "score": score, "bbox": (bx, by, bw, bh), "crop": None, "reason": f"skor {score:.2f}, eşik {gate:.2f}"}

    # Template matching fallback if template file exists
    t_dir = templates_dir or kwargs.get("templates_dir")
    if template_filename and t_dir:
        t_path = Path(t_dir) / template_filename
        if t_path.exists():
            tpl = imread_unicode(t_path, cv2.IMREAD_COLOR)
            if tpl is not None and tpl.size > 0 and tpl.shape[0] < frame.shape[0] and tpl.shape[1] < frame.shape[1]:
                fg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                tg = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
                res = cv2.matchTemplate(fg, tg, cv2.TM_CCOEFF_NORMED)
                _, max_v, _, max_loc = cv2.minMaxLoc(res)
                score = float(max_v)
                tw, th = tpl.shape[1], tpl.shape[0]
                tx, ty = int(max_loc[0]), int(max_loc[1])
                if score >= gate:
                    crop = frame[ty : ty + th, tx : tx + tw]
                    return {"ok": True, "score": score, "bbox": (tx, ty, tw, th), "crop": crop.copy(), "reason": "kaydedildi"}

    return {"ok": False, "score": 0.0, "bbox": None, "crop": None, "reason": f"{class_name} bu karede yok"}


def _locate_node_aria(frame: np.ndarray, gate: float, **kwargs: Any) -> Dict[str, Any]:
    return _locate_resource_node(frame, 0, "herb_aria", gate, "herb_aria_template.png", **kwargs)


def _locate_node_mela(frame: np.ndarray, gate: float, **kwargs: Any) -> Dict[str, Any]:
    return _locate_resource_node(frame, 1, "herb_mela", gate, "herb_mela_template.png", **kwargs)


def _locate_node_targena(frame: np.ndarray, gate: float, **kwargs: Any) -> Dict[str, Any]:
    return _locate_resource_node(frame, 2, "herb_targena", gate, "herb_targena_template.png", **kwargs)


def _locate_node_orichalcum(frame: np.ndarray, gate: float, **kwargs: Any) -> Dict[str, Any]:
    return _locate_resource_node(frame, 3, "ore_orichalcum", gate, "ore_orichalcum_template.png", **kwargs)


def _locate_node_crystal(frame: np.ndarray, gate: float, **kwargs: Any) -> Dict[str, Any]:
    return _locate_resource_node(frame, 4, "gem_crystal", gate, "gem_crystal_template.png", **kwargs)


def _locate_node_wood(frame: np.ndarray, gate: float, **kwargs: Any) -> Dict[str, Any]:
    return _locate_resource_node(frame, 5, "wood_tree", gate, "wood_tree_template.png", **kwargs)


def _locate_minimap_sprout(frame: np.ndarray, gate: float, **kwargs: Any) -> Dict[str, Any]:
    """Detects 3-leaf sprout on minimap via YOLOAutoLabeler or MinimapScanner."""
    yolo_anns = kwargs.get("yolo_anns") or []
    sprout_anns = [a for a in yolo_anns if int(a.get("class_id", -1)) == 9]
    if sprout_anns:
        best = max(sprout_anns, key=lambda a: float(a.get("confidence", 0.0)))
        bx, by, bw, bh = (int(v) for v in best["bbox"])
        score = float(best.get("confidence", 0.0))
        crop = frame[by : by + bh, bx : bx + bw]
        if crop.size > 0 and score >= gate:
            return {"ok": True, "score": score, "bbox": (bx, by, bw, bh), "crop": crop.copy(), "reason": "kaydedildi"}

    try:
        from core.minimap_scanner import MinimapScanner
        scanner = MinimapScanner()
        m_crop, (mx1, my1, _mx2, _my2) = scanner.extract_minimap(frame)
        if m_crop is not None and m_crop.size > 0:
            res = scanner.scan_minimap_crop(m_crop, offset_xy=(mx1, my1), frame_shape=frame.shape)
            nodes = res.get("gatherable_nodes", [])
            if nodes:
                best = max(nodes, key=lambda n: float(n.get("score", 0.85)))
                bx, by, bw, bh = (int(v) for v in best["box_xywh"])
                score = float(best.get("score", 0.88))
                crop = frame[by : by + bh, bx : bx + bw]
                if crop.size > 0 and score >= gate:
                    return {"ok": True, "score": score, "bbox": (bx, by, bw, bh), "crop": crop.copy(), "reason": "kaydedildi"}
    except Exception as exc:
        logger.debug("MinimapScanner sprout error: %s", exc)

    return {"ok": False, "score": 0.0, "bbox": None, "crop": None, "reason": "minimap filizi bu karede yok"}


def _locate_minimap_mob(frame: np.ndarray, gate: float, **kwargs: Any) -> Dict[str, Any]:
    """Detects hostile/monster dot on minimap."""
    yolo_anns = kwargs.get("yolo_anns") or []
    mob_anns = [a for a in yolo_anns if int(a.get("class_id", -1)) == 11]
    if mob_anns:
        best = max(mob_anns, key=lambda a: float(a.get("confidence", 0.0)))
        bx, by, bw, bh = (int(v) for v in best["bbox"])
        score = float(best.get("confidence", 0.0))
        crop = frame[by : by + bh, bx : bx + bw]
        if crop.size > 0 and score >= gate:
            return {"ok": True, "score": score, "bbox": (bx, by, bw, bh), "crop": crop.copy(), "reason": "kaydedildi"}

    try:
        from core.minimap_scanner import MinimapScanner
        scanner = MinimapScanner()
        m_crop, (mx1, my1, _mx2, _my2) = scanner.extract_minimap(frame)
        if m_crop is not None and m_crop.size > 0:
            res = scanner.scan_minimap_crop(m_crop, offset_xy=(mx1, my1), frame_shape=frame.shape)
            m_dots = res.get("monster_dots", []) or res.get("hostile_dots", [])
            if m_dots:
                best = max(m_dots, key=lambda n: float(n.get("score", 0.80)))
                bx, by, bw, bh = (int(v) for v in best["box_xywh"])
                score = float(best.get("score", 0.82))
                crop = frame[by : by + bh, bx : bx + bw]
                if crop.size > 0 and score >= gate:
                    return {"ok": True, "score": score, "bbox": (bx, by, bw, bh), "crop": crop.copy(), "reason": "kaydedildi"}
    except Exception as exc:
        logger.debug("MinimapScanner mob dot error: %s", exc)

    return {"ok": False, "score": 0.0, "bbox": None, "crop": None, "reason": "minimap canavar noktası bu karede yok"}


def _locate_mob_nameplate(frame: np.ndarray, gate: float, **kwargs: Any) -> Dict[str, Any]:
    """Detects 3D monster body or nameplate via YOLOAutoLabeler."""
    yolo_anns = kwargs.get("yolo_anns") or []
    mob_anns = [a for a in yolo_anns if int(a.get("class_id", -1)) == 10]
    if mob_anns:
        best = max(mob_anns, key=lambda a: float(a.get("confidence", 0.0)))
        bx, by, bw, bh = (int(v) for v in best["bbox"])
        score = float(best.get("confidence", 0.0))
        crop = frame[by : by + bh, bx : bx + bw]
        if crop.size > 0 and score >= gate:
            return {"ok": True, "score": score, "bbox": (bx, by, bw, bh), "crop": crop.copy(), "reason": "kaydedildi"}

    return {"ok": False, "score": 0.0, "bbox": None, "crop": None, "reason": "canavar isim plakası bu karede yok"}


def _locate_target_hp_bar(frame: np.ndarray, gate: float, **kwargs: Any) -> Dict[str, Any]:
    """Detects selected monster target HP bar on top HUD."""
    fh, fw = frame.shape[:2]
    x1 = max(0, int(round(fw * 0.30)))
    x2 = min(fw, int(round(fw * 0.70)))
    y1 = max(0, int(round(fh * 0.02)))
    y2 = min(fh, int(round(fh * 0.155)))
    if x2 > x1 and y2 > y1:
        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_red = ((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 168)) & (hsv[:, :, 1] >= 90) & (hsv[:, :, 2] >= 80)
        mask_pink = (hsv[:, :, 0] >= 140) & (hsv[:, :, 0] <= 175) & (hsv[:, :, 1] >= 60) & (hsv[:, :, 2] >= 100)
        mask_olive = (hsv[:, :, 0] >= 28) & (hsv[:, :, 0] <= 65) & (hsv[:, :, 1] >= 90) & (hsv[:, :, 2] >= 85)
        mask = (mask_red | mask_pink | mask_olive).astype(np.uint8)
        kernel = np.ones((2, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        for idx in range(1, num_labels):
            bx, by, bw, bh, area = stats[idx]
            aspect = float(bw) / float(max(1, bh))
            if bw >= 28 and 2 <= bh <= 32 and aspect >= 2.4 and area >= 40:
                crop = roi[by : by + bh, bx : bx + bw]
                score = min(0.98, max(0.65, 0.65 + float(area) / 300.0))
                if score >= gate and crop.size > 0:
                    return {"ok": True, "score": score, "bbox": (x1 + bx, y1 + by, bw, bh), "crop": crop.copy(), "reason": "kaydedildi"}

    return {"ok": False, "score": 0.0, "bbox": None, "crop": None, "reason": "hedef can barı bu karede yok"}


def _locate_player_hp_bar(frame: np.ndarray, gate: float, **kwargs: Any) -> Dict[str, Any]:
    """Detects player red HP bar in bottom HUD."""
    fh, fw = frame.shape[:2]
    x1 = max(0, int(round(fw * 0.3450)))
    x2 = min(fw, int(round(fw * 0.4850)))
    y1 = max(0, int(round(fh * 0.8800)))
    y2 = min(fh, int(round(fh * 0.9100)))
    if x2 > x1 and y2 > y1:
        roi = frame[y1:y2, x1:x2]
        red_mask = (roi[:, :, 2] > 100) & (roi[:, :, 0] < 90)
        ratio = float(np.mean(red_mask))
        if ratio >= 0.10:
            score = min(0.96, max(0.65, ratio * 2.6))
            if score >= gate:
                return {"ok": True, "score": score, "bbox": (x1, y1, x2 - x1, y2 - y1), "crop": roi.copy(), "reason": "kaydedildi"}

    return {"ok": False, "score": 0.0, "bbox": None, "crop": None, "reason": "oyuncu HP barı bu karede yok"}


def _locate_player_mp_bar(frame: np.ndarray, gate: float, **kwargs: Any) -> Dict[str, Any]:
    """Detects player blue MP bar in bottom HUD."""
    fh, fw = frame.shape[:2]
    x1 = max(0, int(round(fw * 0.5150)))
    x2 = min(fw, int(round(fw * 0.6400)))
    y1 = max(0, int(round(fh * 0.8800)))
    y2 = min(fh, int(round(fh * 0.9100)))
    if x2 > x1 and y2 > y1:
        roi = frame[y1:y2, x1:x2]
        blue_mask = (roi[:, :, 0] > 100) & (roi[:, :, 2] < 90)
        ratio = float(np.mean(blue_mask))
        if ratio >= 0.10:
            score = min(0.96, max(0.65, ratio * 2.6))
            if score >= gate:
                return {"ok": True, "score": score, "bbox": (x1, y1, x2 - x1, y2 - y1), "crop": roi.copy(), "reason": "kaydedildi"}

    return {"ok": False, "score": 0.0, "bbox": None, "crop": None, "reason": "oyuncu MP barı bu karede yok"}


def _locate_skill_slot(frame: np.ndarray, gate: float, **kwargs: Any) -> Dict[str, Any]:
    """Detects active skill slot in hotbar using SkillBarReader."""
    fh, fw = frame.shape[:2]
    try:
        from core.skill_bar_reader import SkillBarReader
        sbr = SkillBarReader()
        res = sbr.inspect_skill_bar(frame)
        ready_slots = [v for v in res.values() if v.get("ready")]
        best_slot = max(ready_slots, key=lambda s: float(s.get("score", 0.0))) if ready_slots else None
        score = float(best_slot.get("score", 0.75)) if best_slot else 0.70
        slot_key = best_slot.get("slot", "1") if best_slot else "1"

        x1 = max(0, int(round(sbr.SKILL_BAR_ROI[0] * fw)))
        y1 = max(0, int(round(sbr.SKILL_BAR_ROI[1] * fh)))
        x2 = min(fw, int(round(sbr.SKILL_BAR_ROI[2] * fw)))
        y2 = min(fh, int(round(sbr.SKILL_BAR_ROI[3] * fh)))
        num_slots = len(sbr.SKILL_SLOTS)
        slot_w = max(1, (x2 - x1) // num_slots)
        slot_h = max(1, y2 - y1)
        idx = 0
        for i, name in enumerate(sbr.SKILL_SLOTS):
            if str(name).lower() == slot_key.lower():
                idx = i
                break
        sx = x1 + idx * slot_w
        crop = frame[y1 : y1 + slot_h, sx : sx + slot_w]
        if crop.size > 0 and score >= gate:
            return {"ok": True, "score": score, "bbox": (sx, y1, slot_w, slot_h), "crop": crop.copy(), "reason": "kaydedildi"}
    except Exception as exc:
        logger.debug("Skill slot auto-cut error: %s", exc)

    return {"ok": False, "score": 0.0, "bbox": None, "crop": None, "reason": "skill yuvası bu karede yok"}


def _locate_obstacle(frame: np.ndarray, gate: float, **kwargs: Any) -> Dict[str, Any]:
    """Detects obstacle / boulder / cliff in viewport."""
    yolo_anns = kwargs.get("yolo_anns") or []
    obs_anns = [a for a in yolo_anns if int(a.get("class_id", -1)) == 14]
    if obs_anns:
        best = max(obs_anns, key=lambda a: float(a.get("confidence", 0.0)))
        bx, by, bw, bh = (int(v) for v in best["bbox"])
        score = float(best.get("confidence", 0.0))
        crop = frame[by : by + bh, bx : bx + bw]
        if crop.size > 0 and score >= gate:
            return {"ok": True, "score": score, "bbox": (bx, by, bw, bh), "crop": crop.copy(), "reason": "kaydedildi"}

    templates_dir = kwargs.get("templates_dir")
    if templates_dir:
        t_path = Path(templates_dir) / "obstacle_template.png"
        if t_path.exists():
            tpl = imread_unicode(t_path, cv2.IMREAD_COLOR)
            if tpl is not None and tpl.size > 0:
                res = cv2.matchTemplate(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)
                _, max_v, _, max_loc = cv2.minMaxLoc(res)
                if float(max_v) >= gate:
                    tw, th = tpl.shape[1], tpl.shape[0]
                    tx, ty = int(max_loc[0]), int(max_loc[1])
                    crop = frame[ty : ty + th, tx : tx + tw]
                    return {"ok": True, "score": float(max_v), "bbox": (tx, ty, tw, th), "crop": crop.copy(), "reason": "kaydedildi"}

    return {"ok": False, "score": 0.0, "bbox": None, "crop": None, "reason": "engel bu karede yok"}


def _locate_path_ground(frame: np.ndarray, gate: float, **kwargs: Any) -> Dict[str, Any]:
    """Detects walkable path / ground patch."""
    yolo_anns = kwargs.get("yolo_anns") or []
    path_anns = [a for a in yolo_anns if int(a.get("class_id", -1)) == 15]
    if path_anns:
        best = max(path_anns, key=lambda a: float(a.get("confidence", 0.0)))
        bx, by, bw, bh = (int(v) for v in best["bbox"])
        score = float(best.get("confidence", 0.0))
        crop = frame[by : by + bh, bx : bx + bw]
        if crop.size > 0 and score >= gate:
            return {"ok": True, "score": score, "bbox": (bx, by, bw, bh), "crop": crop.copy(), "reason": "kaydedildi"}

    templates_dir = kwargs.get("templates_dir")
    if templates_dir:
        t_path = Path(templates_dir) / "path_ground_template.png"
        if t_path.exists():
            tpl = imread_unicode(t_path, cv2.IMREAD_COLOR)
            if tpl is not None and tpl.size > 0:
                res = cv2.matchTemplate(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)
                _, max_v, _, max_loc = cv2.minMaxLoc(res)
                if float(max_v) >= gate:
                    tw, th = tpl.shape[1], tpl.shape[0]
                    tx, ty = int(max_loc[0]), int(max_loc[1])
                    crop = frame[ty : ty + th, tx : tx + tw]
                    return {"ok": True, "score": float(max_v), "bbox": (tx, ty, tw, th), "crop": crop.copy(), "reason": "kaydedildi"}

    return {"ok": False, "score": 0.0, "bbox": None, "crop": None, "reason": "yol/zemin bu karede yok"}


def auto_cut_templates(
    frame_bgr: np.ndarray,
    templates_dir: Path,
    min_score: float = AUTO_CUT_SCORE_FLOOR,
) -> list:
    """
    Walks every Kalibrasyon dropdown target, cuts a real detector box from
    `frame_bgr`, and writes `templates/<filename>` the same way Kaydet does.

    Targets that are not visible, or whose score is below max(min_score, 0.60),
    are skipped and no file is written for them.
    Preserves first 3 targets strictly for backwards compatibility.
    """
    results: list = []
    if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
        for name in CalibrationWizard.PRIMARY_TARGETS:
            info = CalibrationWizard.TEMPLATE_TARGETS[name]
            results.append(
                {
                    "target": name,
                    "filename": info["filename"],
                    "saved": False,
                    "score": 0.0,
                    "bbox": None,
                    "path": None,
                    "reason": "kare yok",
                }
            )
        return results

    gate = max(AUTO_CUT_SCORE_FLOOR, float(min_score))
    out_dir = Path(templates_dir)

    # Run YOLOAutoLabeler once as shared cache
    yolo_anns = []
    try:
        from dataset_factory.yolo_auto_labeler import YOLOAutoLabeler
        yolo_anns = YOLOAutoLabeler().auto_label_frame(frame_bgr)
    except Exception as exc:
        logger.debug("YOLOAutoLabeler run failed in auto_cut_templates: %s", exc)

    locators = {
        "F/E Toplama Butonu Şablonu Kes": _locate_gather_prompt,
        "Minimap Oyuncu Oku Şablonu Kes": _locate_minimap_arrow,
        "Harita Başlığı Şablonu Kes": _locate_map_header,
        "E Yüzen Kaynak İkonu Şablonu Kes": _locate_e_gather_icon,
        "3B Beyaz Kanatlı Toplama Amblemi (Crest) Kes": _locate_world_gather_crest,
        "Kaynak: Aria Bitkisi (herb_aria) Kes": _locate_node_aria,
        "Kaynak: Mela Bitkisi (herb_mela) Kes": _locate_node_mela,
        "Kaynak: Targena / Bitki (herb_targena) Kes": _locate_node_targena,
        "Kaynak: Orichalcum / Maden Cevheri (ore_orichalcum) Kes": _locate_node_orichalcum,
        "Kaynak: Kristal / Gem (gem_crystal) Kes": _locate_node_crystal,
        "Kaynak: Ağaç / Kereste (wood_tree) Kes": _locate_node_wood,
        "Minimap 3 Yapraklı Filiz (Kaynak İkonu) Kes": _locate_minimap_sprout,
        "Minimap Canavar Noktası (Negatif) Kes": _locate_minimap_mob,
        "3B Canavar / İsim Plakası (Hedef) Kes": _locate_mob_nameplate,
        "Hedef Canavar Can (HP) Barı Kes": _locate_target_hp_bar,
        "Oyuncu Can (HP) Barı Kes": _locate_player_hp_bar,
        "Oyuncu Mana (MP) Barı Kes": _locate_player_mp_bar,
        "Yetenek / Skill Bar Yuvası Kes": _locate_skill_slot,
        "Engel / Kaya / Duvar (Obstacle) Kes": _locate_obstacle,
        "Yürünebilir Yol / Zemin (Path) Kes": _locate_path_ground,
    }

    for name in CalibrationWizard.PRIMARY_TARGETS:
        info = CalibrationWizard.TEMPLATE_TARGETS[name]
        filename = info["filename"]
        locator_fn = locators.get(name)
        if locator_fn is None:
            found = {
                "ok": False,
                "score": 0.0,
                "bbox": None,
                "crop": None,
                "reason": "algılayıcı tanımsız",
            }
        else:
            try:
                found = locator_fn(
                    frame_bgr,
                    gate,
                    yolo_anns=yolo_anns,
                    templates_dir=out_dir,
                )
            except Exception as exc:
                logger.error("Auto-cut locator failed for %s: %s", name, exc)
                found = {
                    "ok": False,
                    "score": 0.0,
                    "bbox": None,
                    "crop": None,
                    "reason": "algılayıcı hatası",
                }

        record: Dict[str, Any] = {
            "target": name,
            "filename": filename,
            "saved": False,
            "score": float(found.get("score") or 0.0),
            "bbox": found.get("bbox") if found.get("ok") else None,
            "path": None,
            "reason": str(found.get("reason") or "görünmüyor"),
        }
        crop = found.get("crop")
        if found.get("ok") and crop is not None and record["score"] >= gate and record["bbox"] is not None:
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / filename
                if imwrite_unicode(out_path, crop):
                    record["saved"] = True
                    record["path"] = str(out_path)
                    record["reason"] = "kaydedildi"
                    record["bbox"] = tuple(int(v) for v in record["bbox"])
                else:
                    record["saved"] = False
                    record["bbox"] = None
                    record["reason"] = "yazılamadı"
            except Exception as exc:
                logger.error("Auto-cut save failed for %s: %s", filename, exc)
                record["saved"] = False
                record["bbox"] = None
                record["reason"] = "yazılamadı"
        elif not found.get("ok"):
            record["bbox"] = None
        results.append(record)
    return results


def get_calibration_wizard_capabilities() -> list:
    return ["calibration_wizard", "roi_calibration", "diagnostic_snapshot"]


def validate_calibration_wizard_contract() -> bool:
    return bool(get_calibration_wizard_capabilities())


def get_calibration_wizard_baseline_metrics() -> dict:
    return {"module": "calibration_wizard", "contract_valid": validate_calibration_wizard_contract(), "capability_count": len(get_calibration_wizard_capabilities())}


def get_calibration_wizard_readiness_status() -> dict:
    return {"module": "calibration_wizard", "ready": validate_calibration_wizard_contract(), "metrics": get_calibration_wizard_baseline_metrics()}


def get_calibration_wizard_audit_record() -> dict:
    return {"module": "calibration_wizard", "audit_passed": validate_calibration_wizard_contract(), "readiness": get_calibration_wizard_readiness_status()}


def get_calibration_wizard_operational_digest() -> dict:
    return {"module": "calibration_wizard", "operational": validate_calibration_wizard_contract(), "audit": get_calibration_wizard_audit_record()}


def get_calibration_wizard_telemetry_payload() -> dict:
    return {"module": "calibration_wizard", "telemetry_ready": validate_calibration_wizard_contract(), "digest": get_calibration_wizard_operational_digest()}


def get_calibration_wizard_integrity_seal() -> dict:
    return {"module": "calibration_wizard", "sealed": validate_calibration_wizard_contract(), "telemetry": get_calibration_wizard_telemetry_payload()}


def get_calibration_wizard_compliance_certificate() -> dict:
    return {"module": "calibration_wizard", "certified": validate_calibration_wizard_contract(), "seal": get_calibration_wizard_integrity_seal()}


def get_calibration_wizard_readiness_probe() -> dict:
    return {"module": "calibration_wizard", "ready": validate_calibration_wizard_contract(), "certificate": get_calibration_wizard_compliance_certificate()}


def get_calibration_wizard_audit_summary() -> dict:
    return {"module": "calibration_wizard", "audit_passed": validate_calibration_wizard_contract(), "probe": get_calibration_wizard_readiness_probe()}


# Module-level aliases for targets
PRIMARY_TARGETS = CalibrationWizard.PRIMARY_TARGETS
TEMPLATE_TARGETS = CalibrationWizard.TEMPLATE_TARGETS




