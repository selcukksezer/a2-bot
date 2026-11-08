# -*- coding: utf-8 -*-
"""Pixel-perfect YOLO Fabrikası page: embeds the HTML prototype via QWebEngine.

Shell rail stays in app_shell. This page hides the prototype's fake sidebar/titlebar
and bridges dataset/train/freeze/review actions into the existing Python backend.

Default: Qt YOLOFactoryDialog (real QPixmap). Opt-in WebEngine HTML via
GATHERION_YOLO_WEB=1 — review/capture images pushed as JPEG data-URLs (file://
blocked by Chromium + prior CSS !important). Falls back to Qt on failure.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import QObject, Qt, QTimer, QUrl, pyqtSlot
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget

logger = logging.getLogger("Gatherion.YoloPipelineWeb")

PROTOTYPE_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / "prototypes" / "yolo-fabrikasi.html"
)

# Tiny transparent GIF — strip multi-MB prototype embeds so Chromium can load the UI shell.
_TINY_GIF = (
    "data:image/gif;base64,"
    "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
)
_DATA_IMAGE_RE = re.compile(
    r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+",
    re.IGNORECASE,
)

_SHELL_INJECT_CSS = """
html, body {
  background: #0E1013 !important;
  overflow: auto !important;
}
.stage {
  max-width: none !important;
  margin: 0 !important;
  padding: 0 !important;
}
.window {
  border: none !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  background: #0E1013 !important;
}
.titlebar, .sidebar, .stage > p { display: none !important; }
.app {
  display: block !important;
  grid-template-columns: none !important;
  min-height: 100vh !important;
}
.main {
  padding: 16px 18px 24px !important;
  min-height: 100vh !important;
  background: #0E1013 !important;
}
/* Hide prototype fake sidebar art; DO NOT force background-image:none on scenes —
   that !important blocked real data-URL / file images (black review preview). */
#scene, #reviewScene {
  background-color: #0B0E12;
  background-size: cover !important;
  background-position: center !important;
  background-repeat: no-repeat !important;
}
#scene .bbox, #reviewScene .bbox, #scene .minimap, #reviewScene .minimap { display: none !important; }
/* Empty grey thumb column gone — text-only queue (lazy thumbs optional later) */
.queue-item::before { display: none !important; content: none !important; }
.queue-item {
  padding-left: 10px !important;
  min-height: 44px !important;
}
.err-card .thumb-a, .err-card .thumb-b { background-image: none !important; }
"""

_BRIDGE_JS = r"""
(function () {
  if (window.__gatherionBridgeWired) return;
  window.__gatherionBridgeWired = true;

  function $(id) { return document.getElementById(id); }

  function setText(id, text) {
    var el = $(id);
    if (el) el.textContent = text == null ? '' : String(text);
  }

  function setHtml(id, html) {
    var el = $(id);
    if (el) el.innerHTML = html == null ? '' : String(html);
  }

  function purgeDemoContent() {
    // Kill hardcoded prototype demo numbers / fake boxes / fake runs / fake log
    ['statFrames','statTrain','statVal2','statPending','kvFrames','splitTrain','splitValCount',
     'freezeState','flowSummary','stReview','nextTitle','nextText','reviewProgTxt','previewMeta',
     'reviewTitle','reviewSub','sessionVal','bufferVal','epochLabel','runLabel','epochPct'
    ].forEach(function (id) {
      var el = $(id);
      if (el) el.textContent = el.id === 'previewMeta' || el.id === 'reviewTitle' ? 'Canlı kare yok' : '0';
    });
    setText('statPending', '0 kare');
    setText('previewMeta', 'Canlı kare yok');
    setText('reviewTitle', 'Kuyruk boş');
    setText('reviewSub', 'Onay bekleyen kare yok');
    setText('nextTitle', 'Veri yok');
    setText('nextText', 'Diskten gerçek kare/etiket gelince burası dolar.');
    setText('bulkVal', '≥ 0,90');
    var bulk = $('btnBulk');
    if (bulk) bulk.textContent = 'Toplu onaylanacak kare yok';
    var rp = $('reviewProg');
    if (rp) rp.style.width = '0%';

    document.querySelectorAll('#scene .bbox, #reviewScene .bbox').forEach(function (n) { n.remove(); });
    document.querySelectorAll('#scene .minimap, #reviewScene .minimap').forEach(function (n) { n.remove(); });

    ['queueList','detList','runList','logBox','healthList'].forEach(function (id) {
      var el = $(id);
      if (!el) return;
      if (id === 'healthList') return; // rebuilt from real empty/imbalance checks later
      el.innerHTML = '';
    });
    var log = $('logBox');
    if (log) log.innerHTML = '<div><span class="t">[--:--:--]</span> <span class="info">[BİLGİ]</span> Gerçek log bekleniyor…</div>';
    var runs = $('runList');
    if (runs) runs.innerHTML = '<div class="run-row"><div><b>Çalıştırma yok</b><small>models/*.pt bulunamadı</small></div><span class="map">—</span></div>';

    // Strip demo strip cells that lack dedicated ids (empty-class / last-model)
    document.querySelectorAll('.strip .stat').forEach(function (stat, idx) {
      if (idx === 2) {
        var b = stat.querySelector('b');
        var s = stat.querySelector('small');
        if (b) b.textContent = '0 / 16';
        if (s) s.textContent = 'Boş sınıf yok';
      }
      if (idx === 3) {
        var b2 = stat.querySelector('b');
        var s2 = stat.querySelector('small');
        if (b2) b2.textContent = '—';
        if (s2) s2.textContent = 'Yayınlanmış model yok';
      }
      if (idx === 0) {
        var sm = stat.querySelector('b small');
        if (sm) sm.textContent = '/ 100.000';
        var sub = stat.querySelectorAll('small');
        if (sub && sub.length > 1) sub[sub.length - 1].textContent = 'Train 0 · Val 0 · 0 kutu';
      }
      if (idx === 1) {
        var sub1 = stat.querySelector('small');
        if (sub1) sub1.textContent = 'Kuyruk boş';
      }
    });

    ensureEmptyLiveScene();
    ensureEmptyReviewScene();
  }

  function ensureEmptyLiveScene() {
    var scene = $('scene');
    if (!scene) return;
    scene.style.backgroundImage = 'none';
    scene.style.background = '#0B0E12';
    var empty = scene.querySelector('.gatherion-empty');
    if (!empty) {
      empty = document.createElement('div');
      empty.className = 'gatherion-empty';
      empty.style.cssText = 'position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#6d8093;font-size:14px;z-index:5;pointer-events:none;';
      empty.textContent = 'Canlı kare yok';
      scene.appendChild(empty);
    }
  }

  function ensureEmptyReviewScene() {
    var scene = $('reviewScene');
    if (!scene) return;
    scene.style.backgroundImage = 'none';
    scene.style.background = '#0B0E12';
    var empty = scene.querySelector('.gatherion-empty');
    if (!empty) {
      empty = document.createElement('div');
      empty.className = 'gatherion-empty';
      empty.style.cssText = 'position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#6d8093;font-size:14px;z-index:5;pointer-events:none;';
      empty.textContent = 'Kuyruk boş';
      scene.appendChild(empty);
    }
  }

  function paintBoxes(scene, boxes) {
    if (!scene) return;
    scene.querySelectorAll('.bbox').forEach(function (n) { n.remove(); });
    (boxes || []).forEach(function (bx) {
      var el = document.createElement('div');
      var conf = Number(bx.conf || 0);
      el.className = 'bbox ' + (conf >= 0.9 ? 'green' : conf >= 0.75 ? 'amber' : 'red');
      var left = Math.max(0, (Number(bx.cx) - Number(bx.w) / 2) * 100);
      var top = Math.max(0, (Number(bx.cy) - Number(bx.h) / 2) * 100);
      el.style.left = left + '%';
      el.style.top = top + '%';
      el.style.width = (Number(bx.w) * 100) + '%';
      el.style.height = (Number(bx.h) * 100) + '%';
      var lab = document.createElement('label');
      lab.textContent = (bx.class_name || '?') + ' · ' + conf.toFixed(2);
      el.appendChild(lab);
      scene.appendChild(el);
    });
  }

  function applyLive(st) {
    var scene = $('scene');
    if (!scene) return;
    var empty = scene.querySelector('.gatherion-empty');
    if (st.liveJpegB64) {
      scene.style.setProperty('background-image', 'url(data:image/jpeg;base64,' + st.liveJpegB64 + ')', 'important');
      scene.style.setProperty('background-size', 'cover', 'important');
      scene.style.setProperty('background-position', 'center', 'important');
      if (empty) empty.style.display = 'none';
      paintBoxes(scene, st.liveBoxes || []);
    } else {
      scene.style.setProperty('background-image', 'none', 'important');
      scene.style.backgroundColor = '#0B0E12';
      scene.querySelectorAll('.bbox').forEach(function (n) { n.remove(); });
      if (empty) {
        empty.style.display = 'flex';
        empty.textContent = st.liveBadge || 'Canlı kare yok';
      }
    }
    setText('previewMeta', st.previewMeta || 'Canlı kare yok');
  }

  function applyReview(st) {
    var ql = $('queueList');
    var selIdx = (st.reviewIndex != null) ? Number(st.reviewIndex) : 0;
    if (ql) {
      ql.innerHTML = '';
      (st.queue || []).forEach(function (it, idx) {
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'queue-item' + (idx === selIdx && !it.done ? ' sel' : '') + (it.done ? ' done' : '');
        btn.setAttribute('data-qidx', String(idx));
        var short = String(it.stem || '');
        if (short.length > 28) short = short.slice(0, 12) + '…' + short.slice(-12);
        btn.innerHTML = '<b>#' + short + '</b><small>' + it.split + ' · ' + it.n_weak +
          ' zayıf · ' + Number(it.conf || 0).toFixed(2) + '</small>';
        btn.addEventListener('click', function (ev) {
          ev.preventDefault();
          if (window.bridge && typeof window.bridge.selectReview === 'function') {
            window.bridge.selectReview(idx);
          }
        });
        ql.appendChild(btn);
      });
      if (!(st.queue || []).length) {
        ql.innerHTML = '<div class="queue-item" style="opacity:.7"><b>Kuyruk boş</b><small>dataset/review/**/*.json yok</small></div>';
      }
    }

    var scene = $('reviewScene');
    var cur = st.currentReview;
    var empty = scene && scene.querySelector('.gatherion-empty');
    var dataUrl = '';
    if (cur && cur.jpegB64) {
      dataUrl = 'url(data:image/jpeg;base64,' + cur.jpegB64 + ')';
    } else if (cur && cur.image_url) {
      dataUrl = 'url("' + cur.image_url + '")';
    }
    if (cur && dataUrl && scene) {
      scene.style.setProperty('background-image', dataUrl, 'important');
      scene.style.setProperty('background-size', 'cover', 'important');
      scene.style.setProperty('background-position', 'center', 'important');
      if (empty) empty.style.display = 'none';
      paintBoxes(scene, cur.weak || []);
      setText('reviewTitle', cur.title || ('Kare #' + cur.stem));
      setText('reviewSub', cur.sub || '');
      var det = $('detList');
      if (det) {
        det.innerHTML = '';
        (cur.weak || []).forEach(function (bx) {
          var row = document.createElement('div');
          row.className = 'det-row';
          row.textContent = (bx.class_name || '?') + ' · ' + Number(bx.conf || 0).toFixed(2);
          det.appendChild(row);
        });
      }
    } else if (scene) {
      scene.style.setProperty('background-image', 'none', 'important');
      scene.style.backgroundColor = '#0B0E12';
      scene.querySelectorAll('.bbox').forEach(function (n) { n.remove(); });
      if (empty) {
        empty.style.display = 'flex';
        empty.textContent = (cur && cur.error)
          ? String(cur.error)
          : ((st.queue || []).length ? 'Görüntü yüklenemedi' : 'Kuyruk boş');
      }
      if (cur && cur.title) setText('reviewTitle', cur.title);
      else setText('reviewTitle', (st.queue || []).length ? 'Görüntü yok' : 'Kuyruk boş');
      setText('reviewSub', (cur && cur.sub) || (cur && cur.error) || 'Onay bekleyen kare yok');
      setHtml('detList', '');
    }
  }

  function applyRunsAndLog(st) {
    var runs = $('runList');
    if (runs) {
      runs.innerHTML = '';
      (st.runs || []).forEach(function (r) {
        var row = document.createElement('div');
        row.className = 'run-row';
        row.innerHTML = '<div><b>' + r.name + '</b><small>' + (r.meta || '') + '</small></div>' +
          '<span class="map">' + (r.map || '—') + '</span>' +
          (r.live ? '<span class="best-tag">CANLI</span>' : '<span></span>');
        runs.appendChild(row);
      });
      if (!(st.runs || []).length) {
        runs.innerHTML = '<div class="run-row"><div><b>Çalıştırma yok</b><small>models/*.pt yok</small></div><span class="map">—</span></div>';
      }
    }
    var log = $('logBox');
    if (log && st.logText != null) {
      var lines = String(st.logText).split(/\r?\n/).filter(Boolean).slice(-40);
      log.innerHTML = lines.map(function (ln) {
        return '<div>' + ln.replace(/</g, '&lt;') + '</div>';
      }).join('') || '<div><span class="info">[BİLGİ]</span> Log boş</div>';
      log.scrollTop = log.scrollHeight;
    }
  }

  function wire() {
    var b = window.bridge;
    if (!b) return;

    function bind(id, fn) {
      var el = $(id);
      if (!el) return;
      el.addEventListener('click', function (ev) {
        ev.preventDefault();
        ev.stopPropagation();
        try { fn(); } catch (e) { console.error(e); }
      }, true);
    }

    bind('btnFolder', function () { b.openDataset(); });
    bind('btnRefresh', function () { b.refresh(); });
    bind('btnFreeze', function () { b.freeze(); });
    bind('btnTrain', function () { b.train(); });
    bind('btnStopTrain', function () { b.stopTrain(); });
    bind('btnBootstrap', function () { b.bootstrap(); });
    bind('btnScan', function () { b.bootstrap(); });
    bind('btnValidate', function () { b.validate(); });
    bind('btnValDone', function () { b.finishValidate(); });
    bind('btnPublish', function () { b.publish(); });
    bind('btnRollback', function () { b.rollback(); });
    bind('btnSendReview', function () { b.finishCapture(); });
    bind('btnApprove', function () { b.reviewApprove(); });
    bind('btnReject', function () { b.reviewReject(); });
    bind('btnSkip', function () { b.reviewSkip(); });
    bind('btnBulk', function () { b.reviewBulk(); });
    bind('btnClearLog', function () { b.clearLog(); });
    bind('btnCaptureToggle', function () { b.toggleCapture(); });

    document.querySelectorAll('.flow-step[data-step]').forEach(function (el) {
      el.addEventListener('click', function (ev) {
        ev.preventDefault();
        var n = parseInt(el.getAttribute('data-step'), 10);
        if (n) b.goStep(n);
      }, true);
    });
    document.querySelectorAll('[data-goto]').forEach(function (el) {
      el.addEventListener('click', function (ev) {
        ev.preventDefault();
        var n = parseInt(el.getAttribute('data-goto'), 10);
        if (n) b.goStep(n);
      }, true);
    });

    if (typeof b.refresh === 'function') b.refresh();
  }

  function bootChannel() {
    if (typeof QWebChannel === 'undefined') {
      var s = document.createElement('script');
      s.src = 'qrc:///qtwebchannel/qwebchannel.js';
      s.onload = function () {
        new QWebChannel(qt.webChannelTransport, function (ch) {
          window.bridge = ch.objects.bridge;
          wire();
        });
      };
      document.head.appendChild(s);
      return;
    }
    new QWebChannel(qt.webChannelTransport, function (ch) {
      window.bridge = ch.objects.bridge;
      wire();
    });
  }

  window.gatherionApplyState = function (raw) {
    var st = (typeof raw === 'string') ? JSON.parse(raw) : raw;
    if (!st) return;
    setText('statFrames', st.statFrames);
    setText('statTrain', st.statTrain);
    setText('statVal2', st.statVal2);
    setText('statPending', st.statPending);
    setText('kvFrames', st.kvFrames);
    setText('splitTrain', st.splitTrain);
    setText('splitValCount', st.splitValCount);
    setText('freezeState', st.freezeState);
    setText('flowSummary', st.flowSummary);
    setText('stReview', st.stReview);
    setText('nextTitle', st.nextTitle);
    setText('nextText', st.nextText);
    setText('reviewProgTxt', st.reviewProgTxt);
    setText('sessionVal', st.sessionVal);
    var rp = $('reviewProg');
    if (rp && st.reviewProgPct != null) rp.style.width = st.reviewProgPct + '%';
    var bulk = $('btnBulk');
    if (bulk && st.bulkLabel) bulk.textContent = st.bulkLabel;

    document.querySelectorAll('.strip .stat').forEach(function (stat, idx) {
      if (idx === 0) {
        var sub = stat.querySelectorAll('small');
        if (sub && sub.length > 1) {
          sub[sub.length - 1].textContent =
            'Train ' + (st.statTrain || '0') + ' · Val ' + (st.statVal2 || '0') +
            ' · ' + (st.statBoxes || '0') + ' kutu';
        }
      }
      if (idx === 1) {
        var s1 = stat.querySelector('small');
        if (s1) s1.textContent = st.statPendingSub || 'Kuyruk boş';
      }
      if (idx === 2) {
        var b = stat.querySelector('b');
        var s = stat.querySelector('small');
        if (b) b.textContent = st.statEmpty || '0 / 16';
        if (s) s.textContent = st.statEmptySub || 'Boş sınıf yok';
      }
      if (idx === 3) {
        var b2 = stat.querySelector('b');
        var s2 = stat.querySelector('small');
        if (b2) b2.textContent = st.statModel || '—';
        if (s2) s2.textContent = st.statModelSub || '';
      }
    });

    applyLive(st);
    applyReview(st);
    applyRunsAndLog(st);

    if (typeof st.step === 'number' && typeof goStep === 'function') {
      try { goStep(st.step); } catch (e) {}
    }
  };

  purgeDemoContent();
  bootChannel();
})();
"""


def _slim_prototype_path() -> Path:
    """Cached copy of the prototype with huge base64 images replaced (layout/CSS intact)."""
    raw = PROTOTYPE_PATH.read_bytes()
    digest = hashlib.sha1(raw).hexdigest()[:12]
    out = Path(tempfile.gettempdir()) / f"gatherion-yolo-proto-{digest}.html"
    if out.is_file() and out.stat().st_size > 1024:
        return out
    text = raw.decode("utf-8", errors="replace")
    slim = _DATA_IMAGE_RE.sub(_TINY_GIF, text)
    out.write_text(slim, encoding="utf-8")
    logger.info(
        "YOLO prototype slimmed %s → %s bytes (%s)",
        len(raw),
        out.stat().st_size,
        out.name,
    )
    return out


def webengine_allowed() -> bool:
    """Opt-in WebEngine (GATHERION_YOLO_WEB=1). Default Qt — crash-safe, real pixmaps."""
    flag = str(os.environ.get("GATHERION_YOLO_WEB", "0")).strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return False
    platform = str(os.environ.get("QT_QPA_PLATFORM", "")).strip().lower()
    if platform in ("offscreen", "minimal", "vnc", "null"):
        return False
    if QApplication.instance() is None:
        return False
    if not PROTOTYPE_PATH.is_file():
        return False
    return True


def create_yolo_factory_page(
    factory_controller: Optional[Any] = None,
    parent: Optional[QWidget] = None,
) -> QWidget:
    """Shell entry: Qt by default; WebEngine only if GATHERION_YOLO_WEB=1."""
    from ui.yolo_factory_dialog import YOLOFactoryDialog

    if not webengine_allowed():
        logger.info("YOLO page using Qt YOLOFactoryDialog (set GATHERION_YOLO_WEB=1 for HTML)")
        return YOLOFactoryDialog(factory_controller=factory_controller, parent=parent)

    try:
        return YoloPipelineWebPage(
            factory_controller=factory_controller,
            parent=parent,
        )
    except Exception as exc:
        logger.error("YOLO WebEngine page failed, falling back to Qt UI: %s", exc)
        return YOLOFactoryDialog(factory_controller=factory_controller, parent=parent)


class _YoloWebBridge(QObject):
    """JS → Python actions for the embedded prototype."""

    def __init__(self, host: "YoloPipelineWebPage") -> None:
        super().__init__(host)
        self._host = host

    def _be(self) -> Any:
        return self._host.backend

    @pyqtSlot()
    def openDataset(self) -> None:
        self._be().on_open_dataset_folder_clicked()

    @pyqtSlot()
    def refresh(self) -> None:
        self._host.push_state_to_web()

    @pyqtSlot()
    def freeze(self) -> None:
        self._be()._freeze_dataset()
        self._host.push_state_to_web()

    @pyqtSlot()
    def train(self) -> None:
        self._be().on_train_rtx_clicked()

    @pyqtSlot()
    def stopTrain(self) -> None:
        fn = getattr(self._be(), "_stop_training", None)
        if callable(fn):
            fn()

    @pyqtSlot()
    def bootstrap(self) -> None:
        self._be().on_bootstrap_clicked()

    @pyqtSlot()
    def validate(self) -> None:
        fn = getattr(self._be(), "_run_validate", None)
        if callable(fn):
            fn()
        self._host.push_state_to_web()

    @pyqtSlot()
    def finishValidate(self) -> None:
        fn = getattr(self._be(), "_finish_validate", None)
        if callable(fn):
            fn()
        self._host.push_state_to_web()

    @pyqtSlot()
    def publish(self) -> None:
        fn = getattr(self._be(), "_publish_model", None)
        if callable(fn):
            fn()
        self._host.push_state_to_web()

    @pyqtSlot()
    def rollback(self) -> None:
        fn = getattr(self._be(), "_rollback_model", None)
        if callable(fn):
            fn()
        self._host.push_state_to_web()

    @pyqtSlot()
    def finishCapture(self) -> None:
        fn = getattr(self._be(), "_finish_capture_to_review", None)
        if callable(fn):
            fn()
        self._host.push_state_to_web()

    @pyqtSlot()
    def reviewApprove(self) -> None:
        self._be()._review_done("ok")
        self._host.push_state_to_web()

    @pyqtSlot()
    def reviewReject(self) -> None:
        self._be()._review_done("no")
        self._host.push_state_to_web()

    @pyqtSlot()
    def reviewSkip(self) -> None:
        self._be()._review_skip()
        self._host.push_state_to_web()

    @pyqtSlot()
    def reviewBulk(self) -> None:
        self._be()._bulk_approve()
        self._host.push_state_to_web()

    @pyqtSlot(int)
    def selectReview(self, index: int) -> None:
        be = self._be()
        items = getattr(be, "_review_items", None) or []
        if not items:
            return
        idx = max(0, min(int(index), len(items) - 1))
        be._review_index = idx
        fn = getattr(be, "_render_review_frame", None)
        if callable(fn):
            fn()
        self._host.push_state_to_web()

    @pyqtSlot()
    def clearLog(self) -> None:
        fn = getattr(self._be(), "_clear_log", None)
        if callable(fn):
            fn()

    @pyqtSlot()
    def toggleCapture(self) -> None:
        fn = getattr(self._be(), "_on_capture_toggle", None)
        if callable(fn):
            fn()
        self._host.push_state_to_web()

    @pyqtSlot(int)
    def goStep(self, step: int) -> None:
        self._be()._go_step(int(step))
        self._host.push_state_to_web()


class YoloPipelineWebPage(QWidget):
    """Shell tab content: full-bleed prototype HTML + Python bridge."""

    def __init__(
        self,
        factory_controller: Optional[Any] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("YoloPipelineWebPage")
        self.setStyleSheet("QWidget#YoloPipelineWebPage { background:#0E1013; }")
        self._factory_controller = factory_controller

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._web_ok = False
        self.view: Optional[Any] = None
        self._bridge: Optional[_YoloWebBridge] = None
        self._channel: Optional[Any] = None
        self._sync_timer: Optional[QTimer] = None
        self._backend: Optional[Any] = None

        if self._should_use_webengine():
            try:
                self._mount_webengine(layout)
                return
            except Exception as exc:
                logger.warning("WebEngine mount failed (%s) — Qt fallback", exc)

        self._mount_qt_fallback(layout)

    @staticmethod
    def _should_use_webengine() -> bool:
        return webengine_allowed()

    def _mount_qt_fallback(self, layout: QVBoxLayout) -> None:
        layout.addWidget(self.backend, stretch=1)
        self.backend.show()
        logger.info("YOLO page using Qt FactoryDialog fallback")

    def _mount_webengine(self, layout: QVBoxLayout) -> None:
        from PyQt6.QtWebChannel import QWebChannel
        from PyQt6.QtWebEngineCore import QWebEngineSettings
        from PyQt6.QtWebEngineWidgets import QWebEngineView

        os.environ.setdefault(
            "QTWEBENGINE_CHROMIUM_FLAGS",
            "--disable-gpu --disable-software-rasterizer --no-sandbox",
        )

        self.view = QWebEngineView(self)
        self.view.setStyleSheet("background:#0E1013;")
        page = self.view.page()
        page.setBackgroundColor(QColor("#0E1013"))
        settings = page.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)

        self._bridge = _YoloWebBridge(self)
        self._channel = QWebChannel(page)
        self._channel.registerObject("bridge", self._bridge)
        page.setWebChannel(self._channel)

        layout.addWidget(self.view, stretch=1)
        self._web_ok = True

        self.view.loadFinished.connect(self._on_load_finished)
        try:
            proto = str(_slim_prototype_path().resolve())
        except Exception as exc:
            logger.warning("Slim prototype failed (%s); using original", exc)
            proto = str(PROTOTYPE_PATH.resolve())
        QTimer.singleShot(0, lambda: self.view and self.view.load(QUrl.fromLocalFile(proto)))

        self._sync_timer = QTimer(self)
        self._sync_timer.setInterval(2500)
        self._sync_timer.timeout.connect(self.push_state_to_web)
        self._sync_timer.start()
        logger.info("YOLO page using WebEngine prototype")

    @property
    def backend(self) -> Any:
        if self._backend is None:
            from ui.yolo_factory_dialog import YOLOFactoryDialog

            self._backend = YOLOFactoryDialog(
                factory_controller=self._factory_controller,
                parent=None,
            )
            self._backend.setWindowFlags(Qt.WindowType.Tool)
            self._backend.hide()
            if hasattr(self._backend, "_refresh_timer"):
                self._backend._refresh_timer.stop()
        return self._backend

    def _on_load_finished(self, ok: bool) -> None:
        if not ok or self.view is None:
            logger.error("Failed to load YOLO prototype HTML")
            return
        css_json = json.dumps(_SHELL_INJECT_CSS)
        inject = (
            "(function(){"
            f"var css={css_json};"
            "var s=document.createElement('style');"
            "s.id='gatherion-shell-fit';"
            "s.textContent=css;"
            "document.head.appendChild(s);"
            "})();"
        )
        self.view.page().runJavaScript(inject)
        self.view.page().runJavaScript(_BRIDGE_JS)
        QTimer.singleShot(200, self.push_state_to_web)

    def push_state_to_web(self) -> None:
        if not self._web_ok or self.view is None:
            return
        try:
            payload = self.backend.export_web_state()
        except Exception as exc:
            logger.debug("export_web_state failed: %s", exc)
            payload = {
                "statFrames": "0",
                "statTrain": "0",
                "statVal2": "0",
                "statBoxes": "0",
                "statPending": "0 kare",
                "statPendingSub": "Kuyruk boş",
                "liveJpegB64": "",
                "liveBadge": "Canlı kare yok",
                "previewMeta": "Canlı kare yok",
                "liveBoxes": [],
                "queue": [],
                "currentReview": None,
                "runs": [],
                "logText": "",
            }
        try:
            payload.setdefault("extended", get_web_pipeline_extended_state())
        except Exception:
            pass
        raw = json.dumps(payload, ensure_ascii=False)
        self.view.page().runJavaScript(
            f"if (window.gatherionApplyState) window.gatherionApplyState({raw});"
        )

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        if self._sync_timer is not None and self._sync_timer.isActive():
            self._sync_timer.stop()
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception:
                pass
        super().closeEvent(event)


CANONICAL_25_CLASSES: dict[int, str] = {
    0: "player_self",
    1: "player_other",
    2: "monster_hostile",
    3: "monster_passive",
    4: "monster_boss",
    5: "npc_vendor",
    6: "npc_quest",
    7: "resource_herb",
    8: "resource_ore",
    9: "resource_wood",
    10: "resource_fish_spot",
    11: "loot_bag",
    12: "loot_chest",
    13: "portal_gate",
    14: "ui_health_bar",
    15: "ui_mana_bar",
    16: "ui_minimap",
    17: "ui_dialog_box",
    18: "ui_captcha_prompt",
    19: "ui_inventory_slot",
    20: "obstacle_rock",
    21: "mount_entity",
    22: "hp_potion",
    23: "mp_potion",
    24: "buff_scroll",
}

CONFIDENCE_COLOR_TIERS: dict[str, dict[str, Any]] = {
    "high": {"min": 0.85, "max": 1.0, "color": "#22c55e", "label": "Yuksek Guven"},
    "medium": {"min": 0.60, "max": 0.85, "color": "#fb923c", "label": "Orta Guven"},
    "low": {"min": 0.0, "max": 0.60, "color": "#ef4444", "label": "Dusuk Guven"},
}


def confidence_to_hex_color(conf: float) -> str:
    """Return 3-tier confidence hex color (#22c55e, #fb923c, #ef4444)."""
    val = float(conf or 0.0)
    if val >= 0.85:
        return "#22c55e"
    if val >= 0.60:
        return "#fb923c"
    return "#ef4444"


class ApprovedDatasetPool:
    """Lightweight web-bridge view over the approved dataset pool directory."""

    def __init__(self, dataset_dir: Optional[str] = None, only_approved: bool = True) -> None:
        base = Path(dataset_dir) if dataset_dir else Path("dataset")
        self.dataset_dir = base
        self.approved_dir = base / "approved"
        self.pending_dir = base / "review"
        self.rejected_dir = base / "rejected"
        self.only_approved = bool(only_approved)

    def _count_images(self, folder: Path) -> int:
        if not folder.is_dir():
            return 0
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        try:
            return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts)
        except Exception:
            return 0

    def summary(self) -> dict[str, Any]:
        return {
            "pool": "ApprovedDatasetPool",
            "dataset_dir": str(self.dataset_dir),
            "approved_dir": str(self.approved_dir),
            "pending_dir": str(self.pending_dir),
            "rejected_dir": str(self.rejected_dir),
            "approved_count": self._count_images(self.approved_dir),
            "pending_count": self._count_images(self.pending_dir),
            "rejected_count": self._count_images(self.rejected_dir),
            "only_approved": self.only_approved,
            "only_approved_label": "Sadece Onaylananlarla Eğit",
        }


def get_web_pipeline_extended_state(dataset_dir: Optional[str] = None) -> dict:
    """Return extended web pipeline state for ApprovedDatasetPool, 3-tier confidence colors,
    only_approved training mode, and the 25 canonical classes (0..24).
    """
    pool = ApprovedDatasetPool(dataset_dir=dataset_dir, only_approved=True)
    pool_summary = pool.summary()
    class_list = [CANONICAL_25_CLASSES[i] for i in range(25)]
    return {
        "pool_name": "ApprovedDatasetPool",
        "approved_dataset_pool": pool_summary,
        "only_approved": True,
        "only_approved_label": "Sadece Onaylananlarla Eğit",
        "confidence_colors": ["#22c55e", "#fb923c", "#ef4444"],
        "confidence_tiers": CONFIDENCE_COLOR_TIERS,
        "num_classes": 25,
        "canonical_class_count": 25,
        "class_ids": list(range(25)),
        "canonical_classes": dict(CANONICAL_25_CLASSES),
        "class_names": class_list,
    }

