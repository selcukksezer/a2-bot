# YOLO Fabrikası God-File Bölme Hazırlığı (P3-29)

> **Bölme bugün değil — önce veri kaybı dursun.**
> Bu doküman SADECE hazırlıktır: kod bölünmedi, fonksiyon taşınmadı, davranış
> değişmedi. Amaç, `ui/yolo_factory_dialog.py` bölüneceği gün geldiğinde
> işin körü körüne değil, ölçülmüş sınırlar ve bilinen risklerle yapılmasıdır.

- **Kaynak rapor:** `~/workspace/your_files/yolo-sekmesi-ultra-tarama-2026-11-10.md`,
  bölüm 9 madde 29 ("God-file bölme hazırlığı").
- **Taban:** `feat/all-combined-v16` @ `3e739f7`.
- **Dal:** `fix/p3-29-god-file-hazirlik` (bu dokümanın commitlendiği dal).
- **Ölçüm yöntemi:** AST çözümleme (`ast` modülü: sınıf/metot satır aralıkları)
  + `grep` sayımları. Test koşulmadı (kural gereği kodlama ajanı test koşmaz);
  `py_compile` ile sözdizimi doğrulandı.

## 0. Satır sayısı düzeltmesi (rapor vs. v16)

| Kaynak | Satır sayısı |
|---|---|
| Rapor (HEAD `03068f8`) | 13.423 |
| **v16 (`3e739f7`, bu ölçüm)** | **13.960** |
| Görev metnindeki tahmin ("~14.500") | yanlış — gerçek 13.960 |

**Raporun satır numaraları `03068f8` HEAD'ine göredir; v16 ağacında
satırlar kaymıştır.** Bu dokümandaki tüm `dosya:satır` referansları v16
(`3e739f7`) ağacındaki gerçek satırlardır; rapor numaraları körü körüne
alınmamıştır. Toplam: 13.960 satır, 12.884 boş-olmayan satır, 790 yorum satırı.

## 1. Dosyanın üst seviye haritası (modül bölümleri A–G)

| Bölüm | Satır | Boyut | İçerik |
|---|---|---|---|
| **A — Başlık / import** | 1–136 | 136 | `from __future__ import annotations` (:13), stdlib (atexit, json, threading…),
  cv2/numpy, PyQt6 (QtCore/QtGui/QtWidgets), `ui.shell_theme`, `ui.shell_page_frame`,
  `dataset_factory.*` (approved_pool, frame_time_machine, lru_frame_cache,
  quality_dedup_gate, rtx_trainer_engine, synthetic_augmentor) |
| **B — Taksonomi + yardımcılar** | 137–475 | 339 | `TARGET_DATASET_PHOTO_GOAL = 100000` (:153),
  `_build_yolo_class_specs` (:297–322), `CLASS_ID_TO_SPEC` (:327),
  `CLASS_KEY_ALIASES` (:337), `_safe_imread_unicode` (:417–430),
  `normalize_box_class_name` (:433–452), `_safe_imwrite_unicode` (:455–469) |
| **C — Yerleşik fabrika** | 476–832 | 357 | `_BuiltinAutoLabelingDatasetFactory` — `dataset_factory.factory_controller`
  yoksa/eksikse devreye giren yedek denetleyici (banner :472–475) |
| **D — Paylaşılan widget'lar** | 843–2626 | 1.784 | `FactoryKPICard` (:846–905), `FactoryToggleSwitch` (:908–1055),
  `_factory_explain_html` (:1058–1071), `_ReadableExplainLabel` (:1074–1124),
  `_ExplainScroll` (:1127–1184), `_InfoSelectCard` (:1187–1231),
  `_LabeledActionButton` (:1234–1291), **`YOLOPreviewCanvas` (:1294–2163, 870 satır)**,
  `SearchableTagListPanel` (:2166–2341), `ClassSelectDialog` (:2344–2535),
  `ClassBarRow` (:2538–2626) |
| **E — Worker'lar + modül yardımcıları** | 2629–3505 | 877 | `SCAN_RUNNING_STATUS` (:2632), `MIN_PUBLISH_MAP50` importu (:2641 —
  **dosya ortasında geç import, dikkat**), `format_dataset_bootstrap_complete`
  (:2644–2670), `_game_blocks_cuda_training` (:2673–2676),
  `_BackgroundCallWorker` (QThread, :2679–2695), `RTXTrainingWorker` (QThread,
  :2698–2896), `_invoke_controller_rtx_training` (:2899–2984),
  `_resolve_review_image_path` (:2990–3027), `_QUEUE_RENDER_SYNC_LIMIT = 60` /
  `_QUEUE_RENDER_CHUNK = 20` (:3031–3032), `HAND_LABEL_TXT_SOURCE` (:3041),
  `_stop_all_yolo_background_threads` (:3062–3076) + `atexit.register` (:3079),
  `_YoloDataLoadWorker` (QObject, :3082–3128), `_BULK_ASYNC_MIN_ITEMS = 40` (:3148),
  `_process_bulk_approve_items` (:3151–3296), `_bulk_approve_pool_readd` (:3299–3320),
  `_BulkApproveWorker` (QObject, :3323–3382), `_PoisonPurgeWorker` (QObject,
  :3385–3431), `_decode_thumb_qimage` (:3434–3451), `_QueueThumbThread` (QThread,
  :3454–3505) |
| **F — `YOLOFactoryDialog`** | 3508–13899 | **10.392** | 258 metot (ayrıntı bölüm 2'de) |
| **G — Uyumluluk stub'ları** | 13902–13960 | 59 | `_CompatKPI` (:13902–13912) + 12 `get_yolo_factory_dialog_*` /
  `validate_yolo_factory_dialog_contract` fonksiyonu (:13916–13957) — hepsi
  birbirini çağıran tek satırlık döndürücüler |

## 2. `YOLOFactoryDialog` iç bölümleri (F0–F15)

Sınıf docstring'i (:3508–3513): *"YOLO Veri Seti Fabrikası — 6 adımlı üretim
hattı. Yakala → Onayla → Hazırla → Eğit → Doğrula → Yayınla"*. Sınıf sabitleri
`WINDOW_TITLE_TR`, `STEP_NAMES`, `PIPELINE_COLORS` (:3515–3539).

| Bölüm | Satır | Metot | Satır (metot toplamı) | Sorumluluk |
|---|---|---|---|---|
| **F0 — ctor + yol/controller** | 3508–3746 | 10 | 195 | `__init__` (:3542–3664) — tüm durumun ilk değeri;
  `_resolve_or_create_controller`, `_get_builtin_helper`, `_dataset_dir`,
  `_models_dir`, `_freeze_path`, `_pipeline_state_path`,
  `_rejected_stems_path`, `_load_rejected_stems`, `_record_rejected_stems` |
| **F1 — pipeline disk durumu + stale + artifact** | 3748–4225 | 13 | 465 | `_load/_save_pipeline_disk_state`, `_freeze_mtime`,
  `_artifact_mtime`, `_artifacts_are_fresh_vs_freeze`, `_mark_model_stale`,
  `_reconcile_trained_flag_with_artifacts`, `_metrics_snapshot_for_disk`,
  `_candidate_rtx_run_dirs`, `_find_train_artifact_run_dir`,
  `_read_ultralytics_metrics`, `_load_train_metrics_from_artifacts`,
  `_hydrate_last_train_result_from_artifacts` |
| **F2 — tema + UI kurulumu** | 4227–6087 | 20 | 1.839 | `_apply_dark_studio_theme` (:4227–4447, 221 satır),
  `_build_ui` + `_build_step_capture/review/prepare/train/validate/publish`,
  `_build_flow_section`, `_build_strip_section`, `_build_right_rail`,
  `_build_live_journal_strip`, jurnal yardımcıları. **En büyük kurulum
  metodu:** `_build_step_review` (:4841–5332, 492 satır). Canvas sinyalleri
  burada bağlanır (:5233–5238: `boxRemoveRequested`, `boxChangeClassRequested`,
  `boxAdded`, `boxEdited`) |
| **F3 — adım makinesi** | 6089–6451 | 9 | 352 | `_go_step` (:6092–6125), `_mark_step_done`,
  `_refresh_stepper_visuals`, `_update_next_card`, `_on_next_clicked`,
  `_update_blockers`, `_count_empty_classes`, `_update_train_lock`,
  `_refresh_train_summary` |
| **F4 — yakalama** | 6456–6568 | 6 | 108 | `_on_capture_toggle`, `_save_capture_snapshot`,
  `_capture_tick`, `_finish_capture_to_review`, `_on_tm_changed`,
  `_set_preview_view` |
| **F5 — tarama / yükleme / overlay** | 6570–7266 | 25 | 667 | `_scan_review_queue_items` (:6660–6930, 271 satır),
  `_reload_review_queue`, `_apply_review_queue_items`,
  `_request_queue_reload`, `_start_yolo_data_load`, `_stop_yolo_threads`,
  `wait_for_yolo_load`, yükleme overlay metotları, `resizeEvent` |
| **F6 — küçük resimler (thumb)** | 7268–7355 | 6 | 80 | `_wire_queue_thumb_requests`,
  `_ensure_thumb_thread`, `_schedule_visible_thumbs`, `_visible_queue_rows`,
  `_request_visible_thumbs`, `_on_thumb_ready` |
| **F7 — filtre / lejant / kuyruk render** | 7357–8130 | 26 | 749 | Filtre slotları, `_build_class_legend`,
  `_queue_thumb_pixmap`, `_create_queue_item_widget`,
  `_filtered_review_indices` (:7659–7761, 103 satır),
  `_update_filter_action_buttons` (:7763–7879, 117 satır),
  `_render_queue_list*` (chunk'lı render, 7929–8059), `_on_queue_row_changed` |
| **F8 — kare render + kutu eylemleri** | 8132–9173 | 24 | 1.013 | `_imread_cached`, `_create_review_det_row`
  (:8200–8351, 152 satır), `_render_review_frame` (:8353–8481, 129 satır),
  `_accept_box`, `_demote_box_to_weak`, `_reject_box` (:8736–8822, 87 satır),
  `_is_reject_protected_box`, el-etiketi (TXT) koruma predikatları
  (:8836–8942), `_stems_for_sample_wipe`, `_sample_wipe_paths`,
  `_delete_review_sample`, `_batch_delete_review_samples` |
| **F9 — manuel kutu + persist + review_done** | 9175–9832 | 14 | 642 | `_populate_manual_tag_combo`,
  `review_items` property (getter :9252–9253, setter :9256–9257),
  `_on_manual_box_added` (:9284–9381), `_on_box_edited` (:9383–9465),
  `_validate_review_boxes`, `_pool_add_approved`,
  `_persist_review_item` (:9570–9667), `_review_done` (:9669–9832, 164 satır) |
| **F10 — kısayol / eventFilter / lightbox** | 9834–10155 | 15 | 308 | A/R/S kısayolları, `eventFilter`
  (:9952–10053, 102 satır), `_open_review_lightbox`, `_open_capture_lightbox`,
  `_review_skip` |
| **F11 — toplu işlemler** | 10157–11420 | 35 | 1.227 | `_bulk_approve*` (senkron + async),
  `_bulk_prune_low_confidence_boxes` (:10405–10659, 255 satır),
  canlı relabel (filtered/all), `_bulk_reject_filtered`,
  `_requeue_existing_dataset`, `_purge_poisoned_labels` (+ purge thread),
  `_turbo_relabel_filtered`, `_auto_label_unlabeled_suspicious`,
  `keyPressEvent` |
| **F12 — sağlık / dondurma / doğrulama / yayın** | 11422–12503 | 16 | 1.067 | `_rebuild_health_checks`
  (:11422–11558, 137 satır), `_freeze_dataset` (:11645–11796, 152 satır),
  `_run_validate` (:11884–12008), `_finish_validate`, `_publish_model`
  (:12169–12432, 264 satır), `_rollback_model`, `_refresh_publish_checklist`,
  `_refresh_version_list` |
| **F13 — controller sync / dashboard / web / log** | 12505–13192 | 15 | 671 | `_sync_toggles_from_controller`,
  `_collect_live_stats`, `_time_machine_snapshot`, `refresh_dashboard`
  (:12673–12836, 164 satır), `_resolve_review_image`,
  `_latest_disk_preview_path`, `export_web_state` (:12890–13122, 233 satır),
  `_append_log` (:13167–13182), `_mirror_status_to_app_log` |
| **F14 — bootstrap / eğitim** | 13194–13752 | 16 | 541 | `on_bootstrap_clicked`,
  `_on_bootstrap_worker_done/failed`, `_accept_train_result` (:13335–13409),
  `_start_training_worker`, `on_train_rtx_clicked` (:13476–13563),
  `_on_training_finished/failed`, `wait_for_training_completion`,
  `get_onnx_runtime_status` |
| **F15 — çeşitli + kapanış + sağlık API'si** | 13754–13899 | 8 | 139 | `on_open_dataset_folder_clicked`,
  `on_refresh_clicked`, `get_yolo_factory_dialog_health_summary`,
  `__del__` (:13836–13843), `closeEvent` (:13845–13857),
  `get_yolo_factory_dialog_ready_flag/version_info/diagnostic_snapshot` |

Metot satır toplamı 10.063; kalan 329 satır sınıf başlığı/sabitler
(:3508–3541), bölüm banner'ları ve metotlar arası boşluklardır.

## 3. Paylaşılan durum haritası (bölmeyi zorlaştıran asıl şey)

### 3.1 `self._pipeline` — merkezi durum dict'i

- Tanım `__init__` :3594–3608, 18 anahtar: `step`, `pending`, `reviewed`,
  `frozen`, `trained`, `validated`, `published`, `scanned`, `imbalance_fixed`,
  `capturing`, `session`, `format`, `run_id`, `model_stale`, `stale_reason`,
  `stale_at` (+ `__init__` :3640–3641'de eklenen `train_session_ok`,
  `train_completed_at`).
- **110 `self._pipeline[…]` + 52 `self._pipeline.get(…)` erişimi; 182 satır
  `_pipeline` geçiriyor.** Okuyan/yazan bölümler: F1 (disk round-trip
  :3748–3839), F3 (`_go_step` :6092, `_refresh_stepper_visuals` :6138,
  `_update_next_card` :6229, `_update_blockers` :6328, `_update_train_lock`
  :6358), F4 (`_on_capture_toggle` :6456), F5 (`_pending_count` :6613,
  `_sync_pending_from_queue` :6628), F8/F9 (onayda `pending` yeniden hesabı,
  örn. :8612, :8620), F11, F12 (`_freeze_dataset` :11645, `_run_validate`
  :11884, `_publish_model` :12169), F14 (eğitim bitişi :13609).
- **Sonuç:** `_pipeline` bir "herkesin yazdığı global"dir. Bölme öncesi
  P3-27 (tek onay modeli) ve P3-28 (tek sayım kaynağı) yapılmadan F1'in
  ayrılması, durumu iki yerde yaşayan bir dict'e böler. F1 en son
  ayrılacak bölümlerdendir (bkz. bölüm 6, adım 5).

### 3.2 `self._review_items` + `self._review_index` — kuyruk belleği

- **103 `self._review_items` erişimi.** Yazıcılar: `_apply_review_queue_items`
  (:6952, F5), `_remove_review_item_at` (:9178, F8), `review_items` setter
  (:9257, F9 — testler de kullanıyor, bkz. bölüm 5). Okuyucular: F5, F7
  (:7677, :7878, :7989, :8041, :8123), F8 (`_render_review_frame` :8371,
  kutu eylemleri :8547–8748), F9 (:9288, :9387), F11
  (`_bulk_approve_pool_readd` çağrısı :10230).
- `self._review_index`: 14 yazım (:3591, :8129, :10150, :11044, :6963, :7965,
  :9180, :9182…). Kuyruk + render + onay üçlüsü aynı indekse dokunuyor.

### 3.3 `self._last_train_result` — eğitim sonucu önbelleği

- 22 erişim. Yazıcılar: F1 (`_load_pipeline_disk_state` :3807,
  `_mark_model_stale` :3896, `_hydrate_…` :4215/:4225), F12 (`_freeze_dataset`
  :11780). Okuyucular: F3 (`_go_step` :6101, `_update_next_card` :6254), F12
  (`_run_validate` :11885, `_finish_validate` :12030, `_publish_model`
  :12209), F14. **P1-11 (reddedilen eğitimin `_last_train_result`'ı
  temizlemesi) bu alanın tek yazıcısını değiştirir — bölme o düzeltmeden
  sonra yapılmalıdır.**

### 3.4 `self._rejected_stems` — reddedilen kareler

- Yazıcılar: `_load_rejected_stems` (:3716–3729), `_record_rejected_stems`
  (:3731–3746), F8 silme yolları (:9060, :9139). Okuyucular: F5 tarama
  (:6941, :6999), F9 `_review_done` (:9707). **Not:** Selçuk'un 11 Kasım
  05:19 kararı gereği reddet kalıcı silmedir; `rejected_stems` geri yükleme
  yolu P0-2'de iptal edildi — bu alanın bölme planında "geri yükleme"
  varsayımı yoktur.

### 3.5 Önbellekler ve worker handle'ları

- `_queue_thumb_cache` (Dict, :3631; yazım :7345, :7531), `_review_decode_cache`
  (:3627), `_stats_cache`/`_stats_cache_at` + TTL (:3617–3619), `_class_counts_cache`
  (:3585), `_last_stats_sig`/`_last_class_sig` (:3622–3623) — F5/F6/F7/F8/F13
  arasında paylaşılır; LRU davranışı P2-20'nin konusudur, bölme değiştirmez.
- Worker handle çiftleri (thread + worker + running bayrağı + started_at):
  `_yolo_load_*` (:3556–3560, F5), `_bulk_approve_*` (:3569–3573, F11),
  `_purge_*` (:3574–3577, F11), `_training_worker` (:3553, F14),
  `_bootstrap_worker` (:3554, F14), `_thumb_thread` (:3562, F6).
  Kapanış hepsini `closeEvent`/`__del__` → `_stop_yolo_threads` (:7050–7091)
  üzerinden durdurur; ayrıca modül düzeyinde `atexit` kaydı (:3079) vardır
  (P2-18 kapanış ağı — bölme sırasında kayıt yeri değişmemeli).

### 3.6 Widget alanları — en büyük kenetlenme

- F2'de yaratılıp **F2 dışında okunan `self.*` alanı: 90 adet** (AST+grep
  sayımı). Örn. `btn_train_rtx`, `btn_publish`, `btn_approve_filtered`,
  `queue_list`, `review_preview`, `det_host`, `hp_epoch`, `train_progress_bar`,
  `freeze_state_lbl`, `flow_bar`, `block_count`, `class_filter`, `bulk_range`.
- **Sonuç:** UI kurulumunu (F2) handler'lardan ayırmak, 90 alanın ya
  dialog'da kalmasını ya da bir "view" nesnesine taşınmasını gerektirir.
  Taşıma, F3–F15'teki her `self.btn_*` / `self.*_lbl` erişimini etkiler —
  bu yüzden F2 bölünmesi mekanik "kes-yapıştır" değildir; önce erişim
  noktalarının envanteri çıkarılmalıdır (bu dokümanın 90'lık sayısı o
  envanterin başlangıcıdır).

## 4. Bölümler arası çağrı haritası (yönlü)

AST ile çıkarılan sınıf-içi çağrı grafiğinden (yalnızca bölümler arası
kenarlar; aynı bölüm içi çağrılar hariç):

| Kaynak → Hedef | Temsilci çağrılar |
|---|---|
| F0 → F1 | `__init__` → `_load_pipeline_disk_state` (:3646) |
| F2 → F5/F6/F7/F8 | `_build_ui` → `_build_yolo_loading_overlay`, `_wire_queue_thumb_requests`;
  `_build_step_review` → `_build_class_legend`, `_update_filter_action_buttons`,
  `_reject_box` (:5233–5238 canvas sinyal bağlantıları) |
| F3 → F1/F5/F12/F13 | `_go_step` → `_save_pipeline_disk_state`, `_request_queue_reload`,
  `_rebuild_health_checks`, `_run_validate`, `_refresh_publish_checklist`;
  `_update_train_lock` → `_pending_train_blockers`, `_refresh_train_summary` |
| F4 → F3/F13 | `_finish_capture_to_review` → `_go_step`, `_mark_step_done`;
  `_capture_tick` → `_time_machine_snapshot` |
| F5 → F1/F3/F7/F8/F13 | `_on_yolo_data_loaded` → `_apply_review_queue_items`,
  `_collect_live_stats`, `refresh_dashboard`; `_apply_review_queue_items` →
  `_render_queue_list`, `_render_review_frame`, `_sync_pending_from_queue` |
| F6 → F5 | `_request_visible_thumbs` → `_ensure_thumb_thread`, `_visible_queue_rows` |
| F7 → F8 | `_render_queue_list` → `_render_review_frame`;
  `_create_queue_item_widget` → `_frame_conf` |
| F8 → F1/F7/F9 | Kutu eylemleri → `_mark_model_stale`, `_persist_review_item`,
  `_render_queue_list`, `_render_review_frame`; `_delete_review_sample` →
  `_record_rejected_stems`, `_is_hand_label_protected_item` |
| F9 → F1/F3/F7/F8 | `_review_done` → `_persist_review_item`, `_delete_review_sample`,
  `_mark_model_stale`, `_render_queue_list`, `_render_review_frame`,
  `_update_blockers`, `_update_next_card`; `_persist_review_item` →
  `_pool_add_approved`, `_validate_review_boxes` |
| F10 → F8/F9 | `eventFilter`/`keyPressEvent` → `_reject_box`, `_review_done`,
  `_on_step2_approve/reject`, `_review_skip`; lightbox'lar → `_reject_box` |
| F11 → F1/F3/F5/F7/F8/F9 | `_finalize_bulk_approve`, `_bulk_prune_*`, `_bulk_reject_filtered`,
  `_on_purge_finished` → `_reload_review_queue`, `_mark_model_stale`,
  `_render_queue_list`, `_sync_pending_from_queue`, `_update_blockers`,
  `_update_next_card`; `_relabel_single_item` → `_persist_review_item` |
| F12 → F1/F3/F13 | `_freeze_dataset` → `_collect_live_stats`, `_mark_model_stale`,
  `_save_pipeline_disk_state`, `_go_step`, `_update_train_lock`;
  `_publish_model`/`_run_validate` → artifact/stale okuyucuları (F1) |
| F13 → F3/F5/F7/F12 | `refresh_dashboard` → `_request_queue_reload`, `_update_blockers`,
  `_update_next_card`, `_update_train_lock`, `_time_machine_snapshot`;
  `export_web_state` → `_reload_review_queue`, `refresh_dashboard` |
| F14 → F1/F3/F13 | `_on_training_finished` → `_accept_train_result`,
  `_artifacts_are_fresh_vs_freeze`, `_save_pipeline_disk_state`, `_go_step`,
  `refresh_dashboard`; `on_train_rtx_clicked` → `_pending_train_blockers`,
  `_start_training_worker` |
| Herkes → F13 | `_append_log` (:13167–13182) — log/journal aynası; `_mark_model_stale`
  de F1'den `_append_log` çağırır (:3884) |
| Herkes → F1 | `_dataset_dir` (:3685), `_mark_model_stale` (:3884),
  `_save_pipeline_disk_state` (:3815) — en çok çağrılan 3 yardımcı |

### 4.1 Döngüsel bağımlılık riskleri (bölmede modül döngüsü çıkaracak kenarlar)

1. **F3 ↔ F12:** `_go_step` → `_rebuild_health_checks` (F12) → `_go_step`;
   ayrıca `_go_step` → `_run_validate` → `_finish_validate` → `_go_step`.
   Adım makinesi ile sağlık/doğrulama birbirini çağırır — ayrı modüle
   alınırlarsa çift yönlü import gerekir; çözüm: `_go_step`'in F12'ye
   bağımlılığını bir callback/olay ile koparmak (bölme adımında tasarlanacak).
2. **F13 ↔ F5:** `refresh_dashboard` → `_request_queue_reload` →
   `_on_yolo_data_loaded` → `refresh_dashboard`. Dashboard ile yükleme
   birbirini tetikler.
3. **F7 ↔ F8:** `_render_queue_list` → `_render_review_frame` →
   `_create_review_det_row` → `_accept_box`/`_reject_box` →
   `_render_queue_list`. Kuyruk render ile kare render birbirine bağlı.
4. **F13 → F5 → F13:** `export_web_state` → `_reload_review_queue` →
   (yükleme bittiğinde) `refresh_dashboard` → … Web export, yükleme ve
   dashboard üçgeni.
5. **F1 → F13 → F2:** `_mark_model_stale` → `_append_log` →
   `_journal_line_visible` (F2, :5888). Durum katmanı, log üzerinden UI
   katmanına dokunur.
6. **E → F (worker → dialog):** `_BulkApproveWorker` dialog referansı alır
   (`self._dialog`, :3347) ve `_process_bulk_approve_items(_dialog, …)`
   çağırır (:3360); `_PoisonPurgeWorker` benzer. Worker'lar ayrı modüle
   taşınırsa dialog'a geri çağrı için protokol (Protocol/callback) gerekir;
   aksi halde `workers → dialog → workers` döngüsü çıkar.
7. **Modül düzeyi `atexit`:** `_stop_all_yolo_background_threads` (:3062)
   + `atexit.register` (:3079) — worker kayıt defteri
   (`_QUEUE_THUMB…` benzeri global kayıt, :3466/:7023/:10267/:11215
   yorumlarında "güvenlik ağı" diye geçer) E bölümündedir; F15'teki
   `_stop_yolo_threads` ayrı bir yoldur. İki kapanış yolunun tek modülde
   kalması gerekir.

## 5. Dış bağımlılıklar (dosya dışı)

### 5.1 Bu dosyanın import ettikleri (kullanım sayıları, grep)

| Dış modül / ad | Kullanım | Not |
|---|---|---|
| `dataset_factory.approved_pool` (`ApprovedDatasetPool`, `get_approved_pool`, `is_approved`) | 12 + 4 + 4 | F9 `_pool_add_approved`, F11 `_bulk_approve_pool_readd` (:3299, :3374, :10230) |
| `dataset_factory.rtx_trainer_engine` (`RTXTrainerEngine`, `probe_gpu_environment`, `rtx_runs_dir_for`) | 4 + 2 + 3 | F1 artifact/run-dir çözümleme, F14 eğitim |
| `dataset_factory.dataset_validator` (`MIN_PUBLISH_MAP50`) | 13 | Geç import (:2641); F12 yayın kapısı |
| `dataset_factory.frame_time_machine` (`PreGatherTimeMachineBuffer`) | 2 | F4/F13 (`_time_machine_snapshot`) |
| `dataset_factory.lru_frame_cache` (`get_lru_frame_cache`) | 2 | F8 `_imread_cached` |
| `dataset_factory.quality_dedup_gate` (`DatasetQualityGate`) | 2 | C bölümünde (builtin factory) |
| `dataset_factory.synthetic_augmentor` (`SyntheticYOLOAugmentor`) | 2 | C bölümünde |
| `ui.shell_theme` (`SHELL_COLORS`, …) | 6 | F2 tema |
| `ui.shell_page_frame` (`wrap_shell_tab_root_layout`) | 2 | `_build_ui` (:4452) |
| `ui.yolo_factory_dialog` içi: `normalize_box_class_name` 5, `_safe_imread_unicode` 6, `_safe_imwrite_unicode` 5 | — | B bölümünün F8/F9/F11 tarafından kullanımı |

### 5.2 Bu dosyayı import edenler (bölmede kırılacak yüzey)

- **Tek prod giriş noktası:** `main.py:2667–2691` `open_yolo_factory_dialog`
  (`from ui.yolo_factory_dialog import YOLOFactoryDialog`, ctor
  `factory_controller=…`, `parent=…`). Ayrıca `main.py:2694`
  `trigger_ai_training_and_open_dialog` aynı dialogu açar.
- **Web tüketici:** `ui/yolo_pipeline_web.py:830`
  `self.backend.export_web_state()` — F13'ün `export_web_state` metodu
  dışarıdan çağrılır; imzası/sözleşmesi bölmede korunmalıdır.
- **Testlerin import ettiği adlar** (grep, `tests/` + kök `test_*.py`):
  `YOLOFactoryDialog`, `YOLOPreviewCanvas`, `RTXTrainingWorker`,
  `HAND_LABEL_TXT_SOURCE`, `CLASS_ID_TO_SPEC`, `CLASS_KEY_ALIASES`.
  Bu adlar bölme sonrası da `ui.yolo_factory_dialog` üzerinden import
  edilebilir kalmalıdır (re-export) — aksi halde ~20 test dosyası elden
  geçmelidir (tam liste bölüm 6.4'te).
- **Testlerin doğrudan çağırdığı "private" metot:** `YOLOFactoryDialog._scan_review_queue_items(tmp_path, set(...))`
  (`tests/test_label_backup_trash.py:91`,
  `tests/test_yolo_factory_hand_label_protection.py:128`) — tarama
  fonksiyonu F5'ten taşınırsa bu çağrı biçimi de taşınmalıdır.
- **Testlerin kullandığı property:** `dlg.review_items` getter/setter
  (`tests/test_yolo_factory_approved_and_manual_tag.py:185–200`).
- **Yorumla sözleşme kuran dış dosyalar** (import yok ama metin bağımlılığı
  var — bölmede yorumlar da güncellenmeli):
  `dataset_factory/approved_pool.py:1071, 1633, 1647`,
  `dataset_factory/dataset_validator.py:52, 1366, 1435`,
  `dataset_factory/factory_controller.py:7`,
  `dataset_factory/label_trust.py:26, 33, 1865`,
  `dataset_factory/lru_frame_cache.py:25, 31`.

## 6. Bölme planı (sıralı, her adım ayrı dal + ayrı bundle)

**Önkoşullar (bölmeden önce):** P0 (8 madde) ve P1 (9 madde) tamamlanmış
olmalı — özellikle P0-1 (TXT uzlaştırması), P0-2 (yedek), P0-3
(`_reject_box` guard), P1-10 (`_persist` dönüşümü), P1-11
(`_last_train_result` temizliği), P1-14 (taksonomi tek otorite).
Ayrıca P3-26 (tek yazıcı), P3-27 (tek onay modeli), P3-28 (tek sayım)
bölmeden önce bitmeli; aksi halde durum üç ayrı modülde üç ayrı kural
olarak yeniden üretilir. **Bu önkoşullar sağlanmadan adım 5 ve sonrasına
geçilmez.**

| Adım | Hedef modül | Taşınan bölüm | Risk | Doğrulama |
|---|---|---|---|---|
| **1** | `ui/yolo_factory_widgets.py` | **D** (1.784 satır; `YOLOPreviewCanvas` dahil) | **Düşük.** Widget'lar dialog'dan bağımsız sınıflar; tek bağ sinyal (`boxDrawn/boxAdded/boxEdited/boxRemoveRequested/boxChangeClassRequested`, :1302–1306) ve F2'deki `connect` (:5233–5238) + `tag_panel.classSelected` (:5137). `YOLOPreviewCanvas` tek başına 870 satır — kendi içinde de god-widget adayıdır, ama bu adımda bölünmez, bütün taşınır. | `tests/test_pyqt6_yolo.py`, `test_step2_move_resize_and_large_canvas.py`, `test_w25_manual_box_drawing.py`, `test_w27_three_tier_box_colors.py`, `test_live_ui_smoke.py` + `py_compile` |
| **2** | `dataset_factory/factory_controller.py` içine veya `ui/yolo_factory_builtin.py` | **C** (357 satır) | **Düşük–orta.** Tek bağ `_get_builtin_helper` (:3680–3683) ve `_YoloDataLoadWorker`'ın `helper` parametresi (:3107). Karar: builtin factory zaten `dataset_factory` kavramı — doğru ev orasıdır; ama `factory_controller.py:7` yorumu bu dosyaya atıf yapıyor, taşınırken güncellenmeli. | `tests/test_yolo_factory_train_gates.py`, `test_train_click_bootstrap.py` |
| **3** | `dataset_factory/class_taxonomy.py` (mevcut otorite) | **B** (339 satır) | **Orta.** `CLASS_ID_TO_SPEC` / `CLASS_KEY_ALIASES` testler tarafından import ediliyor (bölüm 5.2) — **re-export şart**. `normalize_box_class_name` P1-14'ün konusudur; bu adım P1-14 ile birlikte yapılmalı, ayrı yapılmamalı. `TARGET_DATASET_PHOTO_GOAL` (:153) tek sabit — taşınması kolay ama referansları grep ile doğrulanmalı. | `tests/test_player_character_labeling.py`, `test_work_package_k_filter_actions.py`, `test_yolo_factory_approved_and_manual_tag.py` |
| **4** | `ui/yolo_factory_workers.py` | **E** (877 satır; worker sınıfları + `_process_bulk_approve_items`, `_bulk_approve_pool_readd`, `_decode_thumb_qimage`, `_resolve_review_image_path`) | **Orta.** `_BulkApproveWorker._dialog` geri çağrısı (bölüm 4.1 madde 6) için `Protocol`/callback tanımlanmalı. `atexit.register` (:3079) ve kayıt defteri bu modülde kalmalı; F15 `_stop_yolo_threads` ile çift kapanış yolu korunmalı. `MIN_PUBLISH_MAP50` geç importu (:2641) dosya başına alınmalı. | `tests/test_bulk_approve_prune_guard.py`, `test_yolo_factory_train_gates.py`, `tests/test_all_bot_screens_and_tabs_suite.py` |
| **5** | `ui/yolo_factory_state.py` | **F1** (465 satır) + `_pipeline` erişim arayüzü | **Yüksek.** `_pipeline` 162 doğrudan erişimle her bölüme dağılmış durumda. Önce tüm erişimler `_pipeline` dict'inden bir arayüze (property/metot) indirgenmeli; ancak ondan sonra F1 taşınabilir. P3-26/27/28 bu adımın önkoşuludur. Yanlış sıra = disk round-trip (`_load/_save_pipeline_disk_state`) ile bellek durumu ayrışır, stale bayrağı çelişir (rapor F078). | Tam YOLO süiti + `test_w22_reopen_persistence_guard.py`, `test_step5_validate_to_step6_publish.py`, `test_train_gate_delta.py` |
| **6** | `ui/yolo_factory_view.py` (builder) | **F2** (1.839 satır) | **Orta–yüksek.** 90 widget alanının (bölüm 3.6) dialog'da mı kalacağı, view nesnesine mi taşınacağı bu adımın tasarım kararıdır. Öneri: alanlar dialog'da kalır, builder dialog referansı alıp doldurur (en az kırılma). Sinyal bağlantıları (:5233–5238, :5137) builder'da kalır ama hedef metotlar F8/F9'da — builder → handler bağımlılığı tek yönlü tutulmalı. | `test_live_ui_smoke.py` (özellikle `test_yolo_factory_dialog_lightbox_integration`), `tests/test_all_bot_screens_and_tabs_suite.py` |
| **7a** | `ui/yolo_factory_queue.py` | **F5 + F6 + F7** (667 + 80 + 749 satır) | **Yüksek.** F7↔F8 döngüsü (bölüm 4.1 madde 3) koparılmadan taşınamaz: `_render_review_frame` çağrısı queue modülünden review modülüne callback ile verilmeli. `_scan_review_queue_items` testlerce doğrudan çağrılıyor (bölüm 5.2) — imza korunmalı. | `tests/test_approved_pool_txt_hand_labels.py`, `test_w46_filter_scoped_bulk_actions.py`, `test_work_package_k_filter_actions.py`, `test_reject_box_guard.py` |
| **7b** | `ui/yolo_factory_review.py` | **F8 + F9 + F10** (1.013 + 642 + 308 satır) | **Yüksek.** `_persist_review_item` + `_review_done` + el-etiketi koruma predikatları tek modülde kalmalı (P0-1/P0-3/P1-10'un çekirdeği buradadır). Kısayollar (F10) `eventFilter` üzerinden F8/F9'a bağlanır — dialog'da ince bir yönlendirme katmanı kalmalıdır. | `tests/test_label_backup_trash.py`, `test_yolo_factory_hand_label_protection.py`, `test_demote_requeue_flags.py`, `test_w70_end_to_end_labeling_pipeline.py`, `test_w21_approved_queue_removal.py`, `test_w25_manual_box_drawing.py` |
| **7c** | `ui/yolo_factory_bulk.py` | **F11** (1.227 satır) | **Orta–yüksek.** Toplu onay/budama/relabel/purge hepsi `_persist_review_item` (F9) + `_reload_review_queue` (F5) + `_mark_model_stale` (F1) üzerinden çalışır; üçüne de arayüz gerekir. `_process_bulk_approve_items` zaten adım 4'te taşındığı için bu adım onun üzerine kurulur. | `tests/test_bulk_approve_prune_guard.py`, `test_fix_r35_yolo_factory_prune_ready.py`, `test_w46_filter_scoped_bulk_actions.py` |
| **7d** | `ui/yolo_factory_training.py` | **F12 + F14** (1.067 + 541 satır) | **Orta–yüksek.** F3↔F12 döngüsü (bölüm 4.1 madde 1) koparılmadan taşınamaz. `_accept_train_result` + `_freeze_dataset` + `_publish_model` aynı modülde kalmalı (hepsi F1 artifact okuyucularını kullanır). | `test_yolo_factory_train_gates.py`, `test_train_click_bootstrap.py`, `test_train_gate_delta.py`, `test_step5_validate_to_step6_publish.py` |
| **8** | — (taşınma yok) | **G** (59 satır) + F0/F3/F4/F13/F15 çekirdeği | **—** G'deki 12 stub P2-24 ölü-kod temizliğiyle birlikte değerlendirilir (silinir veya kalır — taşınmaz). F0 (ctor), F3 (adım makinesi), F4 (yakalama), F13 (dashboard/log), F15 (kapanış) dialog çekirdeğinde kalır; `YOLOFactoryDialog` bu çekirdek + modüllere delege eden ince bir sınıf olur. | Tam süit |

**Adım başına kural:** her adım kendi dalında, tek commit'lik taşınma +
`py_compile` + tablodaki test dosyaları + `test_live_ui_smoke.py`.
Taşınma commit'inde davranış değişikliği yapılmaz (import yolu + re-export
dışında diff yalnızca `git diff -w` ile doğrulanır). Her adım ayrı bundle
ile teslim edilir; birleştirme v16 tarzı tek bundle'da toplanır.

### 6.1 Sıra gerekçesi (neden bu sıra)

- **1–4 önce:** dialog sınıfına hiç dokunmadan, modül düzeyindeki bağımsız
  parçalar ayrılır. Kırılma yüzeyi küçüktür; testler hemen doğrular.
- **5 sonra:** `_pipeline` arayüze indirgenmeden F1 taşınamaz; bu iş P3-26/27/28
  ile aynı tasarım masasında yapılır.
- **6 sonra:** F2'nin taşınması 90 alanın kaderine bağlıdır; bu karar
  7a–7d'nin handler'larını da etkiler, o yüzden view'dan önce kararlaştırılır.
- **7a–7d en son:** birbirine en çok bağlı bölümler; her biri kendi dalında,
  tam süit ile. 7a ile 7b'nin sırası değişebilir ama ikisi de 5'ten sonra
  gelmelidir (ikisi de `_pipeline`/`_review_items` arayüzüne dayanır).

## 7. Bölünmeyecekler / karar bekleyenler

- **`_factory_explain_html` (:1058–1071):** P2-24 ölü-kod listesinde adı
  geçiyor ("kullanılmıyorsa"). Bölme sırasında D ile birlikte taşınmamalı;
  önce P2-24 kararı beklenmeli.
- **`_turbo_relabel_filtered` (:11316–11349):** P2-24'te aynı şekilde
  "kullanılmıyorsa" notuyla geçiyor. F11 taşınırken ölü olduğu doğrulanırsa
  taşınmadan silinir (ayrı karar, bu dokümanın kapsamı değil).
- **`get_yolo_factory_dialog_*` stub zinciri (G):** 12 fonksiyon birbirini
  çağıran tek satırlık döndürücüler; bölme konusu değil, P2-24 kapsamındadır.
- **`YOLOPreviewCanvas` (870 satır):** D ile birlikte bütün taşınır; kendi
  iç bölünmesi ayrı bir P3 maddesidir, bu planın parçası değildir.
- **`review_items` property'si ve `_scan_review_queue_items` imzası:**
  testlerce doğrudan kullanılıyor — taşınırken imza ve erişim biçimi
  korunur, test değişikliği gerekmez.
- **Rapor satır numaraları:** rapordaki `dialog:XXXX` referansları 03068f8
  HEAD'ine göredir; v16'da karşılığı bu dokümandaki aralıklardan
  bulunmalıdır. Gelecek taramalar satır numarası yerine `sembol + dal`
  referansı vermelidir.

## 8. Açık noktalar

1. **F2'deki 90 alanın envanteri:** sayı çıkarıldı ama alan başına
   "hangi bölüm okuyor" tablosu çıkarılmadı — adım 6'nın tasarım girdisidir.
2. **`_pipeline` erişimlerinin bölüm bazında tam dökümü:** toplam 162
   erişim sayıldı; her erişimin okuma/yazma ayrımı çıkarılmadı — adım 5'in
   ön çalışmasıdır.
3. **Widget'ların dış kullanımı:** D bölümündeki sınıfları dialog dışında
   kullanan başka dosya var mı (örn. `FactoryKPICard` başka sekmede
   kullanılıyor mu) — grep yalnızca bu dosya içinde yapıldı; adım 1 öncesi
   repo genelinde `FactoryKPICard|FactoryToggleSwitch|SearchableTagListPanel|ClassBarRow`
   araması yapılmalıdır.
4. **P3-26/27/28 kararları:** bu doküman onların tasarımını yapmaz; bölme
   planı onların bitmiş olmasını varsayar. Onlar bitmeden adım 5 ve 7a–7d
   başlatılmamalıdır.
5. **Canlı doğrulama yok:** bu analiz statiktir (AST + grep). Davranış
   iddiaları (örn. "kapanışta iki yol var") kod okumasına dayanır; runtime
   doğrulaması bölme adımlarının testlerinde yapılacaktır.

## 9. Kanıt özeti (bu dokümanın dayandığı ölçümler)

- Dosya: 13.960 satır (`wc -l`, v16 `3e739f7`).
- `YOLOFactoryDialog`: :3508–13899, 10.392 satır, **258 metot** (AST).
- En uzun metotlar: `_build_step_review` 492 (:4841–5332),
  `_scan_review_queue_items` 271 (:6660–6930), `_publish_model` 264
  (:12169–12432), `_bulk_prune_low_confidence_boxes` 255 (:10405–10659),
  `export_web_state` 233 (:12890–13122).
- `self._pipeline[`: 110, `self._pipeline.get(`: 52, `self._review_items`:
  103, `self._last_train_result`: 22 erişim (grep).
- F2'de yaratılıp dışarıda okunan `self.*`: 90 (AST + grep kesişimi).
- Çağrı grafiği: 258 metodun `self.<metot>(…)` çağrıları AST ile çıkarıldı;
  bölüm 4 yalnızca bölümler arası kenarları listeler.
- `py_compile` OK (`python3 -m py_compile ui/yolo_factory_dialog.py`).
- Test koşulmadı; doğrulama listeleri bölüm 6'daki tabloda dosya adıyla
  verilmiştir.
