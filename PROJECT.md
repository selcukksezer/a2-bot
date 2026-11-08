# Project: Gatherion-Local v2.x (Türkçe UI)

## Architecture
Gatherion-Local is a local automation, AI vision, and robotics bot for Aion 2.
- **Core Loop**: `core/state_machine.py` (`BotWorker`), managing modes (`BotMode.COMBAT`, `BotMode.ROUTE_MODE`, `BotMode.EXPLORER`, etc.).
- **Perception**: `core/perception_hub.py`, `core/reflex_layer.py` (ONNX / YOLOv8 / YOLO11), `core/ocr_distance.py`, `core/skill_bar_reader.py`.
- **Action & Input**: `core/input_driver.py` (Interception kernel driver, rootless UDP HID, soft fallback), `core/game_action_bus.py`, `core/human_timing.py`.
- **Navigation & Cartography**: `core/route_manager.py`, `core/auto_cartographer.py`, `core/explorer_mode.py`, `ui/map_viewer.py`.
- **YOLO Training Factory**: `ui/yolo_factory_dialog.py`, `core/game_gpu_throttle.py`, `dataset_factory/label_trust.py`, `dataset_factory/dataset_validator.py`.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | R1.1 GPU Throttle Hardening | Strictly check executable name (`exe_is_aion_game`) and window visibility; exclude browsers, IDEs, explorer, python. | M1 | ORIGINAL_REQUEST §R1 |
| 2 | R2.1 Pre-label & Human Gate | Gate training on human approved labels; prevent unapproved data entering training split. | M2 | ORIGINAL_REQUEST §R2 |
| 3 | R2.2 Label QA Checks | Verify bounding boxes for overlap (IoU > 0.8), zero-area, and out-of-bounds normalized coords. | M2 | ORIGINAL_REQUEST §R2 |
| 4 | R2.3 Stale Model Marking | Invalidate model status and mark as stale whenever dataset labels are modified. | M2 | ORIGINAL_REQUEST §R2 |
| 5 | R1.2 UI Hyperparameter Propagation | Pass chosen model, imgsz, batch, epochs from UI to `RTXTrainingWorker`. | M3 | ORIGINAL_REQUEST §R1 |
| 6 | R1.3 Real Epoch Training | Run genuine YOLO epochs; eliminate mock/short-circuit ~1s completion. | M3 | ORIGINAL_REQUEST §R1 |
| 7 | R1.4 Validation & Publishing | Verify mAP50, warn on stale publish, export and publish ONNX to `models/yolov8n_gatherion.onnx`. | M3 | ORIGINAL_REQUEST §R1 |
| 8 | R3.1 Olive HP Bar Target Detection | Add olive-green HSV mask in `combat_recovery._detect_enemy_target_bar` for wild/neutral mobs. | M4 | ORIGINAL_REQUEST §R3 |
| 9 | R3.2 Unbias SkillBarReader | Eliminate brightness priority bias towards slot '2' in `skill_bar_reader.py`. | M4 | ORIGINAL_REQUEST §R3 |
| 10 | R5.1 Route Manager Sequencing | Fix 3+ point route cycling in `get_next_ready_node()` by sequence index; fix spiral odometry. | M5 | ORIGINAL_REQUEST §R5 |
| 11 | R5.2 Map Viewer Route Ordering | Persist optimized node ordering to `RouteManager._regions` and `save_region()`. | M6 | ORIGINAL_REQUEST §R5 |
| 12 | R4.1 AI Boxes to Cartography | Connect `PerceptionHub.last().targets` and YOLO boxes to cartography; recognize `yz_3d`. | M7 | ORIGINAL_REQUEST §R4 |
| 13 | R6.1 Input Interception Single-Fire | Return early on Interception success; eliminate 4x soft fallback double-injection; log fallback. | M8 | ORIGINAL_REQUEST §R6 |
| 14 | R6.2 Input Jitter & Timing | Replace fixed sleeps with Gaussian jitter and GCD alignment. | M8 | ORIGINAL_REQUEST §R6 |
| 15 | R3.3 Combat Rotation & ms Burst | Round-robin skill keys (`1,2,3,4,q,e,r`) with ms-level burst delay. | M9 | ORIGINAL_REQUEST §R3 |
| 16 | R3.4 Combat Knobs Boot Logging | Log `COMBAT SKILL KNOBS trusted=... burst=... gcd=... keys=...` at boot in `BotWorker.run()`. | M9 | ORIGINAL_REQUEST §R3 |
| 17 | R4.2 Auto Mapped Counter Persistence | Rehydrate `saved_nodes_count` across restarts. | M9 | ORIGINAL_REQUEST §R4 |
| 18 | R4.3 BEKLEMEDE Lock & Stuck Recovery | Pass actual player positions to `ObstacleMemory` to eliminate false stuck evasions and 5m cooldowns. | M9 | ORIGINAL_REQUEST §R4 |
| 19 | R5.3 Gather Combat Suppression | Strictly suppress combat skills and `fight_back` in `ROUTE_MODE`. | M9 | ORIGINAL_REQUEST §R5 |
| 20 | R6.3 Hotkey & Test Validation | Fix test thresholds, verify Interception, run smoke tests, validate opsec. | M10 | ORIGINAL_REQUEST §R6, R7 |
| 21 | R2.4 Human Gate on All Stems | Remove `is_learn_stem` bypass in `label_trust.py`; enforce human approval for all train/val labels. | M11 | ORIGINAL_REQUEST Follow-up §A |
| 22 | R2.5 Genuine Detection Only | Eliminate fixed-coord fake boxes in `omni_game_learner.py` and synthetic random formulas in `yolo_factory_dialog.py`. | M11 | ORIGINAL_REQUEST Follow-up §B |
| 23 | R2.6 Active Exemplar / Few-Shot Memory | Teach AI from user approvals/rejections via crop feature memory in `dataset/learned_exemplars.json`. | M11 | ORIGINAL_REQUEST Follow-up §C |
| 24 | R2.7 Comprehensive Calibration Wizard | Expand targets in `ui/calibration_wizard.py` to cover all targets (3D crest, nodes, mobs, HP/skill bars) with exemplar saving and auto-calibration. | M12 | ORIGINAL_REQUEST Calibration Follow-up |
| 25 | R1.1 Live Data Binding (3 Pages) | Replace hardcoded placeholder strings and mock prototype data across `ui/app_shell.py`, `ui/settings_window.py`, and `ui/analytics_dialog.py` with live operational data from managers, monitors, route nodes, YOLO logs, settings, and analytics, with clean idle defaults. | M_R1 | ORIGINAL_REQUEST 2026-11-07T19:08:10Z §R1 |
| 26 | R2.1 Floating HUD Overlay Repair | Fix layout squishing, overlapping elements, and clipped text on all 8 action buttons (`Başlat`, `Menü`, `Nokta`, `Durdur`, `Devam`, `Savaş`, `Toplam`, `Kapat`) across minimum and default window sizes in `ui/overlay_hud.py` / `ui/overlay_window.py`. | M_R2 | ORIGINAL_REQUEST 2026-11-07T19:08:10Z §R2 |
| 27 | R3.1 YOLO Preview Popup Annotations | Ensure popup image preview dialog in `ui/yolo_panel.py` and `ui/yolo_factory_dialog.py` renders bounding boxes, class names, and confidence scores overlaid on detected targets. | M_R3 | ORIGINAL_REQUEST 2026-11-07T19:08:10Z §R3 |
| 28 | R4.1 Automated PySide UI Verification | Headless smoke tests (`test_live_ui_smoke.py`) verifying live data bindings, no clipped button text, non-overlapping HUD layout, and YOLO preview popup annotations. | M_TEST | ORIGINAL_REQUEST 2026-11-07T19:08:10Z §Verification |
| 29 | S1.1 Proximity Route Builder | Filter nodes by material and sort via Nearest-Neighbor + 2-Opt TSP proximity optimization from player start position. | M_S1 | ORIGINAL_REQUEST 2026-11-07T20:16:21Z §S1 |
| 30 | S2.1 Selective Gathering Engine | Filter 3D viewport candidates and route waypoints so bot only targets and harvests selected materials (e.g. Orichalcum). | M_S2 | ORIGINAL_REQUEST 2026-11-07T20:16:21Z §S2 |
| 31 | S3.1 Selective Gather Config & UI | Add selective gathering schema fields, F9 / Settings selector controls, and main.py signal routing. | M_S3 | ORIGINAL_REQUEST 2026-11-07T20:16:21Z §S3 |
| 32 | S4.1 Map & Radar Selective Visuals | Highlight matching nodes, dim non-matching, draw numbered proximity badges (#1, #2...) and directional arrows on map and radar. | M_S4 | ORIGINAL_REQUEST 2026-11-07T20:16:21Z §S4 |
| 33 | S5.1 Select & Gather Master Test Suite | Comprehensive 5-agent test suite verifying filtering, TSP distance optimization, map rendering, and live bot selective gather. | M_S5 | ORIGINAL_REQUEST 2026-11-07T20:16:21Z §S5 |
| 34 | W24.1 QComboBox Dropdown White Text | Fix dark-on-dark unreadable text in QComboBox dropdown popups on Windows Qt6 to crisp white (#ffffff). | M_W24 | ORIGINAL_REQUEST 2026-11-07T20:16:25Z |
| 35 | K1.1 Dynamic Filter Action Buttons | Isolate bulk actions strictly to filtered review images and update button text dynamically with exact filter context and item counts. | M_K1 | ORIGINAL_REQUEST 2026-11-07T20:28:58Z §K |
| 36 | Y1.1 Reliable A/R Review Shortcuts | Implement reliable QShortcut and keyPressEvent for 'A' (onayla) and 'R' (reddet) guarded when text inputs have focus. | M_Y1 | ORIGINAL_REQUEST 2026-11-07T20:48:28Z §1 |
| 37 | Y1.2 Non-Freezing Bulk Deletion & Batching | Batch file deletions without per-item UI rebuilds, chunk with progress updates / background worker, refresh UI once at end. | M_Y1 | ORIGINAL_REQUEST 2026-11-07T20:48:28Z §2 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Worker 1: GPU Throttle Hardening | `core/game_gpu_throttle.py` | none | DONE |
| M2 | Worker 2: Dataset QA & Active Learning | `dataset_factory/label_trust.py`, `dataset_factory/dataset_validator.py` | none | DONE |
| M3 | Worker 3: YOLO Factory & Publishing | `ui/yolo_factory_dialog.py` | M1, M2 | DONE |
| M4 | Worker 4: Combat Target & Skill Bar | `core/combat_recovery.py`, `core/skill_bar_reader.py` | none | DONE |
| M5 | Worker 5: Route Manager Sequencing | `core/route_manager.py` | none | DONE |
| M6 | Worker 6: Map Viewer Route Persistence | `ui/map_viewer.py` | M5 | DONE |
| M7 | Worker 7: Cartography AI Ingestion | `core/auto_cartographer.py`, `core/explorer_mode.py` | none | DONE |
| M8 | Worker 8: Input Driver Interception & Jitter | `core/input_driver.py`, `core/human_timing.py` | none | DONE |
| M9 | Worker 9: State Machine Integration & Knobs | `core/state_machine.py`, `config/settings.json` | M4, M5, M7, M8 | DONE |
| M10 | Worker 10: Test Suite & End-to-End Verification | Root test files, `tools/run_all_smoke_tests.py` | M1-M9 | DONE |
| M11 | Worker 11: Active Exemplar Learning & Label QA | `dataset_factory/label_trust.py`, `core/omni_game_learner.py`, `ui/yolo_factory_dialog.py`, `dataset_factory/exemplar_memory.py` | M2, M3 | DONE |
| M12 | Worker 12: Calibration Wizard Comprehensive Targets | `ui/calibration_wizard.py` | M11 | DONE |
| M_R1 | Worker R1: Live Data Binding (3 Pages) | `ui/app_shell.py`, `ui/settings_window.py`, `ui/analytics_dialog.py`, `core/system_monitor.py` | none | DONE |
| M_R2 | Worker R2: Floating HUD Overlay Layout Repair | `ui/overlay_hud.py`, `ui/overlay_window.py` | none | DONE |
| M_R3 | Worker R3: YOLO Preview Popup Annotations | `ui/yolo_panel.py`, `ui/yolo_factory_dialog.py` | none | DONE |
| M_TEST | Worker R4: Automated PySide UI Verification Suite | `test_live_ui_smoke.py` | M_R1, M_R2, M_R3 | DONE |
| M_S1 | Worker S1: Material Node Filter & Proximity Builder | `core/route_manager.py`, `core/sequential_router.py` | none | DONE |
| M_S2 | Worker S2: Selective Gathering & Viewport Harvest | `core/state_machine.py`, `core/material_scanner.py` | none | DONE |
| M_S3 | Worker S3: Select Gather Config UI Wiring | `config/schema.py`, `ui/f9_menu.py`, `ui/settings_window.py`, `main.py` | none | DONE |
| M_S4 | Worker S4: Map & Radar Proximity Visuals | `ui/map_viewer.py`, `ui/cartography_radar.py` | none | DONE |
| M_S5 | Worker S5: Select & Gather Master Test Suite | `tests/test_select_and_gather_5_agent_suite.py` | M_S1-M_S4 | DONE |

| M_W24 | Worker W24: QComboBox Dropdown White Text Fix | `ui/yolo_panel.py`, `ui/shell_theme.py` | none | DONE |
| M_K1 | Worker K1: Dynamic Filter Action Buttons | `ui/yolo_factory_dialog.py` | none | DONE |

| M_Y1 | Worker Y1: YOLO Shortcuts & Non-Freezing Bulk Actions | `ui/yolo_factory_dialog.py`, `tests/test_yolo_factory_shortcuts_and_bulk.py` | none | IN_PROGRESS |








## Code Layout & Strict Write Ownership
Strict file boundary matrix for the 10 parallel workers:
- **Worker 1**: `core/game_gpu_throttle.py` (exclusive)
- **Worker 2**: `dataset_factory/label_trust.py`, `dataset_factory/dataset_validator.py` (exclusive)
- **Worker 3**: `ui/yolo_factory_dialog.py` (exclusive)
- **Worker 4**: `core/combat_recovery.py`, `core/skill_bar_reader.py` (exclusive)
- **Worker 5**: `core/route_manager.py` (exclusive)
- **Worker 6**: `ui/map_viewer.py` (exclusive)
- **Worker 7**: `core/auto_cartographer.py`, `core/explorer_mode.py` (exclusive)
- **Worker 8**: `core/input_driver.py`, `core/human_timing.py` (exclusive)
- **Worker 9**: `core/state_machine.py`, `config/settings.json` (exclusive)
- **Worker 10**: `test_false_action_gates.py`, `test_game_action_bus.py`, `test_combat_select_enters_attack.py`, `test_yolo_factory_gates.py`, `tools/run_all_smoke_tests.py` (exclusive)
- **Worker R1**: `ui/app_shell.py`, `ui/settings_window.py`, `ui/analytics_dialog.py`, `core/system_monitor.py` (exclusive)
- **Worker R2**: `ui/overlay_hud.py`, `ui/overlay_window.py`, `test_overlay_hud_layout.py` (exclusive)
- **Worker R3**: `ui/yolo_panel.py`, `ui/yolo_factory_dialog.py` (exclusive)
- **Worker R4 (Test Writer)**: `test_live_ui_smoke.py` (exclusive)
- **Worker R2_B (HUD Hardening)**: `ui/overlay_hud.py`, `ui/overlay_window.py`, `test_overlay_hud_layout.py`, `tests/test_overlay_hud_adversarial.py` (exclusive)
- **Worker R3_B (ComboBox & Defenses)**: `ui/yolo_panel.py`, `ui/shell_theme.py`, `core/system_monitor.py`, `ui/app_shell.py` (exclusive)
- **Worker Y1**: `ui/yolo_factory_dialog.py`, `tests/test_yolo_factory_shortcuts_and_bulk.py` (exclusive)

## Interface Contracts
### `core/game_gpu_throttle.py` ↔ `ui/yolo_factory_dialog.py`
- `find_aion_game_processes() -> List[int]`: Returns only valid Aion 2 client PIDs with visible windows and matching executables (`Aion2.exe`, `AION.bin`, etc.). Never returns browsers, IDEs, or Python.
- `is_game_running() -> bool`: True only when verified game PID is active.

### `dataset_factory/label_trust.py` ↔ `ui/yolo_factory_dialog.py`
- `is_dataset_ready_for_training(dataset_dir) -> Tuple[bool, str]`: Enforces human review completion (`pending == 0`) and runs QA validation.
- `mark_model_stale(reason: str) -> None`: Sets stale status in metadata.

### `core/route_manager.py` ↔ `ui/map_viewer.py`
- `save_region(region_name: str)`: Persists `_regions[region_name]["nodes"]` to `region_data.json` while maintaining exact sequence order.
- `get_next_ready_node(target_region, target_route) -> NodeData`: Returns next node based on sequential index within route, looping back to start.

### `core/input_driver.py` ↔ `core/state_machine.py`
- `key_down(key: str) -> bool` & `key_up(key: str) -> bool`: Uses Interception kernel driver; returns immediately on success without double-injecting soft input.
- `is_keyboard_trusted() -> bool`: Returns True when Interception or hardware HID is active.
