# VioLENS Project Documentation

## 1. Overview

VioLENS is a Python violence-detection project with two practical paths: a Flask website that monitors a webcam or video source, and a standalone webcam script. Its inference pipeline detects people with YOLO, associates detections across frames with DeepSORT, extracts pose landmarks with MediaPipe, and classifies a 20-frame pose sequence with a trained Keras model. The project also contains code and notebooks for generating pose data and training/evaluating models. The web path is the documented primary entry point; the training notebooks are separate workflows.

The model input described in the repository is 20 frames × 132 values per frame (33 landmarks × x, y, z, visibility). The website selects the front-camera Transformer model or the CCTV model according to the submitted camera angle. See [README.md](README.md).

## 2–3. Files and folders

### Project root

| Path | Purpose |
| --- | --- |
| `README.md` | Short project introduction and setup instructions. |
| `PROJECT_DOCUMENTATION.md` | This architecture and usage guide. |
| `requirements.txt` | Python dependency pins for Windows/Python 3.11. |
| `.gitattributes` | Marks `.keras` and `.csv` files for Git LFS. |
| `LICENSE` | Project license text. |
| `yolov8n.pt` | YOLOv8 nano detection weights used by the root-level workflow. |
| `realtime.py` | Standalone webcam entry point; loads the front Transformer and repeatedly processes camera frames. |
| `person_tracker.py` | Root-level pose/person tracking implementation used by the data/runtime scripts; see its code for its specific constructor and return contract. |
| `pose_data_generation.py` | Extracts pose data from images/video and writes CSV training data. |
| `models/` | Root copies of the trained Keras models (retrieved through Git LFS). |
| `dataset/` | Training CSVs separated by front-camera/CCTV and violence/non-violence classes; Git LFS assets. |
| `samples/data_generation/` | Small sample images organized by class for data-generation workflows. |
| `notebooks/poselstm.ipynb` | Interactive LSTM model training/evaluation workflow. |
| `notebooks/posetransformer.ipynb` | Interactive Transformer model training/evaluation workflow. |
| `deep_sort/` | DeepSORT implementation and its PyTorch appearance model, utilities, configuration, and checkpoint. |

### Website

| Path | Purpose |
| --- | --- |
| `Website/main.py` | Script entry point. Sets a project-local Ultralytics settings path, creates the Flask app, and runs the development server. |
| `Website/Website/__init__.py` | Builds/configures Flask, SQLAlchemy, blueprints, login manager, and database tables. |
| `Website/Website/views.py` | Public/authenticated pages, monitoring routes, model selection/loading, MJPEG stream, detection loop, snapshots, and persistence. |
| `Website/Website/auth.py` | Signup, login, logout, account update, and dashboard routes. Passwords are bcrypt-hashed. |
| `Website/Website/models.py` | SQLAlchemy `User` and `Detection` table definitions. |
| `Website/Website/person_tracker.py` | Website detector: YOLO person boxes → DeepSORT tracks → MediaPipe poses → buffered Keras predictions; returns an annotated frame and latest label. |
| `Website/Website/models/` | Website inference models: `pose_transformer.keras`, `pose_violence_detection_cctv.keras`, and `pose_front_lstm.keras` (Git LFS). |
| `Website/Website/yolov8n.pt` | Website-local YOLO weights. |
| `Website/Website/deep_sort/` | Website-local copy of the DeepSORT source, config, and checkpoint required by the tracker. |
| `Website/Website/templates/` | Jinja pages: `base.html` shared layout; `index.html` landing page; `signup.html`/`login.html` account forms; `home.html` user home; `dashboard.html` account/detection dashboard; `real_time_detection.html` monitoring form/page; `test.html` stream display. |
| `Website/instance/database.db` | SQLite database created/used by Flask-SQLAlchemy. |
| `Website/static/screenshots/` | Detection snapshot output directory. |
| `Website/violens-web.stdout.log`, `Website/violens-web.stderr.log` | Existing local run logs; not required for startup. |

`Website/Website/deep_sort/` mirrors the root `deep_sort/` package for the website's own import paths. It contains the same types of modules listed below.

### DeepSORT modules (both copies)

| Path | Purpose |
| --- | --- |
| `deep_sort/deep_sort.py` | High-level DeepSORT interface: feature extraction, detection construction, and tracker updates. |
| `deep_sort/deep/feature_extractor.py` | Loads the appearance embedding network and produces descriptors for image crops. |
| `deep_sort/deep/model.py`, `original_model.py` | PyTorch network definitions for appearance features. |
| `deep_sort/deep/train.py`, `test.py`, `evaluate.py` | Training, test, and evaluation utilities for the appearance model. |
| `deep_sort/deep/checkpoint/ckpt.t7`, `original_ckpt.t7` | Pretrained PyTorch weights used by DeepSORT. |
| `deep_sort/sort/detection.py` | Detection representation and bounding-box conversion. |
| `deep_sort/sort/tracker.py` | Maintains tracks and assigns detections over time. |
| `deep_sort/sort/track.py` | Per-object track state and lifecycle. |
| `deep_sort/sort/kalman_filter.py` | Predicts object motion between observations. |
| `deep_sort/sort/nn_matching.py` | Appearance-distance metric and feature gallery. |
| `deep_sort/sort/iou_matching.py` | IoU-based matching cost. |
| `deep_sort/sort/linear_assignment.py` | Assignment solver used to match detections to tracks. |
| `deep_sort/sort/preprocessing.py` | Bounding-box overlap suppression utilities. |
| `deep_sort/utils/` | Parsing, geometry, visualization, logging, evaluation, and input/output helpers. |
| `deep_sort/configs/deep_sort.yaml` | Tracker thresholds and lifecycle settings. |
| `deep_sort/test_webcam.py` | Webcam demonstration for DeepSORT; not the VioLENS classifier entry point. |
| `deep_sort/README.md` | Upstream DeepSORT usage notes. |

`*.pyc`, `__pycache__/`, `.bak`, and notebook checkpoint files are generated/backup artifacts rather than required application inputs.

## 4. Architecture and workflow

```text
Browser form
  → Flask route in Website/Website/views.py
  → load selected .keras model
  → background monitoring_loop(source, user, angle)
  → OpenCV reads each frame
  → Website/Website/person_tracker.py
      YOLO person detection → DeepSORT tracking → MediaPipe pose
      → normalized 33×4 landmarks → per-person 20-frame buffer
      → Keras violence scores → annotated frame and label
  → MJPEG /video_feed response in browser
  → on repeated violence labels: JPG snapshot + Detection database row
```

The model files and YOLO/DeepSORT weights are inference artifacts, not created at application startup. Dataset generation and notebook training are offline workflows: they consume source images/videos or CSV data and produce pose CSVs, trained Keras models, and plots/evaluation output. The website loads a classifier lazily when monitoring starts rather than while importing the Flask app.

## 5. Inputs and outputs

**Website input:** signup information (email, name, password, phone fields); for monitoring, a source string (camera index such as `0` or a video path) and camera angle. Front view chooses the Transformer model; the other view chooses the CCTV model. A camera or readable video file is required for useful live frames.

**Classifier input:** one sequence shaped `(1, 20, 132)` for a single prediction. The tracker only builds this after obtaining pose landmarks for a person across 20 frames.

**Outputs:** browser pages and an annotated multipart JPEG stream; model labels such as `Violence`/`NonViolence`; SQLite `Detection` entries; and JPEG snapshots under `Website/static/screenshots/` after the monitoring logic's repeated-violence threshold is reached. Offline scripts/notebooks write CSVs, model files, and evaluation results according to their code.

## 6. Important functions and classes

| Symbol | Inputs and processing | Output/dependencies |
| --- | --- | --- |
| `Website.main.app` | Created at import by calling `create_app()`. | Flask app consumed by Flask's development server. |
| `create_app()` | Configures Flask/SQLAlchemy; registers `views` and `auth`; configures login and database. | Configured Flask app; uses `db`, `User`, `Detection`, blueprints. |
| `create_database(app)` | Opens app context and invokes `db.create_all()`. | Creates missing SQLite tables; safe to call repeatedly. |
| `User`, `Detection` | SQLAlchemy columns describe accounts and saved detections. | ORM records stored in SQLite. |
| `auth.login()` / `signup()` / `logout()` | Read form fields; query/update user records; bcrypt checks or hashes password. | Rendered page, redirect, flash message, or database update. |
| `views.monitor()` | Accepts source and camera-angle form fields; loads chosen model; starts a monitoring thread. | Monitoring page or 503 message if the chosen model cannot load. |
| `views.video_feed()` / `generate_stream()` | Reads shared latest frame, JPEG-encodes it, yields multipart chunks. | Browser-compatible MJPEG stream. |
| `load_pose_model(camera_angle)` | Selects one of two website model paths and loads/caches it with Keras. | Cached Keras model; missing file raises `FileNotFoundError`. |
| `monitoring_loop(source, username, alert_phone, cam_view, user_id)` | Opens OpenCV source; builds tracker; processes frames; counts labels and saves snapshots/detection metadata. | Updates shared frame and detection buffer; uses model, tracker, OpenCV, SQLAlchemy-backed persistence. |
| `start_periodic_commit(app, interval)` | Starts a worker/timer to transfer queued detection metadata into ORM rows. | Database commits on a background thread. |
| `PersonTracker.__init__(...)` | Loads YOLO and DeepSORT weights, initializes MediaPipe Pose and per-ID pose buffers. | Ready-to-use tracker. |
| `PersonTracker.update(frame, violence_model)` | Detects people, updates tracks, extracts/normalizes pose, runs model when a track buffer is full, annotates boxes/labels. | `(annotated_frame, latest_label)`; uses YOLO, DeepSORT, MediaPipe, OpenCV, NumPy. |
| `PersonTracker.normalize_pose_with_visibility(vector)` | Reshapes 132 values, centers/scales XYZ using shoulder/hip landmarks, retains visibility. | Normalized 132-value vector. |
| `realtime.main(camera_index=0)` | Loads front Transformer, opens camera, calls tracker repeatedly until `q` or stream end. | Shows an OpenCV window; raises a clear error if camera cannot open. |
| DeepSORT `Tracker`, `Track`, `Detection` | Accept detections/embeddings and motion state; associate objects over frames. | Stable track IDs and updated track state. |
| `pose_data_generation.py` functions | Read source media/images, call MediaPipe pose, format landmarks, and save rows/labels. | CSV pose data; exact options and return values are described in that script. |

## 7. Libraries

| Library | Use |
| --- | --- |
| Flask, Flask-Login | HTTP routes, templates, sessions, authentication. |
| Flask-SQLAlchemy, SQLAlchemy | ORM and SQLite persistence. |
| bcrypt | Password hashing and verification. |
| TensorFlow/Keras | Loads and runs pose-sequence classifiers. |
| OpenCV (`opencv-python`, `opencv-contrib-python`) | Capture/read video, draw annotations, encode streams, write snapshots. |
| Ultralytics | YOLOv8 person detection. |
| MediaPipe | Per-person pose landmarks. |
| PyTorch / torchvision | DeepSORT appearance feature network and related ML utilities. |
| NumPy | Pose vector/sequence manipulation. |
| pandas | CSV/dataframe workflows. |
| SciPy, motmetrics | Assignment/statistical utilities and tracking evaluation. |
| matplotlib | Notebook plots. |
| PyYAML, easydict | DeepSORT configuration parsing. |

## 8. Data flow

1. OpenCV obtains a frame from the selected camera/video.
2. YOLO returns person boxes and confidence scores.
3. DeepSORT assigns persistent IDs to detections.
4. MediaPipe processes each person's crop and returns 33 landmarks. Each landmark is represented by x/y/z/visibility.
5. The website tracker centers/scales pose coordinates and buffers 20 frames per track.
6. Keras returns class scores for the sequence; the tracker draws its latest label on the frame.
7. The monitoring loop shares the frame with the MJPEG generator. Repeated violence detections cause a snapshot and queued metadata, which the background commit helper stores in SQLite.

## 9–10. Setup and run (Windows)

Python 3.11 is the pinned environment used for this checkout. From the repository root:

```powershell
git lfs install
git lfs pull --include="Website/Website/models/*.keras"
conda create -n violens python=3.11 pip -y
conda activate violens
python -m pip install -r requirements.txt
Set-Location .\Website
python main.py
```

Then open `http://127.0.0.1:5000`, sign up, choose a source and camera angle, and start monitoring. To use the standalone OpenCV window instead, run `python ..\realtime.py` from `Website`'s parent project directory (`python realtime.py` at project root); press `q` to quit. Ensure a camera is connected or change `camera_index` in `realtime.py`.

The code may be launched from `Website` as shown because its database, model, and YOLO paths are designed for that entry point. Git LFS is needed to retrieve `.keras` model contents. The environment already used in this workspace was named `violens`; on another machine choose any environment name.

## 11. Expected behavior

The landing page should return HTTP 200. Login/signup pages render forms. After signup/login, the monitoring UI can select webcam index `0` or a local video path. During monitoring, the browser shows an MJPEG feed with person boxes, IDs, pose marks, and classifier labels after a track has accumulated enough valid pose frames. On enough consecutive violence-positive sequence results, a screenshot and detection record are created. A static image or a frame with no visible person may produce no label; predictions require a person pose sequence.

## 12. Common errors

| Symptom | Cause / solution |
| --- | --- |
| `ModuleNotFoundError` on startup | Activate the environment with installed requirements; run from `Website` using `python main.py`. |
| Model not found or file is a tiny text pointer | Install Git LFS and run `git lfs pull --include="Website/Website/models/*.keras"` from the repository root. |
| Keras cannot deserialize `keras.src...` model classes | Use the pinned TensorFlow 2.16.1 / Keras 3 environment in `requirements.txt`; do not run the project in an older TensorFlow/Keras environment. |
| Camera cannot open / stream is blank | Close other apps using the camera, check Windows camera permissions, try index `1`, or provide a valid video path. |
| YOLO weights not found | Start via `Website/main.py` from the `Website` folder; ensure `Website/yolov8n.pt` exists. |
| DeepSORT checkpoint error | Ensure `Website/Website/deep_sort/deep/checkpoint/ckpt.t7` is present and the process starts from `Website`. |
| TensorFlow/Ultralytics first-run delay or CPU messages | Initial imports and XLA compilation can take time; the CPU build is supported but slower than a supported GPU setup. |
| SQLite permission/path errors | Ensure the process can write under `Website/instance/`; `create_app()` creates missing tables there. |
| No violence label on a still image | Classification requires 20 pose frames for a tracked person; use a video/camera sequence with a visible person. |

## Scope note

This guide describes behavior present in the checked-out source. It does not claim that training accuracy in the README represents independent real-world accuracy. The project includes local test/demo and training utilities; the Flask app is the main deployable workflow documented here.
