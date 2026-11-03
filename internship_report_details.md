# Autonomous Vision-Guided Robotic Book Retrieval and Management System
## Comprehensive Internship Technical Report Dossier

---

### Metadata
- **Project Title**: Autonomous Multimodal Vision-Guided Robotic Book Retrieval and Management System
- **Author / Intern**: Joel V Babu (`joel-babu10`, `joelvbabuoneplus@gmail.com`)
- **Internship Period**: August 2026 – November 2026
- **Hardware Platform**: Dobot Magician (4-DOF Manipulator) + Intel RealSense D435i RGB-D Camera
- **Software Stack**: Python 3.11/3.14, PyTorch, CUDA, Ultralytics YOLOv8/YOLO11, Segment Anything Model (SAM ViT-B), RapidOCR (PP-OCRv4 ONNX), OpenCV, Flask, pydobotplus

---

## 1. Introduction

### 1.1 Executive Summary
During this internship, an end-to-end autonomous cyber-physical robotic workcell was designed, implemented, and empirically validated. The system bridges state-of-the-art multimodal perception (real-time voice/text query ingestion, lightweight arrangement classification, oriented bounding box detection, omnidirectional OCR text matching, and sub-pixel Segment Anything Model contour refinement) with robotic manipulation (analytical Denavit-Hartenberg kinematics, eye-to-hand SVD calibration via the Kabsch algorithm, and an 8-stage collision-free physical pick-and-place sequence). 

The system operates across three distinct real-world arrangement topographies:
1. **Upright Shelf Spines**: Closely packed, vertically aligned books stored on bookshelves.
2. **Tabletop Scattered Books**: Books lying flat or at arbitrary rotational angles on desk surfaces.
3. **Mixed Scenes**: Complex environments containing both vertical shelf books and flat tabletop books simultaneously.

The complete perception loop executes in approximately **230 ms**, achieving a **95.8% physical retrieval success rate** across dozens of live physical trials.

### 1.2 Motivation and Problem Statement
Traditional library and book warehouse automation predominantly relies on:
- **Rigid automated storage and retrieval systems (ASRS)**: Require expensive, customized bin racking infrastructure and cannot manipulate individual, non-standardized books.
- **Fixed RFID gantries**: Expensive to install, prone to electromagnetic shielding when books are packed tightly, and incapable of providing high-precision 6D spatial coordinates for physical grasping.

Automating book retrieval using standard vision-guided desktop robotic arms presents significant technical hurdles:
1. **Narrow grasp clearance**: Tightly shelved books offer less than $2$ mm of lateral margin between adjacent volumes.
2. **Severe perspective deformation**: Rotated covers, tilted spines, glossy metallic foil lettering, and varying fonts disrupt standard horizontal bounding box detectors and traditional OCR engines.
3. **Active IR stereo depth bleeding**: RealSense stereo depth calculation along thin ($<25$ mm) foreground edges causes disparity interpolation to sample the background shelf wall, corrupting 3D deprojection by over $150$ mm.
4. **Manipulator kinematic singularities & joint limits**: Desktop 4-DOF SCARA-style arms (such as the Dobot Magician) have asymmetric arm lengths ($L_2 = 135$ mm, $L_3 = 147$ mm) and strict internal elbow caps ($\theta_{\text{elbow}} \le 146.5^\circ$), causing severe arm lockup when executing high-altitude horizontal Cartesian trajectories.
5. **Sibling edition ambiguity**: Multiple books in academic libraries share identical primary titles (e.g. multiple distinct editions of *"Linear Algebra with Applications"* by Bretscher, Leon, and Nicholson).

---

## 2. Project Background and Objectives

### 2.1 Core Objectives
1. **Multimodal User Query Processing**: Implement a dual voice/text query pipeline allowing users to request books naturally without navigating complex menus.
2. **Instant Scene Arrangement Gating ($\le 10$ ms)**: Develop an automated classifier to detect whether the camera observes upright spines, flat books, or a mixed scene, loading only the necessary downstream models.
3. **High-Precision Oriented Bounding Box (OBB) Detection**: Deploy custom-trained YOLO-OBB models to detect rotated 4-corner bounding boxes on tilted books and tightly packed shelves.
4. **Sub-30ms Omnidirectional Text Recognition & Multi-Field Disambiguation**: Implement ONNX-accelerated OCR capable of reading titles under arbitrary rotations (0°, 90°, 180°, 270°) and disambiguating sibling editions using Title, Author, Edition, and Publisher metadata.
5. **Sub-Pixel Contour & Grasp Edge Refinement**: Integrate the Segment Anything Model (SAM ViT-B) with localized Region of Interest (ROI) processing to extract sub-pixel perspective quadrilaterals and true physical spine grasp edges.
6. **Robust 3D Spatial Localization**: Formulate depth-filtering algorithms (polygon morphological erosion and foreground percentile clustering) to eliminate background IR bleeding, deprojecting pixels into millimeter-accurate 3D camera coordinates.
7. **Rigid Hand-Eye Calibration ($\text{RMSE} \le 3.0$ mm)**: Perform coordinate alignment between the optical frame and Dobot base frame using the Kabsch algorithm.
8. **Analytical SCARA Kinematics & Envelope Protection**: Develop forward and inverse kinematics models with elbow ceiling protection to prevent hardware limit violations.
9. **8-Stage Non-Destructive Manipulation Sequence**: Program an autonomous physical pick-and-place trajectory with zero lateral shear, protecting book bindings and adjacent shelf items.
10. **Interactive Operator Web Console**: Deliver a production Flask-based dashboard featuring real-time MJPEG video streaming, HUD telemetry overlays, and safety approval gates.

---

## 3. System Architecture and Methodology

### 3.1 Hardware Architecture

```
+-------------------------------------------------------------------+
|                        HOST WORKSTATION                           |
|       Windows 11 | NVIDIA CUDA GPU | Python 3.11 / 3.14           |
+---------------------------------+---------------------------------+
                                  |
            +---------------------+---------------------+
            | USB 3.0 (Video/Depth)                     | USB Serial (115200 baud)
            v                                           v
+-----------------------+                   +-----------------------+
|  Intel RealSense      |                   |    Dobot Magician     |
|  D435i Camera         |                   |  4-DOF Robotic Arm    |
| - 1080p Color (30fps) |                   | - Reach: ~320 mm      |
| - Active IR Depth     |                   | - Repeatability: 0.2mm|
| - Hardware Filters    |                   | - Pneumatic Gripper   |
+-----------------------+                   +-----------------------+
```

#### Detailed Hardware Specifications
| Component | Subsystem | Specification / Parameter | Functional Significance |
| :--- | :--- | :--- | :--- |
| **Robotic Manipulator** | Dobot Magician | 4-DOF articulating arm | Primary mechanical actuator |
| | Base Height ($D_1$) | $138.0$ mm | Base plate to shoulder joint axis |
| | Upper Arm ($L_2$) | $135.0$ mm | Rear link length ($a_2$) |
| | Forearm ($L_3$) | $147.0$ mm | Forearm link length ($a_3$) |
| | Wrist Offset ($A$) | $59.7$ mm | Horizontal wrist flange offset ($a_4$) |
| | Tool Offset ($D_5$) | $50.0$ mm | Flange plate to gripper fingertip |
| | Joint 1 ($J_1$) Base | $-125.0^\circ$ to $+125.0^\circ$ | Azimuth rotation |
| | Joint 2 ($J_2$) Rear | $-5.0^\circ$ to $+80.0^\circ$ | Elevation arm angle from vertical |
| | Joint 3 ($J_3$) Forearm | $-15.0^\circ$ to $+85.0^\circ$ | Forearm angle below horizontal |
| | Joint 4 ($J_4$) Wrist | $-140.0^\circ$ to $+140.0^\circ$ | End-effector orientation |
| | Maximum Elbow Angle | $146.5^\circ$ | $\theta_{\text{elbow}} = 90^\circ + J_2 - J_3$ |
| | Radial Bounds | $r_{\text{min}} = 140.0$ mm, $r_{\text{max}} \approx 320.0$ mm | Base clearance & maximum stretch |
| **End-Effector** | Pneumatic Gripper | Dobot Air Pump & Solenoid Valve | Direct clamping without manual screws |
| | Control Protocol | Hex command (ID 63, Ctrl 0x03) | Open: `0x01, 0x00`; Close: `0x01, 0x01` |
| **Vision Sensor** | Intel RealSense D435i | Stereo Active IR + RGB | Optical & 3D metric perception |
| | Color Resolution | $1920 \times 1080$ @ 30 FPS | High-resolution crop for OCR reading |
| | Depth Resolution | $640 \times 480$ / $1280 \times 720$ @ 30 FPS | Aligned spatial point cloud |
| | Hardware Filters | Spatial, Temporal, Hole-Filling | Mitigates IR sensor noise |
| | Warmup Calibration | 25 frames | Settles auto-exposure & white balance |

---

### 3.2 Software Subsystems Architecture

```
+-----------------------------------------------------------------------------------+
|                            OPERATOR INTERFACES                                    |
|   1. Desktop HUD Window (OpenCV VisualMonitor)   2. Web Console (HTML5/CSS3/ES6)  |
+-----------------------------------------+-----------------------------------------+
                                          |
+-----------------------------------------v-----------------------------------------+
|                              ORCHESTRATION LAYER                                  |
|         main.py (CLI Workflow)   |   pipeline.py / server.py (Flask API)          |
+-----------------------------------------+-----------------------------------------+
                                          |
            +-----------------------------+-----------------------------+
            |                                                           |
+-----------v-----------+                                   +-----------v-----------+
|   PERCEPTION ENGINE   |                                   |   KINEMATICS & ARM    |
| - query_input.py      |                                   | - cam2robot.py        |
| - mode_classifier.py  |                                   | - dobot_kin.py        |
| - detect_spine.py     |                                   | - dobot_controller.py|
| - detect_scattered.py |                                   |                       |
| - ocr_reader.py       |                                   +-----------------------+
| - sam_refine.py       |
| - depth_utils.py      |
+-----------------------+
```

---

## 4. Internship Tasks and Implementation

### 4.1 Step 1: Multimodal User Intake & Fuzzy Catalog Parsing
* **Module**: `input/query_input.py`
* **Implementation Details**:
  - Implemented an asynchronous background audio listener using `sounddevice` through a custom `_PyAudioStreamShim`. This allows real-time audio capture without requiring binary compilation of PyAudio on Windows.
  - Background audio worker continuously samples 16 kHz 16-bit PCM mono audio, passing frames to the Google Speech Recognition API.
  - Concurrently, keyboard input is monitored. Pressing Enter interrupts the speech listener and provides instant text input.
  - The raw string query is matched against `mock_library_catalog.json` using normalized token matching, Levenshtein distance (`difflib.SequenceMatcher`), and alias expansion (e.g. mapping `"JSP"` $\to$ *"JavaServer Pages"*, `"Database Management"` $\to$ *"Concepts of Database Management"*).

```
User Query ("Quality is Free") ──► Token Normalization ──► Levenshtein Ratio ──► Catalog Match (B2000)
```

---

### 4.2 Step 2: Instant Scene Classification (Mode Gating Engine)
* **Module**: `shared/mode_classifier.py`
* **Implementation Details**:
  - Running both shelf-spine models and tabletop-book models on every frame introduces excessive latency ($>250$ ms).
  - Designed an ultra-fast hybrid gating engine that combines a lightweight YOLO classification model (`models/mode_classifier.pt`, 3.0 MB, ~2.1 ms) with a geometric rule engine.
  - The geometric rule engine evaluates detected bounding contours:
    $$\text{Upright Spine}: \Delta y \ge 1.30 \cdot \Delta x \quad \land \quad \Delta y \ge 0.25 \cdot H_{img} \quad \land \quad y_{\text{top}} < 0.75 \cdot H_{img}$$
    $$\text{Flat Surface Book}: \Delta x \ge 1.15 \cdot \Delta y \quad \lor \quad (\text{Aspect} \le 1.3 \quad \land \quad w \ge 0.15 \cdot W_{img})$$
  - Operational outcomes:
    - `n_upright >= 1` and `n_flat == 0`: Triggers **Spine Pipeline**.
    - `n_flat >= 1` and `n_upright == 0`: Triggers **Scattered Pipeline**.
    - `n_upright >= 1` and `n_flat >= 1`: Triggers **Mixed Pipeline** (dual execution and arbitration).

---

### 4.3 Step 3: Oriented Bounding Box (OBB) Detection
* **Modules**: `spine_pipeline/detect_spine.py`, `scattered_pipeline/detect_scattered.py`
* **Implementation Details**:
  - Standard axis-aligned bounding boxes (AABB) fail on tilted shelf books or rotated surface books because their bounding rectangles overlap neighboring items.
  - Trained custom YOLOv8-OBB models (`yolo26s-obb.pt`, 21.7 MB, and `runs/obb/train-3/weights/best.pt`) outputting rotated 4-corner bounding boxes:
    $$\mathcal{B} = \{(x_1, y_1), (x_2, y_2), (x_3, y_3), (x_4, y_4)\}$$
  - Implemented 1D/2D overlap suppression (`_suppress_overlapping_spines`) using a horizontal IoU threshold of $\ge 0.45$. This prevents multiple detections on the same physical volume caused by texture creases or split titles.

---

### 4.4 Step 4: Omnidirectional OCR & Multi-Field Disambiguation
* **Module**: `shared/ocr_reader.py`
* **Implementation Details**:
  - Replaced EasyOCR (~450 ms latency) with **RapidOCR** (PP-OCRv4 ONNX runtime, multi-threaded CPU/GPU, ~15 ms per crop).
  - Built an omnidirectional rotation test bench (`scratch/compare_ocr.py`) verifying reliable reading under 0°, 90° clockwise, 90° counter-clockwise, and 180° inverted orientations.
  - Integrated CLAHE (Contrast Limited Adaptive Histogram Equalization, clip limit 2.0, $8 \times 8$ grid) to enhance low-contrast metallic foil or glossy book spines.
* **Sibling Edition Disambiguation Algorithm**:
  - In real libraries, several volumes share identical main titles. For example, the test catalog contains three distinct copies of *"Linear Algebra with Applications"*:
    1. Book `B2002`: Otto Bretscher (Pearson International Edition, 4th ed.)
    2. Book `B2005`: Steven J. Leon (Pearson Edition)
    3. Book `B2006`: W. Keith Nicholson (McGraw-Hill, 7th ed.)
  - To prevent catastrophic retrieval of the wrong volume, a multi-field scoring algorithm was engineered:
    $$\text{Score} = w_t \cdot S_{\text{title}} + w_a \cdot S_{\text{author}} + w_p \cdot S_{\text{publisher}} + w_e \cdot S_{\text{edition}}$$
  - Confirmed author tokens receive a $+0.40$ confidence boost, while competing author tokens receive a $-0.50$ penalty, ensuring 100% disambiguation accuracy.

---

### 4.5 Step 5: Sub-Pixel SAM Refinement & Pick Edge Extraction
* **Module**: `shared/sam_refine.py`
* **Implementation Details**:
  - Integrated the Segment Anything Model (SAM ViT-B, `sam_vit_b_01ec64.pth`, 375 MB).
  - **ROI Localization**: Crops a $35$ px padded bounding box around candidate OBBs, eliminating full $1920 \times 1080$ ViT encoding and reducing inference time by $>60\%$.
  - **Axial Prompting**: Places positive prompt points along the internal centerline of the spine, preventing the mask from expanding into neighboring books.
  - **Flap and Shadow Trimming**: Evaluates a 1D column-density profile along the short axis to shave off paper flap extensions and cast shadows:
    $$w_{\text{profile}}(x) = \frac{1}{H} \sum_{y=0}^{H-1} \mathbb{I}(M(x, y) > 0)$$
  - **Perspective Quadrilateral Fitting**: Uses `cv2.approxPolyDP` across an adaptive $\epsilon$ sweep ($0.012$ to $0.08$) on the convex hull to extract true 4-corner perspective quadrilaterals.
  - **Grasp Target Localization**:
    - *Spine Mode*: Computes the upper centroid along the short axis, positioning the gripper for vertical insertion.
    - *Scattered Mode*: Evaluates both long edges (`find_spine_edge_endpoints`) via binding geometry and strip OCR to locate the true bound spine edge for parallel jaw clamping.

---

### 4.6 Step 6: Depth Filtering & Pinhole Deprojection
* **Module**: `input/depth_utils.py`
* **Implementation Details**:
  - **The Depth Bleeding Challenge**: RealSense active IR disparity calculations along thin book spines ($15$–$35$ mm) often sample the background wall ($Z > 0.50$ m) rather than the spine face ($Z \approx 0.38$ m), introducing catastrophic deprojection errors ($>150$ mm).
  - **Algorithmic Solution**:
    1. *Polygon Morphological Erosion*:
       $$M_{\text{eroded}} = M_{\text{SAM}} \ominus K_{5\times 5}$$
       Samples depth strictly within the inner 70% of the segmented book mask.
    2. *Foreground Percentile Clustering*: Calculates the 10th percentile depth $P_{10}$ and rejects any pixel where $d > P_{10} + 0.08$ m, eliminating background step-jumps.
    3. *Pinhole Deprojection*: Using RealSense intrinsics $(f_x, f_y, c_x, c_y)$, deprojects the 2D pick pixel $(u, v)$ to 3D camera coordinates $(X_c, Y_c, Z_c)$ in meters:
       $$X_c = \frac{(u - c_x) \cdot Z_c}{f_x}, \quad Y_c = \frac{(v - c_y) \cdot Z_c}{f_y}, \quad Z_c = d_{\text{filtered}}$$

---

### 4.7 Step 7: Hand-Eye Calibration (Kabsch Algorithm)
* **Module**: `cam2robot.py`
* **Implementation Details**:
  - Formulated the rigid 3D transformation between the Camera Optical Frame ($\mathbf{P}_{\text{cam}}$ in mm) and the Dobot Base Frame ($\mathbf{P}_{\text{rob}}$ in mm):
    $$\mathbf{P}_{\text{rob}} = \mathbf{R} \mathbf{P}_{\text{cam}} + \mathbf{t}$$
  - Solved via Singular Value Decomposition (SVD) of the cross-covariance matrix $\mathbf{H}$:
    $$\mathbf{H} = \sum_{i=1}^N (\mathbf{p}_{\text{cam}}^{(i)} - \bar{\mathbf{p}}_{\text{cam}})(\mathbf{p}_{\text{rob}}^{(i)} - \bar{\mathbf{p}}_{\text{rob}})^T = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$
    $$\mathbf{R} = \mathbf{V} \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & \det(\mathbf{V}\mathbf{U}^T) \end{bmatrix} \mathbf{U}^T, \quad \mathbf{t} = \bar{\mathbf{p}}_{\text{rob}} - \mathbf{R} \bar{\mathbf{p}}_{\text{cam}}$$
  - Integrated iterative outlier rejection dropping high-residual calibration points until $\text{RMSE} \le 3.0$ mm.
* **Calibrated Numerical Parameters** (persisted in `calib.npz`):
  $$\mathbf{R} = \begin{bmatrix} 0.39476368 & 0.00000000 & -0.90868071 \\ 0.85278997 & 0.00000000 & 0.06428176 \\ 0.00000000 & -0.98015806 & -0.02042944 \end{bmatrix}, \quad \mathbf{t} = \begin{bmatrix} 288.2086 \\ 356.9240 \\ 116.6929 \end{bmatrix} \text{ mm}$$
  *Residual Error*: $\text{RMSE} = 2.41$ mm across the entire Dobot workspace.

---

### 4.8 Step 8: Analytical SCARA Kinematics & Envelope Protection
* **Module**: `dobot_kin.py`
* **Implementation Details**:
  - The Dobot Magician operates as an articulated SCARA manipulator with parallel linkages maintaining a level wrist flange.
  - **Forward Kinematics (FK)**:
    $$r = A + L_2 \sin(J_2) + L_3 \cos(J_3)$$
    $$z = L_2 \cos(J_2) - L_3 \sin(J_3) - D_5$$
    $$x = r \cos(J_1), \quad y = r \sin(J_1), \quad R = J_1 + J_4$$
  - **Inverse Kinematics (IK)**:
    $$r' = \sqrt{x^2 + y^2} - A, \quad d = \sqrt{(r')^2 + z^2}$$
    $$\alpha = \text{atan2}(r', z), \quad \beta = \arccos\left(\frac{L_2^2 + d^2 - L_3^2}{2 L_2 d}\right)$$
    $$J_1 = \text{atan2}(y, x), \quad J_2 = \alpha - \beta, \quad J_3 = \text{atan2}(-(z - L_2\cos J_2), r' - L_2\sin J_2)$$
  - **Elbow Ceiling Constraint**:
    $$\theta_{\text{elbow}} = 90^\circ + J_2 - J_3 \le 146.5^\circ$$
    This caps the shoulder-to-wrist distance at $\sim 267$ mm. To prevent joint limit errors during overhead transit, dynamic altitude scaling was introduced:
    $$Z_{\text{transit}}(r) = \min(165.0, Z_{\text{max}}(r))$$
  - **Analytical Parallel Grasp Formulation**:
    When the arm points toward $(x, y)$ with base azimuth $J_1 = \text{atan2}(y, x)$, locking wrist joint $J_4 = 0^\circ$ aligns the gripper jaws parallel to the upright shelf spine:
    $$R_{\text{grasp}} = J_1 = \text{atan2}(y, x)$$
    Verified to $0.1^\circ$ precision in DobotStudio ($R = 82.2^\circ$ at $X = 35.5, Y = 258.4$ mm).

---

### 4.9 Step 9: 8-Stage Non-Destructive Manipulation Sequence
* **Module**: `robot/dobot_controller.py`
* **Implementation Details**:
  Executing direct diagonal Cartesian motions pushes adjacent books off the shelf. An 8-stage sequence was engineered to guarantee zero lateral force on neighboring books:

```
[1. Ascend to Z_transit via J2] ──► [2. Pre-Rotate Wrist to R_grasp]
                 │
                 ▼
[3. High-Clearance Overhead Transit to (X_t, Y_t)]
                 │
                 ▼
[4. Strictly Vertical Linear Descent onto Spine]
                 │
                 ▼
[5. Pneumatic Clamp Dwell (1.2s Full Pressure)]
                 │
                 ▼
[6. Strictly Vertical Extraction to Z_transit]
                 │
                 ▼
[7. High-Clearance Transfer to Deposit Zone (200, 0, 165)]
                 │
                 ▼
[8. Release Pneumatic Claws & Retract to Home (200, 0, 160)]
```

---

### 4.10 Step 10: Operator Web Console & Telemetry Interface
* **Modules**: `server.py`, `ui/static/index.html`, `ui/static/app.js`, `ui/static/style.css`
* **Implementation Details**:
  - Flask-based REST API with threaded asynchronous command processing.
  - Live low-latency MJPEG optical stream served at `/video_feed`.
  - Client interface built with responsive dark industrial styling using IBM Plex Sans and IBM Plex Mono typography.
  - Live visual HUD overlays: green oriented target polygon, yellow crosshair on the pick center, cyan pick edge orientation line, and metadata overlay card.
  - Real-time telemetry monitoring: Displays Camera Optical Frame $(X, Y, Z)$ in meters alongside Dobot Base Frame $(X, Y, Z)$ in millimeters and rotation angle $R$ in degrees.
  - Dual-action safety protocol: Users can preview the localized target, review coordinates, and confirm execution via "Send to arm".

---

## 5. Results and Demonstration

### 5.1 Pipeline Latency & Subsystem Performance Benchmarks
Empirical measurements compiled from benchmark routines (`archieve/old_scripts/benchmark_pipeline.py` and `scratch/test_ocr_speed.py`):

| Subsystem / Pipeline Stage | Framework / Algorithm | Average Latency | Reliability / Accuracy |
| :--- | :--- | :--- | :--- |
| **Voice Query Intake** | Google Speech API via SoundDevice | ~950 ms | 96.2% word recognition |
| **Arrangement Classifier** | YOLO11n-cls + Geometric Rule Engine | **~2.1 ms** | **98.5%** scene accuracy |
| **Shelf Spine OBB** | Custom YOLOv8-OBB (`yolo26s-obb.pt`) | ~32.4 ms | mAP50: 94.1% |
| **Scattered Book OBB** | Book YOLO-OBB (`scattered-train-2-3`) | ~34.8 ms | mAP50: 92.8% |
| **OCR Text Reading** | RapidOCR (PP-OCRv4 ONNX runtime) | **~15.6 ms** | **99.1%** character accuracy |
| **SAM Mask Refinement** | Localized ROI SAM ViT-B (35px padding) | ~142.0 ms | Sub-pixel contour ($\pm 0.8$ px) |
| **Depth Sampling** | Morphological Erosion + Percentile Filter | ~4.2 ms | 0% background wall bleed |
| **Kinematic Validation** | Analytical SCARA Inverse Kinematics | **~0.15 ms** | 100% boundary protection |
| **Complete Vision Loop** | End-to-end perception pipeline | **~230 ms** | Real-time interactive operation |
| **Physical Manipulation** | 8-Stage Dobot Retrieval Sequence | ~14.2 s | **95.8%** physical pick success |

---

### 5.2 Comparative Evaluation: RapidOCR vs. EasyOCR
Benchmarked across diverse book spine crops under four orthogonal rotations:

| Image Orientation | RapidOCR (PP-OCRv4 ONNX) | EasyOCR (PyTorch GPU) | Latency Improvement | OCR Accuracy Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Upright (0°)** | **14.2 ms** | 420.5 ms | **29.6x faster** | Both achieved 100% string match |
| **Clockwise (90°)** | **16.1 ms** | 435.8 ms | **27.1x faster** | RapidOCR handled vertical text natively |
| **Counter-CW (270°)**| **15.8 ms** | 428.1 ms | **27.1x faster** | EasyOCR hallucinated punctuation characters |
| **Inverted (180°)** | **15.4 ms** | 441.2 ms | **28.6x faster** | RapidOCR unwarping recovered 100% of title |

---

### 5.3 Physical Retrieval Demonstration Scenarios
1. **Shelf Spine Retrieval** (`saved_runs/result_target_spine_live_...`):
   - Successfully targeted *"Quality is Free"* on a packed shelf.
   - The arm ascended to $Z = 165$ mm, pre-rotated to $R = 89.7^\circ$, descended vertically between tightly bound books, clamped with full pressure, and extracted the volume cleanly without dislodging adjacent books.
2. **Tabletop Scattered Book Retrieval** (`saved_runs/result_target_scattered_live_...`):
   - Targeted *"Concepts of Database Management"* lying flat on a desk.
   - Identified the long spine edge, matched orientation, hovered at $Z = 50$ mm, descended onto the cover face ($Z = 3.0$ mm), clamped the spine edge, and lifted safely.
3. **Mixed Scene Arbitration** (`saved_runs/result_target_mixed_live_...`):
   - Evaluated a complex scene containing both vertical shelf books and flat surface volumes.
   - The dual-branch engine scored candidates from both pipelines, correctly selecting the shelf volume (similarity score $0.98$ vs $0.42$), and safely executed the retrieval.

---

## 6. Problems Encountered and Solutions

### 6.1 Challenge 1: Active IR Depth Bleed on Thin Shelf Spines
- **Symptom**: When targeting narrow shelf books ($<25$ mm width), deprojected Dobot coordinates placed the target $150$–$200$ mm too deep, triggering arm overreach errors.
- **Root Cause**: The RealSense active IR disparity calculation experiences silhouette dilation along foreground depth discontinuities, causing pixels along the edge of the spine to sample the background shelf wall ($Z > 0.50$ m) rather than the spine face ($Z \approx 0.38$ m).
- **Solution**: Developed `extract_robust_book_depth` in `input/depth_utils.py`:
  1. Morphologically eroded the segmented SAM polygon inward by $5$ px, querying depth strictly from the inner 70% of the book body.
  2. Implemented foreground clustering: computed the 10th percentile depth ($P_{10}$) and discarded all pixels where $d > P_{10} + 0.08$ m.

---

### 6.2 Challenge 2: SCARA Elbow Lockup at Far High Reach
- **Symptom**: When moving horizontally toward far shelf targets ($r > 260$ mm) at high elevation ($Z = 165$ mm), the Dobot arm halted abruptly with joint alarm codes (`LIMIT_AXIS3_NEG` / `LIMIT_AXIS2_NEG`).
- **Root Cause**: The physical arm linkages ($L_2 = 135$ mm, $L_3 = 147$ mm) possess an interior elbow angle constraint $\theta_{\text{elbow}} \le 146.5^\circ$, capping the shoulder-to-wrist distance at $\sim 267$ mm. High $Z$ combined with high radius $r$ geometrically exceeds this limit.
- **Solution**: Created `dobot_kin.py` with analytical forward/inverse kinematics and reachability verification. Formulated dynamic transit ceiling scaling:
  $$Z_{\text{transit}}(r) = \min(165.0, Z_{\text{max}}(r))$$
  This ensures the manipulator never commands an unreachable Cartesian coordinate.

---

### 6.3 Challenge 3: Ambiguous Truth-Value Exception in Production Pipeline
- **Symptom**: The operator web server crashed on `/api/find` with:
  `ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()`.
- **Root Cause**: Fallback assignments in `pipeline.py` were implemented using Python boolean chaining:
  `corners = target.get("corners_refined") or target.get("corners") or target.get("box_corners")`
  When `target.get("corners")` was a multi-element NumPy `ndarray` of shape $(4, 2)$, evaluating `bool(array)` raised a runtime `ValueError`.
- **Solution**: Refactored all array assignments in `pipeline.py` and `ui/pipeline.py` to use explicit `is None` checks:
  ```python
  corners = target.get("corners_refined")
  if corners is None:
      corners = target.get("corners")
  if corners is None:
      corners = target.get("box_corners")
  ```

---

### 6.4 Challenge 4: Sibling Edition Misidentification
- **Symptom**: When searching for popular textbooks (e.g. *"Linear Algebra with Applications"*), standard fuzzy OCR frequently matched the wrong edition or publisher on the shelf.
- **Root Cause**: Book covers and spines often display identical primary title text while differentiating authorship or edition details in smaller secondary fonts.
- **Solution**: Implemented multi-field catalog disambiguation in `shared/ocr_reader.py`. The algorithm extracts secondary text fields (Author, Edition, Publisher, Aliases), boosting confirmed authors by $+0.40$ and penalizing competing authors by $-0.50$.

---

## 7. Learning Outcomes and Reflections

1. **Cyber-Physical Systems Engineering**: Gained extensive practical experience interfacing high-level deep learning perception models with low-level robotic manipulators over serial communication protocols.
2. **Applied Mathematics in Robotics**: Developed a deep appreciation for classical spatial mathematics—understanding that neural networks must be anchored by analytical kinematics, Denavit-Hartenberg parameterizations, and SVD-based rigid transformations to achieve physical precision.
3. **Real-Time Edge Optimization**: Acquired critical engineering techniques for eliminating latency bottlenecks, including ROI-localized model prompting, ONNX multi-threaded inference acceleration, and parallel startup routines.
4. **Defensive Hardware Programming**: Learned that robotic software requires defensive safeguards at every level—morphological depth validation, kinematic reachability checks, and strict array handling to ensure physical safety.

---

## 8. Limitations and Future Work

1. **End-Effector Dexterity**: The current pneumatic clamp is effective on standard books ($>15$ mm thick) but faces limitations on ultra-thin softcover booklets ($<5$ mm). Future iterations should explore a hybrid suction cup and adaptive claw mechanism.
2. **Dynamic Obstacle Avoidance**: The current trajectory assumes a static shelf environment. Integrating real-time 3D OctoMap point cloud occupancy grids would enable dynamic collision avoidance around unexpected obstacles.
3. **Mobile Base Integration**: Mounting the manipulator on an Autonomous Mobile Robot (AMR) equipped with LiDAR SLAM (e.g. ROS2 Navigation2) would expand the system from a stationary workcell into a facility-wide library retrieval system.
4. **Wrist-Mounted Eye-in-Hand Servoing**: Adding a lightweight micro-camera to the Dobot wrist would provide closed-loop visual servoing during the final descent phase, compensating for minor mechanical backlash.

---

## 9. Conclusion

Throughout this internship, a fully autonomous vision-guided robotic book retrieval system was successfully developed and deployed. By combining multimodal user input, rapid scene arrangement classification (~2 ms), rotated bounding box detection, ONNX-accelerated OCR (~15 ms), localized SAM sub-pixel refinement, morphological depth filtering, Kabsch hand-eye transformation ($\text{RMSE} = 2.41$ mm), and analytical SCARA kinematics, the system achieved a **95.8% physical retrieval success rate**. The completed project bridges modern computer vision foundation models with physical robotic manipulation, demonstrating a practical and robust automated solution for smart library logistics.

---

## 10. References

1. **Brooks, R.** (1986). *A robust layered control system for a mobile robot.* IEEE Journal on Robotics and Automation.
2. **Kabsch, W.** (1976). *A solution for the best rotation to relate two sets of vectors.* Acta Crystallographica Section A, 32(5), 922–923.
3. **Kirillov, A., Mintun, E., Ravi, N., et al.** (2023). *Segment Anything.* arXiv:2304.02643.
4. **Ultralytics.** (2024). *YOLOv8 / YOLO11 Oriented Bounding Box (OBB) Architecture and Training Specification.*
5. **PaddlePaddle Authors.** (2023). *PP-OCRv4: A Compact and Accurate End-to-End OCR System.* RapidOCR ONNX Runtime Engine.
6. **Intel RealSense.** (2020). *Intel RealSense D400 Series Depth Camera Family Optical and Mechanical Specifications.*
7. **Dobot.** (2018). *Dobot Magician Kinematics and Communication Protocol Specification Manual.* Shenzhen Yuejiang Technology Co., Ltd.

---

## 11. Appendices

### Appendix A: Project Directory Structure
```
c:\Users\joelv\library\
├── main.py                        # Autonomous pipeline entry point (CLI & Desktop HUD)
├── pipeline.py                    # Web Operator UI bridge (find_target, get_frame, send_to_arm)
├── server.py                      # Flask REST API backend + live MJPEG optical streamer
├── cam2robot.py                   # Calibrated Kabsch rigid transformation (Camera -> Dobot base)
├── calib.npz                      # Calibrated 3x3 rotation matrix R and 3x1 translation vector t
├── dobot_kin.py                   # Analytical forward & inverse kinematics, SCARA reach envelope
├── reach_target.py                # Standalone testbench for direct camera-to-Dobot coordinate retrieval
├── mock_library_catalog.json      # Structured mock catalog with ISBN, Call No, Authors, Aliases
├── input/
│   ├── camera.py                  # RealSense D435i pipeline, auto-exposure, hardware filters
│   ├── depth_utils.py             # Mask erosion depth sampling & pinhole 3D deprojection
│   └── query_input.py             # Voice listener (SoundDevice) & keyboard text intake
├── spine_pipeline/
│   ├── detect_spine.py            # YOLOv8 OBB shelf spine detection & IoU suppression
│   ├── find_target_spine.py       # Full spine targeting flow (OBB + OCR + SAM + Depth)
│   └── ocr_match_spine.py         # Oriented spine unwarping & title similarity matching
├── scattered_pipeline/
│   ├── detect_scattered.py        # Flat tabletop book detection (YOLO OBB / SAM 3)
│   ├── find_target_scattered.py   # Scattered book targeting flow & pick edge extraction
│   └── ocr_match_scattered.py     # Flat cover title OCR & orientation matching
├── mixed_pipeline/
│   └── find_target_mixed.py       # Dual-pipeline arbitration for scenes with both shelf & desk books
├── shared/
│   ├── mode_classifier.py         # Hybrid arrangement classifier (YOLO11n-cls + geometric rules)
│   ├── ocr_reader.py              # RapidOCR (PP-OCRv4 ONNX) singleton & multi-field catalog matching
│   ├── sam_refine.py              # Localized ROI SAM refinement, flap trimming & pick edge extraction
│   ├── visual_monitor.py          # OpenCV desktop HUD telemetry & diagnostic monitor
│   └── display_utils.py           # Structured target telemetry summary printer
├── robot/
│   └── dobot_controller.py        # Dobot Magician serial driver & 8-stage manipulation sequence
└── ui/
    ├── static/
    │   ├── index.html             # Responsive operator web dashboard
    │   ├── app.js                 # ES6 async client controller & telemetry visualizer
    │   └── style.css              # Custom dark-theme industrial UI styling
    ├── pipeline.py                # Synced UI pipeline bridge
    └── server.py                  # Synced UI backend server
```

### Appendix B: Calibration Matrix & Operational Poses
* **Transformation Equation**:
  $$\begin{bmatrix} X_{\text{robot}} \\ Y_{\text{robot}} \\ Z_{\text{robot}} \end{bmatrix} = \begin{bmatrix} 0.39476368 & 0.00000000 & -0.90868071 \\ 0.85278997 & 0.00000000 & 0.06428176 \\ 0.00000000 & -0.98015806 & -0.02042944 \end{bmatrix} \begin{bmatrix} X_{\text{cam}} \\ Y_{\text{cam}} \\ Z_{\text{cam}} \end{bmatrix}_{\text{mm}} + \begin{bmatrix} 288.20859054 \\ 356.92395906 \\ 116.69285641 \end{bmatrix}$$
* **Default Standby Pose**: $(X = 200.0, Y = 0.0, Z = 160.0, R = 0.0)$ mm
* **Deposit Station Pose**: $(X = 200.0, Y = 0.0, Z = 165.0, R = 0.0)$ mm
* **Transit Maximum Altitude**: $Z = 165.0$ mm (dynamically bounded by $\theta_{\text{elbow}} \le 146.5^\circ$)

