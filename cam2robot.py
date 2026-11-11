"""Camera -> Dobot transform. Import cam_to_robot() from your pipeline."""
from pathlib import Path
import numpy as np

CALIB_FILE = Path(__file__).resolve().parent / "calib.npz"
CALIB_SPINE = Path(__file__).resolve().parent / "calib_spine.npz"
CALIB_SCATTERED = Path(__file__).resolve().parent / "calib_scattered.npz"
CALIB_VERTICAL = Path(__file__).resolve().parent / "calib_vertical.npz"
CALIB_FRONT_FACING = Path(__file__).resolve().parent / "calib_front_facing.npz"


def fit_rigid(cam_mm, rob_mm):
    """Least-squares rigid fit (Kabsch/SVD): rob = R @ cam + t"""
    cam = np.asarray(cam_mm, float)
    rob = np.asarray(rob_mm, float)
    cc, rc = cam.mean(0), rob.mean(0)
    H = (cam - cc).T @ (rob - rc)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    t = rc - R @ cc
    return R, t


def residuals(R, t, cam_mm, rob_mm):
    pred = np.asarray(cam_mm, float) @ R.T + t
    return np.linalg.norm(pred - np.asarray(rob_mm, float), axis=1)


def fit_with_rejection(cam_mm, rob_mm, target_rmse=3.0, min_pts=4):
    """Fit, then drop the worst point until RMSE <= target (or min_pts left)."""
    cam = np.asarray(cam_mm, float)
    rob = np.asarray(rob_mm, float)
    keep = np.ones(len(cam), bool)
    while True:
        R, t = fit_rigid(cam[keep], rob[keep])
        err = residuals(R, t, cam, rob)
        rmse = float(np.sqrt(np.mean(err[keep] ** 2)))
        if rmse <= target_rmse or keep.sum() <= min_pts:
            return R, t, rmse, keep, err
        keep[np.where(keep)[0][np.argmax(err[keep])]] = False


def is_calibrated(path=CALIB_FILE) -> bool:
    """Checks whether the calibration file exists."""
    return Path(path).exists()


def save(R, t, path=CALIB_FILE):
    global _cal
    np.savez(str(path), R=R, t=t)
    if Path(path).resolve() == CALIB_FILE.resolve():
        _cal = (np.asarray(R, float), np.asarray(t, float))


def save_spine(R, t):
    """Saves spine calibration to calib_spine.npz and updates active calib.npz."""
    save(R, t, path=CALIB_SPINE)
    save(R, t, path=CALIB_FILE)


def save_scattered(R, t, also_default=False):
    """Saves scattered calibration to calib_scattered.npz (and optionally calib.npz)."""
    save(R, t, path=CALIB_SCATTERED)
    if also_default:
        save(R, t, path=CALIB_FILE)


def save_vertical(R, t, also_default=False):
    """Saves scattered vertical / front-facing calibration to calib_vertical.npz and calib_front_facing.npz."""
    save(R, t, path=CALIB_VERTICAL)
    save(R, t, path=CALIB_FRONT_FACING)
    if also_default:
        save(R, t, path=CALIB_FILE)


def load(path=CALIB_FILE, mode=None):
    if mode in ("scattered", "table", "laying", "scattered_laying"):
        p = CALIB_SCATTERED if CALIB_SCATTERED.exists() else Path(path)
    elif mode in ("vertical", "scattered_vertical", "front_facing", "facing"):
        p = CALIB_VERTICAL if CALIB_VERTICAL.exists() else (CALIB_FRONT_FACING if CALIB_FRONT_FACING.exists() else (CALIB_SPINE if CALIB_SPINE.exists() else Path(path)))
    elif mode in ("spine", "shelf"):
        p = CALIB_SPINE if CALIB_SPINE.exists() else Path(path)
    else:
        p = Path(path)

    if not p.exists():
        if CALIB_FILE.exists():
            p = CALIB_FILE
        else:
            raise FileNotFoundError(
                f"Calibration file '{p}' not found! "
                f"Please run 'python calibrate_spine.py' or 'python calibrate_scattered.py' first."
            )
    d = np.load(str(p))
    return d["R"], d["t"]


_cal = None


def cam_to_robot(xyz_m, mode=None):
    """
    Transforms RealSense color-frame point(s) in meters -> Dobot base frame in millimeters.
    Supports single point (3,) or array of points (N, 3).
    mode: 'spine', 'front_facing', 'scattered', or None.
    Note: 'front_facing' explicitly shares spine calibration, per user rule.
    """
    global _cal
    if mode is not None:
        R, t = load(mode=mode)
    else:
        if _cal is None:
            _cal = load()
        R, t = _cal

    cam_arr = np.asarray(xyz_m, float)
    if np.any(np.abs(cam_arr) > 10.0):
        cam_mm = cam_arr
    else:
        cam_mm = cam_arr * 1000.0
    if cam_mm.ndim == 1:
        return R @ cam_mm + t
    return (cam_mm @ R.T) + t


def get_spine_parallel_rotation(x, y):
    """
    Computes gripper rotation R parallel to the upright book spine.
    Fitted across 4 verified ground-truth poses from DobotStudio:
      - Point 1 (Right,     X=+15.1mm): R = 123.1° (J1=87.0°,  J4=+36.1°)
      - Point 2 (Center,    X=-26.1mm): R = 108.8° (J1=95.4°,  J4=+13.4°)
      - Point 3 (Left,      X=-59.1mm): R = 125.9° (J1=102.5°, J4=+23.4°)
      - Point 4 (Mid-shelf, X=-42.1mm): R = 133.8° (J1=99.4°,  J4=+34.5°)
    """
    if y > 150.0:
        r_val = 0.00128308 * (x ** 3) + 0.10159631 * (x ** 2) + 0.80625222 * x + 83.47633655
        return round(float(np.clip(r_val, 85.0, 140.0)), 1)
    az = float(np.degrees(np.arctan2(y, x)))
    return round(az, 1)


def compute_parallel_grasp_rotation(book_angle_deg, x, y, is_front_facing=False, flip_180=False):
    """
    Computes gripper world rotation R so that jaws enter 180° parallel to the book.
    - book_angle_deg: angle of the book or spine in Dobot X-Y coordinates (deg), or None for auto.
      For shelf spines, uses verified ground-truth curve get_spine_parallel_rotation(x, y).
      For front-facing upright books, top rim runs across shelf (theta ≈ 0° or J1 - 90°).
    - x, y: target position in Dobot coordinates (mm). Base azimuth J1 = arctan2(y, x).
    - Automatically chooses the 180° symmetric branch that minimizes wrist rotation |J4| = |R - J1|
      and keeps J4 strictly within Dobot physical joint limits (-140° <= J4 <= +140°).
    """
    j1 = float(np.degrees(np.arctan2(y, x)))
    if book_angle_deg is None or (book_angle_deg == 90.0 and not is_front_facing):
        target_r = float(get_spine_parallel_rotation(x, y))
    else:
        target_r = float(book_angle_deg)

    if is_front_facing:
        target_r -= 90.0
    if flip_180:
        target_r += 180.0

    best_j4 = None
    best_dist = 999.0
    for k in (-2, -1, 0, 1, 2):
        candidate_r = target_r + k * 180.0
        candidate_j4 = candidate_r - j1
        if -140.0 <= candidate_j4 <= 140.0:
            if abs(candidate_j4) < best_dist:
                best_dist = abs(candidate_j4)
                best_j4 = candidate_j4

    if best_j4 is None:
        best_j4 = float(np.clip(target_r - j1, -140.0, 140.0))

    r_final = round(j1 + best_j4, 1)
    return r_final, round(best_j4, 1)

