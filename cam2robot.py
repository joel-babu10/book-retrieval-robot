"""Camera -> Dobot transform. Import cam_to_robot() from your pipeline."""
from pathlib import Path
import numpy as np

CALIB_FILE = Path(__file__).resolve().parent / "calib.npz"


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
    _cal = (np.asarray(R, float), np.asarray(t, float))


def load(path=CALIB_FILE):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Calibration file '{p}' not found! "
            f"Please run 'python calibrate.py' first to perform camera-robot calibration."
        )
    d = np.load(str(p))
    return d["R"], d["t"]


_cal = None


def cam_to_robot(xyz_m):
    """
    Transforms RealSense color-frame point(s) in meters -> Dobot base frame in millimeters.
    Supports single point (3,) or array of points (N, 3).
    """
    global _cal
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
