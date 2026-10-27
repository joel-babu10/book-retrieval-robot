"""Dobot Magician kinematics, fitted to DobotStudio readouts (error < 0.1 mm).

Planar arm in the vertical plane through the base axis (all mm / deg):
    r = A + L2*sin(J2) + L3*cos(J3)      (horizontal reach of the flange)
    z =     L2*cos(J2) - L3*sin(J3)      (height of the flange)
    x, y = r*cos(J1), r*sin(J1)          (J1 = atan2(y, x))
    R = J1 + J4                          (tool rotation in the world frame)
J2 = rear arm angle from vertical, J3 = forearm angle BELOW horizontal.
Elbow interior angle = 90 + J2 - J3; the arm cannot stretch past ~143 deg of it,
i.e. the shoulder->wrist distance is capped (~267 mm). That cap is why the arm
"cannot move" at far radius when it is high up.
"""
from dataclasses import dataclass
import json, os
import numpy as np

# Physical Dimensions from Dobot Magician Kinematics Reference Guide:
D1 = 138.0   # Base height to shoulder axis (mm)
L2 = 135.0   # Rear arm (upper arm) length a2 (mm)
L3 = 147.0   # Forearm length a3 (mm)
A  = 59.7    # Horizontal wrist/platform offset a4 (mm) - matches Dobot Magician firmware to 0.0001mm
D5 = 50.0    # Vertical downward offset from wrist plate to gripper tip (mm)


@dataclass
class Limits:
    j1: tuple = (-125.0, 125.0)
    j2: tuple = (-5.0, 80.0)       # lower stops: your arm raised LIMIT_AXIS2_NEG at -5.06
    j3: tuple = (-15.0, 85.0)      # ... and LIMIT_AXIS3_NEG when commanded -15.13
    j4: tuple = (-140.0, 140.0)
    elbow_max: float = 146.5       # deg; your Studio poses prove >= 145.1 is reachable
    r_min: float = 140.0           # mm, keep the tool away from the base
    z_min: float = -50.0           # mm, table safety floor


def load_limits(path="dobot_limits.json"):
    """Override defaults with values you measured on your arm (json keys = Limits fields)."""
    lim = Limits()
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if os.path.exists(p):
        for k, v in json.load(open(p)).items():
            setattr(lim, k, tuple(v) if isinstance(v, list) else v)
    return lim


LIM = load_limits()


def fk(j1, j2, j3, j4=0.0):
    """Joint angles (deg) -> (x, y, z, R)."""
    a, b, c = np.radians([j1, j2, j3])
    r = A + L2 * np.sin(b) + L3 * np.cos(c)
    z = L2 * np.cos(b) - L3 * np.sin(c)
    return float(r * np.cos(a)), float(r * np.sin(a)), float(z), float(j1 + j4)


def ik(x, y, z):
    """(x, y, z) -> (j1, j2, j3) in deg, elbow-up, or None if geometrically unreachable."""
    r = float(np.hypot(x, y))
    rp = r - A
    d = float(np.hypot(rp, z))
    if d > L2 + L3 - 1e-6 or d < abs(L2 - L3) + 1e-6:
        return None
    alpha = np.arctan2(rp, z)
    beta = np.arccos(np.clip((L2 ** 2 + d ** 2 - L3 ** 2) / (2 * L2 * d), -1, 1))
    j2 = alpha - beta
    ex, ez = L2 * np.sin(j2), L2 * np.cos(j2)
    j3 = np.arctan2(-(z - ez), rp - ex)
    return float(np.degrees(np.arctan2(y, x))), float(np.degrees(j2)), float(np.degrees(j3))


def flange_to_tcp(x, y, z):
    """Dobot flange coordinate (Z=0 at shoulder axis) -> Pneumatic gripper fingertip coordinate (Z_tcp = Z - d5)."""
    return float(x), float(y), float(z - D5)


def tcp_to_flange(x, y, z):
    """Pneumatic gripper fingertip coordinate -> Dobot flange coordinate (Z_flange = Z_tcp + d5)."""
    return float(x), float(y), float(z + D5)


class DobotMagicianKinematics:
    """Standard D-H Reference Kinematics from Dobot Magician Kinematics Reference Guide.
    Corrected with exact asymmetric Law of Cosines for L2 != L3."""
    def __init__(self, d1=D1, a2=L2, a3=L3, a4=A, d5=D5):
        self.d1, self.a2, self.a3, self.a4, self.d5 = d1, a2, a3, a4, d5

    def forward_kinematics(self, j1_deg, j2_deg, j3_deg, j4_deg=0.0):
        """Standard D-H forward kinematics returning (x, y, z) and 4x4 T_0^5 tracking matrix."""
        j1, j2, j3, j4 = np.radians([j1_deg, j2_deg, j3_deg, j4_deg])
        r = self.a2 * np.cos(j2) + self.a3 * np.cos(j3) + self.a4
        x, y = r * np.cos(j1), r * np.sin(j1)
        z = self.d1 + self.a2 * np.sin(j2) + self.a3 * np.sin(j3) - self.d5
        phi = j1 + j4
        t_matrix = np.array([
            [np.cos(phi), -np.sin(phi), 0.0, x],
            [np.sin(phi),  np.cos(phi), 0.0, y],
            [0.0,          0.0,         1.0, z],
            [0.0,          0.0,         0.0, 1.0]
        ])
        return (float(x), float(y), float(z)), t_matrix

    def inverse_kinematics(self, x, y, z):
        """Asymmetric Law of Cosines IK for Dobot Magician (L2 != L3)."""
        j1 = np.arctan2(y, x)
        r_arm = np.hypot(x, y) - self.a4
        z_arm = (z + self.d5) - self.d1
        d_sq = r_arm**2 + z_arm**2
        d = np.sqrt(d_sq)
        if d > self.a2 + self.a3 or d < abs(self.a2 - self.a3):
            return None
        cos_alpha2 = np.clip((self.a2**2 + d_sq - self.a3**2) / (2.0 * self.a2 * d), -1.0, 1.0)
        cos_alpha3 = np.clip((self.a3**2 + d_sq - self.a2**2) / (2.0 * self.a3 * d), -1.0, 1.0)
        alpha2 = np.arccos(cos_alpha2)
        alpha3 = np.arccos(cos_alpha3)
        gamma = np.arctan2(z_arm, r_arm)
        return float(np.degrees(j1)), float(np.degrees(gamma + alpha2)), float(np.degrees(gamma - alpha3))


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def j4_for(R, j1, fold=False, lim=LIM):
    """J4 giving tool rotation R at base angle j1. fold=True may use R+-180 (symmetric gripper)."""
    c = [wrap180(R - j1)]
    if fold:
        c += [wrap180(R - j1 + 180.0)]
    c = [v for v in c if lim.j4[0] <= v <= lim.j4[1]]
    return min(c, key=abs) if c else None


def check(x, y, z, lim=LIM, margin=0.0):
    """-> (ok, reason, (j1, j2, j3)). margin (deg) shrinks every angular limit."""
    r = float(np.hypot(x, y))
    if r < lim.r_min:
        return False, f"too close to base (r={r:.0f} < {lim.r_min:.0f})", None
    if z < lim.z_min:
        return False, f"below floor (z={z:.0f} < {lim.z_min:.0f})", None
    j = ik(x, y, z)
    if j is None:
        return False, f"geometrically unreachable (r={r:.0f}, z={z:.0f})", None
    j1, j2, j3 = j
    elbow = 90.0 + j2 - j3
    for name, v, (lo, hi) in (("J1", j1, lim.j1), ("J2", j2, lim.j2), ("J3", j3, lim.j3)):
        if v < lo + margin or v > hi - margin:
            return False, f"{name}={v:.1f} outside [{lo + margin:.1f}, {hi - margin:.1f}]", j
    if elbow > lim.elbow_max - margin:
        return False, f"arm too stretched at r={r:.0f}, z={z:.0f} (elbow {elbow:.1f} > {lim.elbow_max - margin:.1f})", j
    return True, "ok", j


def z_max_at(r, lim=LIM, margin=0.0):
    """Highest flange z reachable at horizontal radius r (None if r is unreachable)."""
    for z in np.arange(200.0, lim.z_min - 1.0, -1.0):
        if check(r, 0.0, float(z), lim, margin)[0]:
            return float(z)
    return None


def segment_ok(p0, p1, lim=LIM, margin=0.0, step=3.0, skip_first=False):
    """Sample the straight Cartesian line p0->p1; every sample must be reachable."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / step)) + 1)
    for i, t in enumerate(np.linspace(0.0, 1.0, n)):
        if skip_first and i == 0:
            continue   # current pose is where the arm already is
        p = p0 + t * (p1 - p0)
        ok, why, _ = check(p[0], p[1], p[2], lim, margin)
        if not ok:
            return False, f"at ({p[0]:.0f},{p[1]:.0f},{p[2]:.0f}): {why}"
    return True, "ok"


def stage_equations(x, y, z, R=None, lim=LIM, margin=0.9):
    """Closed-form joints for your J1 -> J2 -> Z-down method, for ANY target (x, y, z) [flange coords, mm].

      r = hypot(x, y);  r' = r - A;  J1 = atan2(y, x);  J4 = R - J1
      stage 2  J3 held at -14.1 deg (lim.j3[0] + 0.9): J2 = asin((r' - L3*cos J3) / L2)
               height reached:                         z_A = L2*cos J2 - L3*sin J3
               if that over-stretches the elbow (90 + J2 - J3 > cap) the arm rides the stretch circle instead:
                                                       z_A = sqrt(D_cap^2 - r'^2),  D_cap^2 = L2^2 + L3^2 - 2*L2*L3*cos(cap)
      stage 3  straight down at the same r:            (J2, J3) = ik(x, y, z)
    z_A is the highest the arm can be at that radius; the target must satisfy z <= z_A.
    """
    r = float(np.hypot(x, y))
    rp = r - A
    az = float(np.degrees(np.arctan2(y, x)))
    j3h = lim.j3[0] + margin
    out = {"r": r, "az": az, "j1": az, "j3_hold": j3h, "ok": False, "reason": "", "route": ""}
    c3 = np.cos(np.radians(j3h))
    s = (rp - L3 * c3) / L2
    cap = np.radians(lim.elbow_max - margin)
    d_cap = float(np.sqrt(L2 ** 2 + L3 ** 2 - 2 * L2 * L3 * np.cos(cap)))
    if abs(s) > 1.0 or rp > d_cap:
        out["reason"] = f"radius {r:.0f} mm is beyond the arm's reach"
        return out
    j2h = float(np.degrees(np.arcsin(s)))
    if j2h < lim.j2[0] + margin:
        out["reason"] = (f"radius {r:.0f} mm is too close for the J1 -> J2 -> Z-down method: the arm cannot rise there "
                         f"(J2 would need {j2h:.1f} deg < {lim.j2[0] + margin:.1f}); the book must be at r >= ~205 mm")
        return out
    z_h = L2 * np.cos(np.radians(j2h)) - L3 * np.sin(np.radians(j3h))
    if 90.0 + j2h - j3h <= lim.elbow_max - margin:
        z_a, route = float(z_h), "J3 held, J2 only"
    else:
        z_a, route = float(np.sqrt(max(d_cap ** 2 - rp ** 2, 0.0))), "riding the stretch circle"
    ext = ik(r, 0.0, z_a)                               # joints at the extension point (r, z_A)
    grab = ik(x, y, z)
    out.update(z_A=z_a, route=route, extend=(az, ext[1], ext[2]) if ext else None,
               grab=(az, grab[1], grab[2]) if grab else None)
    if R is not None:
        out["j4"] = wrap180(R - az)
    if ext is None or grab is None:
        out["reason"] = "geometrically unreachable"
    elif z > z_a + 1e-6:
        out["reason"] = f"target z={z:.0f} is above the arm's ceiling at r={r:.0f} (z_A={z_a:.0f})"
    else:
        out["ok"], out["reason"] = True, "ok"
    return out


# --- Tool Offset Compensation (Claws relative to J4 rotation axis) ---
TOOL_OFFSET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tool_offset.json")
_tool_offset = None


def _rz(deg):
    """2D Rotation matrix around Z axis for angle in degrees."""
    r = np.radians(deg)
    return np.array([[np.cos(r), -np.sin(r)],
                     [np.sin(r),  np.cos(r)]], dtype=float)


def set_tool_offset(ox_oy):
    """Save claw offset o = [ox, oy] in mm to tool_offset.json."""
    global _tool_offset
    ox, oy = float(ox_oy[0]), float(ox_oy[1])
    _tool_offset = np.array([ox, oy], dtype=float)
    data = {
        "ox": ox,
        "oy": oy,
        "magnitude_mm": round(float(np.hypot(ox, oy)), 2),
        "angle_deg": round(float(np.degrees(np.arctan2(oy, ox))), 2)
    }
    with open(TOOL_OFFSET_FILE, "w") as f:
        json.dump(data, f, indent=2)
    return _tool_offset


def get_tool_offset():
    """Load claw offset o = [ox, oy] in mm from tool_offset.json (default [12.31, -9.62])."""
    global _tool_offset
    if _tool_offset is not None:
        return _tool_offset
    if os.path.exists(TOOL_OFFSET_FILE):
        try:
            with open(TOOL_OFFSET_FILE, "r") as f:
                d = json.load(f)
                _tool_offset = np.array([float(d["ox"]), float(d["oy"])], dtype=float)
                return _tool_offset
        except Exception:
            pass
    # Default verified claw offset from DobotStudio paired touches
    _tool_offset = np.array([12.31, -9.62], dtype=float)
    return _tool_offset


def claw_to_flange_xy(claw_x, claw_y, r_deg):
    """
    Model: flange_xy = claw_xy - Rz(R) @ o
    Translates desired claw landing position into Dobot flange command.
    """
    o = get_tool_offset()
    r_mat = _rz(r_deg)
    flange = np.array([claw_x, claw_y], dtype=float) - r_mat @ o
    return float(flange[0]), float(flange[1])


def flange_to_claw_xy(flange_x, flange_y, r_deg):
    """
    Model: claw_xy = flange_xy + Rz(R) @ o
    Translates commanded Dobot flange position into actual physical claw tip position.
    """
    o = get_tool_offset()
    r_mat = _rz(r_deg)
    claw = np.array([flange_x, flange_y], dtype=float) + r_mat @ o
    return float(claw[0]), float(claw[1])


if __name__ == "__main__":
    import sys
    x, y, z = (float(v) for v in sys.argv[1:4]) if len(sys.argv) >= 4 else (118.127, 260.6505, 126.29)
    R = float(sys.argv[4]) if len(sys.argv) >= 5 else -9.56
    e = stage_equations(x, y, z, R)
    print(f"target ({x}, {y}, {z})  r={e['r']:.1f}  ->  {e['reason']}  [{e['route']}]")
    if e.get("extend"):
        print(f"stage 1  J1 = {e['j1']:.2f}   J4 = {e.get('j4', float('nan')):.2f}")
        print(f"stage 2  J2 = {e['extend'][1]:.2f}  J3 = {e['extend'][2]:.2f}   (height z_A = {e['z_A']:.1f})")
    if e.get("grab"):
        print(f"stage 3  J2 = {e['grab'][1]:.2f}  J3 = {e['grab'][2]:.2f}   (straight down to z = {z:.1f})")
