import numpy as np
import mujoco
import mujoco.viewer
from numpy.linalg import norm
import matplotlib.pyplot as plt

# ============================================================
# RCM + Null-Space Energy-Tank Controller
# with MuJoCo passive-dissipation diagnostics
#
# XML inspection:
#   <joint armature="0.1" damping="1" ... />
#
# Therefore the seven Panda joints inherit viscous damping = 1.
# No joint frictionloss or joint spring is specified in the supplied XML.
#
# Main energy balance used here:
#
#   S(t)-S(0)
#     = W_ext + W_ref - E_mj_diss + W_unmodelled
#
# Hence the corrected residual is
#
#   R = S-S0-W_ext-W_ref+E_mj_diss
#
# If all relevant energy channels are captured:
#   R ~= 0
#
# Energy-tank internal check:
#
#   dE_tank/dt ~= alpha*P_diss_controller + u^T F2
#
# ============================================================


# ============================================================
# User configuration
# ============================================================

# Recommended first:
#   "fixed_passivity"
# Then:
#   "external_pulse"
# Finally:
#   "circle_tracking"
TEST_MODE = "circle_tracking"
# TEST_MODE = "fixed_passivity"
# TEST_MODE = "external_pulse"

XML_PATH = "model/franka_emika_panda/scene_panda_nohand.xml"

T_END = 25.0
LOG_EVERY = 5

# ------------------------------------------------------------
# Surgical rod geometry
# ------------------------------------------------------------
AXIS_LOCAL_IDX = 2
L_ROD = 0.20
D_RCM = 0.10

# ------------------------------------------------------------
# Main task: fixed RCM point
# ------------------------------------------------------------
KP_TASK1 = np.array([600.0, 600.0, 600.0])
KD_TASK1 = 0.707 * np.sqrt(KP_TASK1)

# ------------------------------------------------------------
# Secondary task: rod tip
# ------------------------------------------------------------
KP_TASK2 = np.array([300.0, 300.0, 300.0])
KD_TASK2 = 0.707 * np.sqrt(KP_TASK2)

# ------------------------------------------------------------
# Energy tank
# ------------------------------------------------------------
E_TANK_INIT = 15.0
E_MAX = 50.0
E_THRESH = 5.0
S_EPS = 1e-8
ALPHA = 1.0

KP_DRIFT = np.array([15.0, 15.0, 15.0])

# ------------------------------------------------------------
# Circle trajectory
# ------------------------------------------------------------
R_CIRC = 0.03
F_CIRC = 0.15
T_RAMP = 2.0

# ------------------------------------------------------------
# External torque pulse
# ------------------------------------------------------------
TAU_PULSE_AMP = 8.0
PULSE_T = 0.5
PULSE_GAP = 5.0
N_PULSES = 4
T_FIRST_PULSE = 2.5

# ------------------------------------------------------------
# Approximate cross-priority Coriolis compensation
# ------------------------------------------------------------
ENABLE_TAU_COMP = True
MUCOMP_WARMUP_SEC = 0.5
MUCOMP_LPF_BETA = 0.9


# ============================================================
# Trajectory helpers
# ============================================================
def smoothstep_quintic(tau: float):
    tau = float(np.clip(tau, 0.0, 1.0))
    s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    ds = 30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4
    return s, ds


def circle_ref(t, center, plane_u, plane_v, r_circ, omega_circ, T_ramp):

    ramp, dramp = smoothstep_quintic(np.clip(t / T_ramp, 0.0, 1.0))

    r_eff = r_circ * ramp
    r_eff_dot = r_circ * dramp / T_ramp

    theta = omega_circ * t
    c = np.cos(theta)
    si = np.sin(theta)

    pos = center + r_eff * c * plane_u + r_eff * si * plane_v

    vel = (r_eff_dot * c - r_eff * omega_circ * si) * plane_u + (
        r_eff_dot * si + r_eff * omega_circ * c
    ) * plane_v

    return pos, vel


# ============================================================
# Dynamics / geometry utilities
# ============================================================
def full_mass_matrix(model, data, nv_use=7):
    M_full = np.zeros((model.nv, model.nv), dtype=float)

    mujoco.mj_fullM(model, M_full, data.qM)

    return M_full[:nv_use, :nv_use].copy()


def dyn_consistent_projector(J, M, eps=1e-8):
    Minv = np.linalg.inv(M)

    Lambda_inv = J @ Minv @ J.T

    Lambda = np.linalg.inv(Lambda_inv + eps * np.eye(J.shape[0]))

    Jbar = Minv @ J.T @ Lambda

    N = np.eye(M.shape[0]) - Jbar @ J

    return Jbar, N


def nullspace_basis_raw(J, tol=1e-10):
    _, S, Vt = np.linalg.svd(J, full_matrices=True)

    r = np.sum(S > tol)

    V2 = Vt.T[:, r:]

    return V2.T


def align_nullspace_basis(Z2_raw, Z2_prev):
    if Z2_prev is None:
        return Z2_raw

    A = Z2_raw @ Z2_prev.T

    U, _, Vt = np.linalg.svd(A)

    R = U @ Vt

    return R.T @ Z2_raw


def compute_J2bar(J1, M, Z2_prev=None, tol=1e-10, eps=1e-8):

    Z2_raw = nullspace_basis_raw(J1, tol=tol)

    if Z2_raw.shape[0] == 0:
        return (np.zeros((0, M.shape[0])), None)

    Z2 = align_nullspace_basis(Z2_raw, Z2_prev)

    Lambda2 = Z2 @ M @ Z2.T

    Jbar2 = np.linalg.solve(Lambda2 + eps * np.eye(Lambda2.shape[0]), Z2 @ M)

    return Jbar2, Z2


def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def compute_g_only(model, data, nv_use=7):
    qvel_bk = data.qvel.copy()

    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    g = data.qfrc_bias[:nv_use].copy()

    data.qvel[:] = qvel_bk
    mujoco.mj_forward(model, data)

    return g


def compute_Cqd_given_qd(model, data, qd_use_7, g_7):

    qvel_bk = data.qvel.copy()

    data.qvel[:] = 0.0
    data.qvel[:7] = qd_use_7

    mujoco.mj_forward(model, data)

    c_bias = data.qfrc_bias[:7].copy()

    data.qvel[:] = qvel_bk
    mujoco.mj_forward(model, data)

    return c_bias - g_7


def clamp_tau(tau, ctrl_lo, ctrl_hi, frac=0.35):

    lim = frac * np.minimum(np.abs(ctrl_lo), np.abs(ctrl_hi))

    lim = np.maximum(lim, 1e-6)

    return np.clip(tau, -lim, lim)


# ============================================================
# Main
# ============================================================
def main():

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    model = mujoco.MjModel.from_xml_path(XML_PATH)

    data = mujoco.MjData(model)

    dt = float(model.opt.timestep)

    # --------------------------------------------------------
    # IDs
    # --------------------------------------------------------
    ee_site_name = "ee_site"

    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)

    if ee_site_id < 0:
        raise RuntimeError(f"Cannot find site '{ee_site_name}'.")

    joint3_name = "joint3"

    jid3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint3_name)

    if jid3 < 0:
        raise RuntimeError(f"Cannot find joint '{joint3_name}'.")

    dof3 = int(model.jnt_dofadr[jid3])

    # --------------------------------------------------------
    # Reset home
    # --------------------------------------------------------
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)

    mujoco.mj_forward(model, data)

    # --------------------------------------------------------
    # Inspect MuJoCo passive parameters
    # --------------------------------------------------------
    damping_coeff = model.dof_damping[:7].copy()

    frictionloss_coeff = np.zeros(7)

    # frictionloss belongs to joints, not directly DOFs.
    # All seven Panda joints are 1-DOF hinge joints here.
    for jname in ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)

        dofadr = int(model.jnt_dofadr[jid])

        frictionloss_coeff[dofadr] = model.dof_frictionloss[dofadr]

    print("=" * 78)
    print("MuJoCo passive-force inspection")
    print("=" * 78)

    print("dof_damping[:7]      =", damping_coeff)

    print("dof_frictionloss[:7] =", frictionloss_coeff)

    print("Expected from supplied XML: " "damping = 1 for every arm joint, " "frictionloss = 0.")

    print("=" * 78)

    # --------------------------------------------------------
    # Initial rod geometry
    # --------------------------------------------------------
    p_ee0 = data.site_xpos[ee_site_id].copy()

    R_ee0 = data.site_xmat[ee_site_id].reshape(3, 3).copy()

    axis_world0 = R_ee0[:, AXIS_LOCAL_IDX].copy()

    p_rcm_target = p_ee0 + D_RCM * axis_world0

    p_tip0 = p_ee0 + L_ROD * axis_world0

    # --------------------------------------------------------
    # Circle plane
    # --------------------------------------------------------
    axis0 = axis_world0 / norm(axis_world0)

    tmp = np.array([1.0, 0.0, 0.0])

    if abs(np.dot(tmp, axis0)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])

    plane_u = tmp - np.dot(tmp, axis0) * axis0

    plane_u /= norm(plane_u)

    plane_v = np.cross(axis0, plane_u)

    omega_circ = 2.0 * np.pi * F_CIRC

    # --------------------------------------------------------
    # Actuator limits
    # --------------------------------------------------------
    ctrl_lo = model.actuator_ctrlrange[:7, 0].copy()

    ctrl_hi = model.actuator_ctrlrange[:7, 1].copy()

    m1 = 3
    m2 = 3

    # --------------------------------------------------------
    # Energy tank state
    # --------------------------------------------------------
    x2_hat = np.zeros(m2)
    x2_hat_inited = False

    E_tank = float(E_TANK_INIT)

    s = float(np.sqrt(2.0 * E_tank))

    # --------------------------------------------------------
    # External pulse timing
    # --------------------------------------------------------
    pulse_starts = [T_FIRST_PULSE + k * (PULSE_T + PULSE_GAP) for k in range(N_PULSES)]

    # --------------------------------------------------------
    # Integrated energy variables
    # --------------------------------------------------------
    Wext = 0.0
    Wref = 0.0

    # MuJoCo physical passive dissipation
    E_mj_passive_diss = 0.0

    # Parameter-based viscous damping check
    E_mj_damping_param = 0.0

    # frictionloss estimate using parameter f*|qdot|
    E_mj_friction_param = 0.0

    # Cross-level compensation work
    W_comp = 0.0

    # --------------------------------------------------------
    # Logs
    # --------------------------------------------------------
    T_log = []

    tip_log = []
    tipd_log = []

    V1_log = []
    V2_log = []

    rcm_err_log = []
    tip_err_actual_log = []
    drift_log = []

    Etank_log = []
    Stotal_log = []

    Wext_log = []
    Wref_log = []

    Emj_passive_log = []
    Emj_damping_param_log = []
    Emj_friction_param_log = []

    Wcomp_log = []

    residual_raw_log = []
    residual_mj_log = []

    Pmujoco_passive_log = []
    Pdamping_param_log = []
    Pfriction_param_log = []

    Pcomp_log = []

    Ptank_theory_log = []
    Ptank_numeric_log = []
    tank_power_residual_log = []

    tau_sat_log = []

    S0 = None

    total_steps = 0

    tau_comp_prev = np.zeros(7)

    Z2_track = None

    print(f"TEST_MODE = {TEST_MODE}")

    print(f"E_tank init / threshold = " f"{E_TANK_INIT:.3f} / " f"{E_THRESH:.3f} J")

    # ========================================================
    # Simulation
    # ========================================================
    with mujoco.viewer.launch_passive(model, data) as viewer:

        viewer.cam.distance = 3.5
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20

        while viewer.is_running():

            mujoco.mj_forward(model, data)

            qd = data.qvel[:7].copy()

            t = float(data.time)

            if t >= T_END:
                break

            # =================================================
            # MuJoCo passive force BEFORE controller modifies data
            # =================================================
            tau_passive_mj = data.qfrc_passive[:7].copy()

            # Signed power performed by passive force on robot.
            #
            # For pure damping:
            #   tau_passive = -b*qdot
            #   qdot^T tau_passive <= 0
            P_passive_signed = float(np.dot(qd, tau_passive_mj))

            # Positive number = dissipated power.
            #
            # We use -P_signed rather than max(0,...) so that
            # the exact signed energy bookkeeping remains visible.
            P_mj_passive_diss = -P_passive_signed

            E_mj_passive_diss += P_mj_passive_diss * dt

            # -------------------------------------------------
            # Independent parameter-based damping calculation
            #
            # supplied XML:
            #   damping = 1
            #
            # P = sum b_i*qdot_i^2
            # -------------------------------------------------
            P_damping_param = float(np.sum(damping_coeff * qd**2))

            E_mj_damping_param += P_damping_param * dt

            # -------------------------------------------------
            # Parameter-based Coulomb friction estimate
            #
            # supplied XML has frictionloss = 0,
            # so this should stay zero.
            # -------------------------------------------------
            P_friction_param = float(np.sum(frictionloss_coeff * np.abs(qd)))

            E_mj_friction_param += P_friction_param * dt

            # =================================================
            # Jacobians
            # =================================================
            J_pos_full = np.zeros((3, model.nv), dtype=float)

            J_rot_full = np.zeros((3, model.nv), dtype=float)

            mujoco.mj_jacSite(model, data, J_pos_full, J_rot_full, ee_site_id)

            J_pos = J_pos_full[:, :7].copy()

            J_rot = J_rot_full[:, :7].copy()

            p_ee = data.site_xpos[ee_site_id].copy()

            R_ee = data.site_xmat[ee_site_id].reshape(3, 3).copy()

            axis_world = R_ee[:, AXIS_LOCAL_IDX]

            skew_axis = skew(axis_world)

            # =================================================
            # Main task: RCM point
            # =================================================
            p_rcm = p_ee + D_RCM * axis_world

            J1 = J_pos - D_RCM * skew_axis @ J_rot

            e1 = p_rcm - p_rcm_target

            xdot1 = J1 @ qd

            F1 = -KP_TASK1 * e1 - KD_TASK1 * xdot1

            tau_task = J1.T @ F1

            # =================================================
            # Secondary coordinate: rod tip
            # =================================================
            p_tip = p_ee + L_ROD * axis_world

            J2 = J_pos - L_ROD * skew_axis @ J_rot

            # =================================================
            # Dynamically consistent decomposition
            # =================================================
            M = full_mass_matrix(model, data, nv_use=7)

            J1Mplus, _ = dyn_consistent_projector(J1, M, eps=1e-8)

            J2bar, Z2 = compute_J2bar(J1, M, Z2_prev=Z2_track, tol=1e-10, eps=1e-8)

            if Z2 is None or Z2.shape[0] == 0:
                print("[WARN] Null space empty.")

                g_7 = compute_g_only(model, data, nv_use=7)

                tau_raw = tau_task + g_7

                tau = np.clip(tau_raw, ctrl_lo, ctrl_hi)

                data.ctrl[:7] = tau

                mujoco.mj_step(model, data)

                viewer.sync()

                total_steps += 1

                continue

            Z2_track = Z2

            # Dietrich Eq. (11)
            w1 = J2 @ J1Mplus @ xdot1

            v2 = J2bar @ qd

            w2 = J2 @ Z2.T @ v2

            x2 = p_tip.copy()

            if not x2_hat_inited:
                x2_hat = x2.copy()
                x2_hat_inited = True

            # =================================================
            # Energy tank modulation
            # =================================================
            w1_ref = w1 + KP_DRIFT * (x2 - x2_hat)

            allow_tap = E_tank > E_THRESH and s > S_EPS

            if allow_tap:
                n_gain = w1_ref / s
            else:
                n_gain = np.zeros(m2)

            u = n_gain * s

            x2hat_dot = u + w2

            x2_hat = x2_hat + x2hat_dot * dt

            # =================================================
            # Secondary reference
            # =================================================
            if TEST_MODE in ("fixed_passivity", "external_pulse"):

                x2_d = p_tip0.copy()
                x2_d_dot = np.zeros(3)

            elif TEST_MODE == "circle_tracking":

                x2_d, x2_d_dot = circle_ref(t, p_tip0, plane_u, plane_v, R_CIRC, omega_circ, T_RAMP)

            else:
                raise ValueError(f"Unknown TEST_MODE={TEST_MODE}")

            # =================================================
            # Secondary force
            # =================================================
            e2 = x2_hat - x2_d

            F2 = -KP_TASK2 * e2 + KD_TASK2 * (x2_d_dot - x2hat_dot)

            tau_null = J2bar.T @ Z2 @ J2.T @ F2

            # =================================================
            # Moving-reference supply power
            #
            # V2 = 1/2 (xhat-xd)^T K (xhat-xd)
            #
            # F2 = -K e
            #      -D*xhat_dot
            #      +D*xd_dot
            #
            # This implementation keeps the same reference-port
            # expression used in the previous diagnostic version.
            # =================================================
            P_ref = float(np.dot(-KP_TASK2 * e2 + KD_TASK2 * x2hat_dot, x2_d_dot))

            Wref += P_ref * dt

            # =================================================
            # Controller damping / tank balance
            # =================================================
            Pd1 = float(np.dot(xdot1, KD_TASK1 * xdot1))

            Pd2 = float(np.dot(x2hat_dot, KD_TASK2 * x2hat_dot))

            Pdiss = Pd1 + Pd2

            P_ex = float(np.dot(u, F2))

            P_tank_theory = ALPHA * Pdiss + P_ex

            E_tank_old = float(E_tank)

            if s > S_EPS:
                s_dot = P_tank_theory / s
            else:
                s_dot = ALPHA * Pdiss / max(S_EPS, 1e-12)

            s = float(np.clip(s + s_dot * dt, 0.0, np.sqrt(2.0 * E_MAX)))

            E_tank = 0.5 * s * s

            P_tank_numeric = (E_tank - E_tank_old) / dt

            P_tank_residual = P_tank_numeric - P_tank_theory

            # =================================================
            # Gravity / cross-level compensation
            # =================================================
            tau_comp = np.zeros(7)

            g_7 = compute_g_only(model, data, nv_use=7)

            if ENABLE_TAU_COMP and t > MUCOMP_WARMUP_SEC:

                Jbar_full = np.vstack([J1, J2bar])

                Minv = np.linalg.inv(M)

                Lambda1_inv = J1 @ Minv @ J1.T

                Lambda1 = np.linalg.inv(Lambda1_inv + 1e-8 * np.eye(m1))

                Lambda2_mat = Z2 @ M @ Z2.T

                Lambda2_blk = np.linalg.inv(Lambda2_mat + 1e-8 * np.eye(Z2.shape[0]))

                Lambda = np.zeros((7, 7))

                Lambda[:m1, :m1] = Lambda1

                Lambda[m1:, m1:] = Lambda2_blk

                qd_a = J1Mplus @ xdot1

                qd_b = Z2.T @ v2

                Cqd_a = compute_Cqd_given_qd(model, data, qd_a, g_7)

                Cqd_b = compute_Cqd_given_qd(model, data, qd_b, g_7)

                term_a = Jbar_full @ np.linalg.solve(M, Cqd_a)

                term_b = Jbar_full @ np.linalg.solve(M, Cqd_b)

                mu_v_a = Lambda @ term_a

                mu_v_b = Lambda @ term_b

                mu12_v2 = mu_v_b[:m1]

                mu21_v1 = mu_v_a[m1:]

                v_comp = np.zeros(7)

                v_comp[:m1] = mu12_v2

                v_comp[m1:] = mu21_v1

                tau_comp_raw = Jbar_full.T @ v_comp

                tau_comp_raw = clamp_tau(tau_comp_raw, ctrl_lo, ctrl_hi, frac=0.35)

                tau_comp = MUCOMP_LPF_BETA * tau_comp_prev + (1.0 - MUCOMP_LPF_BETA) * tau_comp_raw

                tau_comp_prev = tau_comp.copy()

            # Work of approximate compensation term.
            #
            # In the ideal analytical Dietrich structure this
            # term belongs to the dynamics transformation.
            # Here we log it independently to diagnose whether
            # the numerical approximation has nonzero net work.
            P_comp = float(np.dot(qd, tau_comp))

            W_comp += P_comp * dt

            # =================================================
            # Final command
            # =================================================
            tau_raw = tau_task + tau_null + g_7 + tau_comp

            tau = np.clip(tau_raw, ctrl_lo, ctrl_hi)

            data.ctrl[:7] = tau

            # =================================================
            # External disturbance
            # =================================================
            data.qfrc_applied[:] = 0.0

            tau3 = 0.0

            if TEST_MODE == "external_pulse":

                for tk in pulse_starts:

                    if tk <= t <= tk + PULSE_T:

                        phase = (t - tk) / PULSE_T

                        tau3 = TAU_PULSE_AMP * np.sin(np.pi * phase)

                        break

            data.qfrc_applied[dof3] = tau3

            P_ext = float(np.dot(qd, data.qfrc_applied[:7]))

            Wext += P_ext * dt

            # =================================================
            # Logging
            # =================================================
            if total_steps % LOG_EVERY == 0:

                V1 = 0.5 * float(np.dot(e1, KP_TASK1 * e1))

                V2_hat = 0.5 * float(np.dot(e2, KP_TASK2 * e2))

                Ekin = 0.5 * float(qd.T @ M @ qd)

                Stotal = Ekin + V1 + V2_hat + E_tank

                if S0 is None:
                    S0 = Stotal

                # ---------------------------------------------
                # Raw residual:
                # ignores MuJoCo passive dissipation
                # ---------------------------------------------
                residual_raw = Stotal - S0 - Wext - Wref

                # ---------------------------------------------
                # MuJoCo-passive corrected residual
                #
                # dS = P_supply - P_mj_diss
                #
                # => S-S0-W_supply+E_mj_diss = 0
                # ---------------------------------------------
                residual_mj = Stotal - S0 - Wext - Wref + E_mj_passive_diss

                T_log.append(t)

                tip_log.append(p_tip.copy())

                tipd_log.append(x2_d.copy())

                V1_log.append(V1)
                V2_log.append(V2_hat)

                rcm_err_log.append(norm(e1))

                tip_err_actual_log.append(norm(x2 - x2_d))

                drift_log.append(norm(x2 - x2_hat))

                Etank_log.append(E_tank)

                Stotal_log.append(Stotal)

                Wext_log.append(Wext)

                Wref_log.append(Wref)

                Emj_passive_log.append(E_mj_passive_diss)

                Emj_damping_param_log.append(E_mj_damping_param)

                Emj_friction_param_log.append(E_mj_friction_param)

                Wcomp_log.append(W_comp)

                residual_raw_log.append(residual_raw)

                residual_mj_log.append(residual_mj)

                Pmujoco_passive_log.append(P_mj_passive_diss)

                Pdamping_param_log.append(P_damping_param)

                Pfriction_param_log.append(P_friction_param)

                Pcomp_log.append(P_comp)

                Ptank_theory_log.append(P_tank_theory)

                Ptank_numeric_log.append(P_tank_numeric)

                tank_power_residual_log.append(P_tank_residual)

                tau_sat_log.append(norm(tau_raw - tau))

            if total_steps % 200 == 0:

                print(
                    f"t={t:6.2f}  "
                    f"tap={allow_tap}  "
                    f"Etank={E_tank:7.3f} J  "
                    f"RCM={norm(e1)*1000:7.3f} mm  "
                    f"Emj={E_mj_passive_diss:7.3f} J"
                )

            mujoco.mj_step(model, data)

            viewer.sync()

            total_steps += 1

    # ========================================================
    # Array conversion
    # ========================================================
    if len(T_log) == 0:
        print("[WARN] No data logged.")
        return

    T_log = np.asarray(T_log)

    tip_log = np.asarray(tip_log)
    tipd_log = np.asarray(tipd_log)

    V1_log = np.asarray(V1_log)
    V2_log = np.asarray(V2_log)

    rcm_err_log = np.asarray(rcm_err_log)

    tip_err_actual_log = np.asarray(tip_err_actual_log)

    drift_log = np.asarray(drift_log)

    Etank_log = np.asarray(Etank_log)

    Stotal_log = np.asarray(Stotal_log)

    Wext_log = np.asarray(Wext_log)

    Wref_log = np.asarray(Wref_log)

    Emj_passive_log = np.asarray(Emj_passive_log)

    Emj_damping_param_log = np.asarray(Emj_damping_param_log)

    Emj_friction_param_log = np.asarray(Emj_friction_param_log)

    Wcomp_log = np.asarray(Wcomp_log)

    residual_raw_log = np.asarray(residual_raw_log)

    residual_mj_log = np.asarray(residual_mj_log)

    Pmujoco_passive_log = np.asarray(Pmujoco_passive_log)

    Pdamping_param_log = np.asarray(Pdamping_param_log)

    Pfriction_param_log = np.asarray(Pfriction_param_log)

    Pcomp_log = np.asarray(Pcomp_log)

    Ptank_theory_log = np.asarray(Ptank_theory_log)

    Ptank_numeric_log = np.asarray(Ptank_numeric_log)

    tank_power_residual_log = np.asarray(tank_power_residual_log)

    tau_sat_log = np.asarray(tau_sat_log)

    # ========================================================
    # Figure 1: task behavior
    # ========================================================
    fig = plt.figure(figsize=(20, 10))

    ax1 = fig.add_subplot(2, 3, 1, projection="3d")

    ax1.plot(tip_log[:, 0], tip_log[:, 1], tip_log[:, 2], label="Rod tip actual")

    ax1.plot(tipd_log[:, 0], tipd_log[:, 1], tipd_log[:, 2], label="Secondary reference")

    ax1.scatter(*p_rcm_target, s=50, label="RCM point")

    ax1.set_title("Rod tip / reference / RCM")

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.legend()

    ax2 = fig.add_subplot(2, 3, 2)

    ax2.plot(T_log, rcm_err_log * 1000.0)

    ax2.set_title("Main task: RCM error")

    ax2.set_xlabel("time [s]")

    ax2.set_ylabel("error [mm]")

    ax2.grid(True)

    ax3 = fig.add_subplot(2, 3, 3)

    ax3.plot(T_log, tip_err_actual_log * 1000.0)

    ax3.set_title("Secondary actual tip tracking error")

    ax3.set_xlabel("time [s]")

    ax3.set_ylabel("error [mm]")

    ax3.grid(True)

    ax4 = fig.add_subplot(2, 3, 4)

    ax4.plot(T_log, Etank_log)

    ax4.axhline(E_THRESH, linestyle="--", label="tank threshold")

    ax4.set_title("Energy tank")

    ax4.set_xlabel("time [s]")

    ax4.set_ylabel("Etank [J]")

    ax4.legend()
    ax4.grid(True)

    ax5 = fig.add_subplot(2, 3, 5)

    ax5.plot(T_log, V2_log)

    ax5.set_title("Secondary virtual potential V2")

    ax5.set_xlabel("time [s]")

    ax5.set_ylabel("V2 [J]")

    ax5.grid(True)

    ax6 = fig.add_subplot(2, 3, 6)

    ax6.plot(T_log, drift_log * 1000.0)

    ax6.set_title("Drift ||x2-x2_hat||")

    ax6.set_xlabel("time [s]")

    ax6.set_ylabel("drift [mm]")

    ax6.grid(True)

    plt.tight_layout()

    # ========================================================
    # Figure 2: main energy diagnostics
    # ========================================================
    fig2, axes = plt.subplots(4, 1, figsize=(14, 15))

    # --------------------------------------------------------
    # Diagnostic 1
    # --------------------------------------------------------
    axes[0].plot(T_log, Stotal_log - Stotal_log[0], label="S(t)-S(0)")

    axes[0].plot(T_log, Wext_log + Wref_log, "--", label="W_ext + W_ref")

    axes[0].plot(
        T_log, (Wext_log + Wref_log - Emj_passive_log), ":", label="W_ext + W_ref - E_MuJoCo,diss"
    )

    axes[0].set_title("Diagnostic 1: complete energy balance")

    axes[0].set_ylabel("energy [J]")

    axes[0].grid(True)
    axes[0].legend()

    # --------------------------------------------------------
    # Diagnostic 2
    # --------------------------------------------------------
    axes[1].plot(T_log, residual_raw_log, label="raw: S-S0-Wext-Wref")

    axes[1].plot(T_log, residual_mj_log, label=("corrected: " "S-S0-Wext-Wref+E_MuJoCo,diss"))

    axes[1].axhline(0.0, linestyle="--")

    axes[1].set_title("Diagnostic 2: energy-balance residual")

    axes[1].set_ylabel("residual [J]")

    axes[1].grid(True)
    axes[1].legend()

    # --------------------------------------------------------
    # Diagnostic 3
    # --------------------------------------------------------
    axes[2].plot(T_log, Ptank_theory_log, label="alpha*Pdiss + u^T F2")

    axes[2].plot(T_log, Ptank_numeric_log, "--", label="dEtank/dt numeric")

    axes[2].plot(T_log, tank_power_residual_log, ":", label="tank residual")

    axes[2].set_title("Diagnostic 3: energy-tank power balance")

    axes[2].set_ylabel("power [W]")

    axes[2].grid(True)
    axes[2].legend()

    # --------------------------------------------------------
    # Diagnostic 4
    # --------------------------------------------------------
    axes[3].plot(T_log, Emj_passive_log, label="- integral(qdot^T qfrc_passive) dt")

    axes[3].plot(T_log, Emj_damping_param_log, "--", label="integral sum(b_i*qdot_i^2) dt")

    axes[3].plot(T_log, Emj_friction_param_log, ":", label="frictionloss parameter estimate")

    axes[3].set_title("Diagnostic 4: MuJoCo passive dissipation")

    axes[3].set_xlabel("time [s]")

    axes[3].set_ylabel("dissipated energy [J]")

    axes[3].grid(True)
    axes[3].legend()

    plt.tight_layout()

    # ========================================================
    # Figure 3: passive power & compensation work diagnostics
    # ========================================================
    fig3, axes3 = plt.subplots(2, 1, figsize=(14, 9))

    axes3[0].plot(T_log, Pmujoco_passive_log, label="MuJoCo passive dissipated power")

    axes3[0].plot(T_log, Pdamping_param_log, "--", label="sum(b_i*qdot_i^2)")

    axes3[0].plot(T_log, Pfriction_param_log, ":", label="sum(frictionloss_i*|qdot_i|)")

    axes3[0].set_title("MuJoCo passive dissipated power")

    axes3[0].set_ylabel("power [W]")

    axes3[0].grid(True)
    axes3[0].legend()

    axes3[1].plot(T_log, Wcomp_log, label="integral(qdot^T tau_comp) dt")

    axes3[1].axhline(0.0, linestyle="--")

    axes3[1].set_title("Numerical mu_comp cumulative work")

    axes3[1].set_xlabel("time [s]")

    axes3[1].set_ylabel("work [J]")

    axes3[1].grid(True)
    axes3[1].legend()

    plt.tight_layout()
    plt.show()

    # ========================================================
    # Summary
    # ========================================================
    final_dS = Stotal_log[-1] - Stotal_log[0]

    final_Wsupply = Wext_log[-1] + Wref_log[-1]

    final_Emj = Emj_passive_log[-1]

    final_Edamp = Emj_damping_param_log[-1]

    final_Efric = Emj_friction_param_log[-1]

    print()
    print("=" * 78)
    print("ENERGY DIAGNOSTIC SUMMARY")
    print("=" * 78)

    print(f"mode                                = {TEST_MODE}")

    print(f"final S-S0                          = " f"{final_dS: .9e} J")

    print(f"final W_ext                         = " f"{Wext_log[-1]: .9e} J")

    print(f"final W_ref                         = " f"{Wref_log[-1]: .9e} J")

    print(f"final W_ext+W_ref                   = " f"{final_Wsupply: .9e} J")

    print("-" * 78)

    print(f"MuJoCo passive dissipated energy    = " f"{final_Emj: .9e} J")

    print(f"parameter viscous damping energy    = " f"{final_Edamp: .9e} J")

    print(f"parameter frictionloss energy       = " f"{final_Efric: .9e} J")

    print(f"passive-vs-damping difference       = " f"{final_Emj-final_Edamp: .9e} J")

    print("-" * 78)

    print(f"raw residual                        = " f"{residual_raw_log[-1]: .9e} J")

    print(f"MuJoCo-corrected residual           = " f"{residual_mj_log[-1]: .9e} J")

    print(f"max |MuJoCo-corrected residual|     = " f"{np.max(np.abs(residual_mj_log)): .9e} J")

    print("-" * 78)

    print(f"final numerical mu_comp work        = " f"{Wcomp_log[-1]: .9e} J")

    print(
        f"max |tank power residual|           = "
        f"{np.max(np.abs(tank_power_residual_log)): .9e} W"
    )

    print(f"max torque clipping                 = " f"{np.max(tau_sat_log): .9e} N m")

    print(f"clipping sample ratio               = " f"{np.mean(tau_sat_log > 1e-6): .3%}")

    print(f"final tank energy                   = " f"{Etank_log[-1]: .9f} J")

    print("=" * 78)

    print()
    print("Interpretation:")
    print(
        "1) If qfrc_passive dissipation and "
        "sum(b*qdot^2) almost coincide, the supplied XML's "
        "passive loss is essentially the joint viscous damping."
    )

    print(
        "2) If the previous ~3.22 J raw deficit is mainly MuJoCo damping, "
        "the corrected residual should move close to zero."
    )

    print(
        "3) If a sizable corrected residual remains, inspect W_comp next. "
        "The approximate mu_comp implementation can perform nonzero net work."
    )


if __name__ == "__main__":
    main()
