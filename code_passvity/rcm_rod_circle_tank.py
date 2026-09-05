import numpy as np
import mujoco
import mujoco.viewer
from numpy.linalg import norm
import matplotlib.pyplot as plt


# ============================================================
# Trajectory helpers
# ============================================================
def smoothstep_quintic(tau: float):
    tau = float(np.clip(tau, 0.0, 1.0))
    s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    ds = 30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4
    return s, ds


def circle_ref(t, center, plane_u, plane_v, r_circ, omega_circ, T_ramp):
    """圆形轨迹，用半径斜坡实现平滑起步（起点位置和速度都连续为0）。
    theta = omega_circ * t 匀速转动；r_eff 从 0 平滑爬升到 r_circ。
    """
    ramp, dramp = smoothstep_quintic(np.clip(t / T_ramp, 0.0, 1.0))
    r_eff = r_circ * ramp
    r_eff_dot = r_circ * dramp / T_ramp

    theta = omega_circ * t
    theta_dot = omega_circ
    c, si = np.cos(theta), np.sin(theta)

    pos = center + r_eff * c * plane_u + r_eff * si * plane_v
    vel = (r_eff_dot * c - r_eff * theta_dot * si) * plane_u + (
        r_eff_dot * si + r_eff * theta_dot * c
    ) * plane_v
    return pos, vel


# ============================================================
# Utils (与原脚本一致)
# ============================================================
def full_mass_matrix(m, d, nv_use=7):
    M_full = np.zeros((m.nv, m.nv), dtype=float)
    mujoco.mj_fullM(m, M_full, d.qM)
    return M_full[:nv_use, :nv_use].copy()


def dyn_consistent_projector(J, M, eps=1e-8):
    Minv = np.linalg.inv(M)
    Lambda_inv = J @ Minv @ J.T
    Lambda = np.linalg.inv(Lambda_inv + eps * np.eye(J.shape[0]))
    Jbar = Minv @ J.T @ Lambda
    N = np.eye(M.shape[0]) - Jbar @ J
    return Jbar, N


def nullspace_basis_raw(J, tol=1e-10):
    U, S, Vt = np.linalg.svd(J, full_matrices=True)
    r = np.sum(S > tol)
    V2 = Vt.T[:, r:]
    Z2 = V2.T
    return Z2


def align_nullspace_basis(Z2_raw, Z2_prev):
    if Z2_prev is None:
        return Z2_raw
    A = Z2_raw @ Z2_prev.T
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    return R.T @ Z2_raw


def compute_J2bar(J, M, Z2_prev=None, tol=1e-10, eps=1e-8):
    Z2_raw = nullspace_basis_raw(J, tol=tol)
    if Z2_raw.shape[0] == 0:
        return np.zeros((0, M.shape[0])), None
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
    XML_PATH = "model/franka_emika_panda/scene_panda_nohand.xml"
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)

    ee_site_name = "ee_site"
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
    if ee_site_id < 0:
        raise RuntimeError(f"Cannot find site '{ee_site_name}'.")

    joint3_name = "joint3"
    jid3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint3_name)
    if jid3 < 0:
        raise RuntimeError(f"Cannot find joint named '{joint3_name}' in the model.")
    dof3 = int(model.jnt_dofadr[jid3])
    print(f"[PulseTorque] joint3='{joint3_name}' jid={jid3}, dof_index={dof3}")

    key_name = "home"
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    # ------------------------------------------------------------------
    # 手术杆几何设定
    #
    #   末端法兰(ee_site) ----10cm----> 穿刺点(RCM, 固定) ----10cm----> 杆末端(画圆)
    #                     |<--------------- 20cm (L_ROD) -------------->|
    #
    # 杆沿 ee_site 局部坐标系的某个轴方向延伸。
    # !!! 假设该轴是局部 z 轴 (旋转矩阵第 3 列)。如果实际不是，改 AXIS_LOCAL_IDX。
    # ------------------------------------------------------------------
    AXIS_LOCAL_IDX = 2
    L_ROD = 0.20  # 杆总长 [m]
    D_RCM = 0.10  # 穿刺点距末端法兰的距离 [m]（杆中点）

    p_ee0 = data.site_xpos[ee_site_id].copy()
    R_ee0 = data.site_xmat[ee_site_id].reshape(3, 3).copy()
    axis_world0 = R_ee0[:, AXIS_LOCAL_IDX].copy()

    # 主任务目标：穿刺点位置，在初始构型下算一次，此后固定不变
    p_rcm_target = p_ee0 + D_RCM * axis_world0

    # 次任务参考系：杆末端初始位置，作为画圆的圆心
    p_tip0 = p_ee0 + L_ROD * axis_world0

    # 画圆所在平面：垂直于初始杆方向
    axis0 = axis_world0 / norm(axis_world0)
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, axis0)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    plane_u = tmp - np.dot(tmp, axis0) * axis0
    plane_u /= norm(plane_u)
    plane_v = np.cross(axis0, plane_u)

    r_circ = 0.03  # 圆的半径 [m]
    f_circ = 0.15  # 转动频率 [Hz]
    omega_circ = 2.0 * np.pi * f_circ
    T_ramp = 2.0  # 幅值爬升时间 [s]

    # 主任务增益 (穿刺点位置)
    Kp_task1 = np.array([600.0, 600.0, 600.0])
    Kd_task1 = 0.707 * np.sqrt(Kp_task1)

    # 次任务增益 (杆末端位置)
    Kp_task2 = np.array([300.0, 300.0, 300.0])
    Kd_task2 = 0.707 * np.sqrt(Kp_task2)

    ctrl_lo = model.actuator_ctrlrange[:7, 0].copy()
    ctrl_hi = model.actuator_ctrlrange[:7, 1].copy()

    m1, m2 = 3, 3  # 主任务(穿刺点位置,3维)，次任务(杆末端位置,3维)

    # ---------------- 能量罐 ----------------
    x2_hat = np.zeros(m2)
    x2_hat_inited = False
    E_tank, E_max, E_thresh = 15.0, 50.0, 5.0
    s_eps, alpha = 1e-8, 1.0
    s = np.sqrt(2 * E_tank)
    Kp_drift = np.array([15.0, 15.0, 15.0])

    # ---- 外部"人手"扰动脉冲 (仍加在 joint3 上) ----
    tau_pulse_amp = 0.0  # [N m] 非零脉冲，用于验证外部功与储能变化
    pulse_T = 0.5
    pulse_gap = 5.0
    n_pulses = 6
    t_first_pulse = 2.5
    pulse_starts = [t_first_pulse + k * (pulse_T + pulse_gap) for k in range(n_pulses)]

    t_end = 35.0
    log_every = 5

    T_log, tip_log, tipd_log = [], [], []
    V1_log, V2_log = [], []
    rcm_err_log = []
    Etank_log, Stotal_log, Stotal_corr_log = [], [], []
    Pdiss_log, Pex_log, dS_log = [], [], []
    Pext_log, drift_log = [], []

    # ---------- 三组能量诊断 ----------
    # 1) 外部累计做功 Wext 与总储能增量 S(t)-S(0)
    # 2) 被动性平衡残差 S(t)-S(0)-Wext
    # 3) 能量罐功率平衡 dEtank/dt 与 alpha*Pdiss + u^T F2
    Wext_log, passivity_balance_log = [], []
    Ptank_theory_log, Ptank_numeric_log = [], []
    tank_power_residual_log = []
    tau_sat_log = []

    Wext = 0.0
    S0 = None
    total_steps = 0
    S_prev, t_prev = None, None
    Sref_int = 0.0  # 固定参考时应保持为 0

    ENABLE_TAU_COMP = True
    MUCOMP_WARMUP_SEC = 0.5
    MUCOMP_LPF_BETA = 0.9
    tau_comp_prev = np.zeros(7)

    Z2_track = None

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.5
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20

        while viewer.is_running():
            mujoco.mj_forward(model, data)

            q = data.qpos[:7].copy()
            qd = data.qvel[:7].copy()
            t = float(data.time)
            if t >= t_end:
                break

            # ---------- Jacobians (末端法兰 ee_site) ----------
            J_pos_full = np.zeros((3, model.nv), dtype=float)
            J_rot_full = np.zeros((3, model.nv), dtype=float)
            mujoco.mj_jacSite(model, data, J_pos_full, J_rot_full, ee_site_id)
            J_pos = J_pos_full[:, :7].copy()
            J_rot = J_rot_full[:, :7].copy()

            p_ee = data.site_xpos[ee_site_id].copy()
            R_ee = data.site_xmat[ee_site_id].reshape(3, 3).copy()
            axis_world = R_ee[:, AXIS_LOCAL_IDX]

            skew_axis = skew(axis_world)

            # ---------- 主任务：穿刺点 (刚体上固定偏移点) ----------
            p_rcm = p_ee + D_RCM * axis_world
            J1 = J_pos - D_RCM * skew_axis @ J_rot  # RCM 点的雅可比

            e1 = p_rcm - p_rcm_target
            xdot1 = J1 @ qd
            F1 = -Kp_task1 * e1 - Kd_task1 * xdot1
            tau_task = J1.T @ F1

            # ---------- 次任务：杆末端 ----------
            p_tip = p_ee + L_ROD * axis_world
            J2 = J_pos - L_ROD * skew_axis @ J_rot  # 杆末端点的雅可比

            # ---------- projector ----------
            M = full_mass_matrix(model, data, nv_use=7)
            J1Mplus, _ = dyn_consistent_projector(J1, M, eps=1e-8)

            J2bar, Z2 = compute_J2bar(J1, M, Z2_prev=Z2_track, tol=1e-10, eps=1e-8)
            if Z2 is None or Z2.shape[0] == 0:
                print("warning: null space empty")
                tau_raw = np.clip(tau_task, ctrl_lo, ctrl_hi)
                data.ctrl[:7] = tau_raw
                mujoco.mj_step(model, data)
                viewer.sync()
                total_steps += 1
                continue
            Z2_track = Z2

            # w1 / w2 分解 (Dietrich et al. 2016, eq.11)
            w1 = J2 @ J1Mplus @ xdot1
            v2 = J2bar @ qd
            w2 = J2 @ Z2.T @ v2

            x2 = p_tip.copy()  # 次任务坐标 = 杆末端位置

            if not x2_hat_inited:
                x2_hat = x2.copy()
                x2_hat_inited = True

            w1_ref = w1 + Kp_drift * (x2 - x2_hat)
            allow_tap = (E_tank > E_thresh) and (s > s_eps)

            if allow_tap:
                n_gain = w1_ref / s
            else:
                n_gain = np.zeros(m2)

            u = n_gain * s
            x2hat_dot = u + w2
            x2_hat = x2_hat + x2hat_dot * dt

            # ---------- 被动性基础验证：固定次任务参考 ----------
            # 固定参考不会通过 x2_d_dot 向系统额外注入轨迹能量。

            x2_d, x2_d_dot = circle_ref(t, p_tip0, plane_u, plane_v, r_circ, omega_circ, T_ramp)
            # x2_d = p_tip0.copy()
            # x2_d_dot = np.zeros(3)
            e2 = x2_hat - x2_d
            F2 = -Kp_task2 * e2 + Kd_task2 * (x2_d_dot - x2hat_dot)

            tau_null = J2bar.T @ Z2 @ J2.T @ F2

            # ---------- 参考轨迹运动带来的"虚拟功率"修正 ----------
            # 主任务(RCM点)目标固定不动 => Pref1 = 0，不需要修正。
            # 次任务(杆末端)在追踪一个持续运动的圆 x2_d(t)，
            # 这部分参考速度会给 Stotal 注入一个和 tau_ext 无关的功率项，
            # 需要单独积分扣除，才能干净地检验能量罐核算是否守恒。
            Pref2 = float(np.dot(-Kp_task2 * e2 + Kd_task2 * x2hat_dot, x2_d_dot))
            Sref_int += Pref2 * dt

            # ---------- 能量罐更新 ----------
            Pd1 = float(np.dot(xdot1, Kd_task1 * xdot1))
            Pd2 = float(np.dot(x2hat_dot, Kd_task2 * x2hat_dot))
            Pdiss = Pd1 + Pd2
            P_ex = float(np.dot(u, F2))

            # 理论罐功率：阻尼耗散回收 + 次任务与罐之间的功率交换
            P_tank_theory = alpha * Pdiss + P_ex
            E_tank_old = float(E_tank)

            if s > s_eps:
                s_dot = P_tank_theory / s
            else:
                s_dot = (alpha * Pdiss) / max(s_eps, 1e-12)

            s = float(np.clip(s + s_dot * dt, 0.0, np.sqrt(2.0 * E_max)))
            E_tank = 0.5 * s * s

            # 实际离散罐功率；与理论值的差反映积分误差/能量上下限截断
            P_tank_numeric = (E_tank - E_tank_old) / dt
            P_tank_residual = P_tank_numeric - P_tank_theory

            # ---------- mu_comp (跨层耦合补偿，数值近似) ----------
            tau_comp = np.zeros(7)
            g_7 = compute_g_only(model, data, nv_use=7)

            if ENABLE_TAU_COMP and (t > MUCOMP_WARMUP_SEC):
                Jbar_full = np.vstack([J1, J2bar])
                Minv = np.linalg.inv(M)

                Lambda1_inv = J1 @ Minv @ J1.T
                Lambda1 = np.linalg.inv(Lambda1_inv + 1e-8 * np.eye(m1))
                Lambda2_inv = Z2 @ M @ Z2.T
                Lambda2_blk = np.linalg.inv(Lambda2_inv + 1e-8 * np.eye(Z2.shape[0]))

                Lambda = np.zeros((7, 7))
                Lambda[:m1, :m1] = Lambda1
                Lambda[m1:, m1:] = Lambda2_blk

                qd_a = (J1Mplus @ xdot1).copy()
                qd_b = (Z2.T @ v2).copy()

                Cqd_a = compute_Cqd_given_qd(model, data, qd_a, g_7)
                Cqd_b = compute_Cqd_given_qd(model, data, qd_b, g_7)

                Minv_cqd_a = np.linalg.solve(M, Cqd_a)
                Minv_cqd_b = np.linalg.solve(M, Cqd_b)

                term_a = Jbar_full @ Minv_cqd_a
                term_b = Jbar_full @ Minv_cqd_b

                mu_v_a = Lambda @ term_a
                mu_v_b = Lambda @ term_b

                mu12_v2 = mu_v_b[:m1]
                mu21_v1 = mu_v_a[m1:]

                v_comp = np.zeros(7)
                v_comp[:m1] = mu12_v2
                v_comp[m1:] = mu21_v1

                tau_comp_raw1 = Jbar_full.T @ v_comp
                tau_comp_raw = clamp_tau(tau_comp_raw1, ctrl_lo, ctrl_hi, frac=0.35)
                tau_comp = MUCOMP_LPF_BETA * tau_comp_prev + (1.0 - MUCOMP_LPF_BETA) * tau_comp_raw
                tau_comp_prev = tau_comp.copy()

            tau_raw = tau_task + tau_null + g_7 + tau_comp
            tau = np.clip(tau_raw, ctrl_lo, ctrl_hi)
            data.ctrl[:7] = tau

            # ---------- 外部扰动脉冲 ----------
            data.qfrc_applied[:] = 0.0
            tau3 = 0.0
            for tk in pulse_starts:
                if tk <= t <= tk + pulse_T:
                    phase = (t - tk) / pulse_T
                    tau3 = tau_pulse_amp * np.sin(np.pi * phase)
                    break
            data.qfrc_applied[dof3] = tau3
            P_ext = float(np.dot(qd, data.qfrc_applied[:7]))

            # 每个仿真步累计外部输入功，不能只在降采样日志时积分
            Wext += P_ext * dt

            # ---------- Logging ----------
            if total_steps % log_every == 0:
                V1 = 0.5 * float(np.dot(e1, Kp_task1 * e1))
                V2_hat = 0.5 * float(np.dot(e2, Kp_task2 * e2))

                Ekin = 0.5 * float(qd.T @ M @ qd)
                Stotal = Ekin + V1 + V2_hat + float(E_tank)
                Stotal_corr = Stotal - Sref_int  # 固定参考时与 Stotal 相同

                if S0 is None:
                    S0 = Stotal
                passivity_balance = Stotal - S0 - Wext

                if (S_prev is None) or (t_prev is None) or (t <= t_prev):
                    dSdt = 0.0
                else:
                    dSdt = (Stotal - S_prev) / (t - t_prev)
                S_prev, t_prev = Stotal, t

                drift = float(norm(x2 - x2_hat))

                T_log.append(t)
                tip_log.append(p_tip.copy())
                tipd_log.append(x2_d.copy())
                V1_log.append(V1)
                V2_log.append(V2_hat)
                rcm_err_log.append(float(norm(e1)))
                Etank_log.append(float(E_tank))
                Stotal_log.append(Stotal)
                Stotal_corr_log.append(Stotal_corr)
                Pdiss_log.append(Pdiss)
                Pex_log.append(P_ex)
                dS_log.append(dSdt)
                Pext_log.append(P_ext)
                drift_log.append(drift)

                Wext_log.append(Wext)
                passivity_balance_log.append(passivity_balance)
                Ptank_theory_log.append(P_tank_theory)
                Ptank_numeric_log.append(P_tank_numeric)
                tank_power_residual_log.append(P_tank_residual)
                tau_sat_log.append(float(norm(tau_raw - tau)))

            if total_steps % 200 == 0:
                print(
                    f"t={t:5.2f}  allow_tap={allow_tap}  E_tank={E_tank:6.2f}  "
                    f"|RCM err|={norm(e1)*1000:.3f}mm  |tip err|={norm(e2)*1000:.2f}mm"
                )

            mujoco.mj_step(model, data)
            viewer.sync()
            total_steps += 1

    # ============================================================
    # Plotting
    # ============================================================
    if len(T_log) == 0:
        print("[WARN] No data logged; skip plotting.")
        return

    T_log = np.asarray(T_log)
    tip_log = np.asarray(tip_log)
    tipd_log = np.asarray(tipd_log)
    V1_log = np.asarray(V1_log)
    V2_log = np.asarray(V2_log)
    rcm_err_log = np.asarray(rcm_err_log)
    Etank_log = np.asarray(Etank_log)
    Stotal_log = np.asarray(Stotal_log)
    Stotal_corr_log = np.asarray(Stotal_corr_log)
    drift_log = np.asarray(drift_log)
    Pdiss_log = np.asarray(Pdiss_log)
    Pex_log = np.asarray(Pex_log)
    Pext_log = np.asarray(Pext_log)

    Wext_log = np.asarray(Wext_log)
    passivity_balance_log = np.asarray(passivity_balance_log)
    Ptank_theory_log = np.asarray(Ptank_theory_log)
    Ptank_numeric_log = np.asarray(Ptank_numeric_log)
    tank_power_residual_log = np.asarray(tank_power_residual_log)
    tau_sat_log = np.asarray(tau_sat_log)

    fig = plt.figure(figsize=(20, 10))

    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    ax1.plot(tip_log[:, 0], tip_log[:, 1], tip_log[:, 2], label="Rod tip actual")
    ax1.plot(tipd_log[:, 0], tipd_log[:, 1], tipd_log[:, 2], label="Circle reference")
    ax1.scatter(*p_rcm_target, color="r", s=50, label="RCM point (fixed)")
    ax1.set_title("Rod tip vs. circle reference, RCM point")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.legend()

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(T_log, rcm_err_log * 1000.0)
    ax2.set_title("Main task: RCM point error ||e1|| [mm]")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("error [mm]")
    ax2.grid(True)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(T_log, V2_log)
    ax3.set_title("Secondary task potential V2 (tip circle tracking)")
    ax3.set_xlabel("time [s]")
    ax3.set_ylabel("V2 [J]")
    ax3.grid(True)

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(T_log, Etank_log)
    ax4.set_title("Energy tank Etank")
    ax4.set_xlabel("time [s]")
    ax4.set_ylabel("Etank [J]")
    ax4.grid(True)

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(T_log, Stotal_log, label="S (raw)")
    ax5.plot(T_log, Stotal_corr_log, label="S_corr (扣除参考轨迹虚拟功率)")
    ax5.set_title("Total energy: raw vs. reference-corrected")
    ax5.set_xlabel("time [s]")
    ax5.set_ylabel("S [J]")
    ax5.legend()
    ax5.grid(True)

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(T_log, drift_log * 1000.0)
    ax6.set_title("Drift ||x2 - x2_hat|| [mm]")
    ax6.set_xlabel("time [s]")
    ax6.set_ylabel("drift [mm]")
    ax6.grid(True)

    plt.tight_layout()

    # ============================================================
    # 三组能量诊断图
    # ============================================================
    fig_diag, axes = plt.subplots(3, 1, figsize=(13, 12))

    # 诊断 1：外部累计做功与总储能增量
    axes[0].plot(T_log, Stotal_log - Stotal_log[0], label="S(t) - S(0)")
    axes[0].plot(T_log, Wext_log, "--", label="integral P_ext dt")
    axes[0].set_title("Diagnostic 1: external work vs. stored-energy change")
    axes[0].set_ylabel("energy [J]")
    axes[0].grid(True)
    axes[0].legend()

    # 诊断 2：被动性累计能量平衡残差
    axes[1].plot(T_log, passivity_balance_log, label="S(t)-S(0)-integral P_ext dt")
    axes[1].axhline(0.0, linestyle="--")
    axes[1].set_title("Diagnostic 2: passivity energy-balance residual")
    axes[1].set_ylabel("residual [J]")
    axes[1].grid(True)
    axes[1].legend()

    # 诊断 3：能量罐理论功率与数值功率
    axes[2].plot(T_log, Ptank_theory_log, label="alpha*Pdiss + u^T F2")
    axes[2].plot(T_log, Ptank_numeric_log, "--", label="dEtank/dt (numeric)")
    axes[2].plot(T_log, tank_power_residual_log, ":", label="tank power residual")
    axes[2].set_title("Diagnostic 3: energy-tank power balance")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("power [W]")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    print("=" * 70)
    print("Energy diagnostic summary")
    print(f"final S-S0             = {Stotal_log[-1] - Stotal_log[0]: .6e} J")
    print(f"final integral P_ext   = {Wext_log[-1]: .6e} J")
    print(f"final balance residual = {passivity_balance_log[-1]: .6e} J")
    print(f"max |balance residual| = {np.max(np.abs(passivity_balance_log)): .6e} J")
    print(f"max |tank residual|    = {np.max(np.abs(tank_power_residual_log)): .6e} W")
    print(f"max torque clipping    = {np.max(tau_sat_log): .6e} N m")
    print(f"clipping sample ratio  = {np.mean(tau_sat_log > 1e-6): .3%}")
    print("Simulation finished + plots generated.")


if __name__ == "__main__":
    main()
