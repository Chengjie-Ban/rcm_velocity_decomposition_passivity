import numpy as np
import mujoco
import mujoco.viewer
from numpy.linalg import norm
import matplotlib.pyplot as plt


# Trajectory
# ============================================================
def smoothstep_quintic(tau: float):
    s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    ds = 30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4
    return 0, 0


def s_curve_pingpong_traj(t: float, p0: np.ndarray, p1: np.ndarray, T: float):
    if T <= 1e-6:
        return p1.copy(), np.zeros_like(p1)
    t_mod = t % (2.0 * T)
    if t_mod <= T:
        tau = np.clip(t_mod / T, 0.0, 1.0)
        s, ds = smoothstep_quintic(tau)
        pos = p0 + s * (p1 - p0)
        vel = (ds / T) * (p1 - p0)
    else:
        t2 = t_mod - T
        tau = np.clip(t2 / T, 0.0, 1.0)
        s, ds = smoothstep_quintic(tau)
        pos = p1 + s * (p0 - p1)
        vel = (ds / T) * (p0 - p1)
    return pos.astype(float), vel.astype(float)


# ============================================================
# Utils
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
    V2 = Vt.T[:, r:]  # n*n-r
    Z2 = V2.T
    return Z2  # n-r*n


def align_nullspace_basis(Z2_raw, Z2_prev):  # 保持z2连续
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


def make_selection_matrix(n, idx):
    m = len(idx)
    S = np.zeros((m, n), dtype=float)
    for r, j in enumerate(idx):
        S[r, j] = 1.0
    return S


# ---------- 关键：稳定 μcomp 需要的“在指定 qd 下重算 Cqd” ----------
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

    key_name = "home"
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    # gains
    Kp_task = np.array([500.0, 500.0, 500.0], dtype=float)
    Kd_task = 0.707 * np.sqrt(Kp_task)

    Kp_null = np.array([25, 25, 15, 15, 12, 9, 6], dtype=float)
    Kd_null = 0.707 * np.sqrt(Kp_null)
    q_nominal = np.array([0, 0.5, 0, -2.0, 0, 2.5, -0.785], dtype=float)
    # 0, 0.5, 0, -2.0, 0, 2.5, -0.785
    # 0.5, 0, 0.5, 0, 0, 2.5, -0.785

    ctrl_lo = model.actuator_ctrlrange[:7, 0].copy()
    ctrl_hi = model.actuator_ctrlrange[:7, 1].copy()

    # main task
    t_hold = 0
    p0 = data.site_xpos[ee_site_id].copy() + np.array([0.1, 0.0, 0.1], dtype=float)
    p1 = p0  # + np.array([0.2, 0.0, 0.2], dtype=float)
    T_s = 1.2

    # task dims
    m1, m2 = 3, 4

    # secondary task: x2 = J2 q
    idx = [
        1,
        2,
        3,
        4,
    ]
    J2 = make_selection_matrix(7, idx)  # (4,7)J2没问题

    # energy tank
    x2_hat = np.zeros(m2)
    x2_hat_inited = False
    E_tank, E_max, E_thresh = 15.0, 50.0, 14.0
    s_eps, alpha = 1e-8, 1
    s = np.sqrt(2 * E_tank)
    Kp_drift = np.array([10, 10, 10, 10], dtype=float)

    # ============================================================
    # Logging
    # ============================================================
    t_end = 3.0
    log_every = 5

    T_log = []
    y_log = []
    yd_log = []
    V1_log = []
    V2_log = []
    qerr_log = []
    Etank_log = []
    Stotal_log = []

    Pdiss_log = []
    Pex_log = []
    dS_log = []
    sat_log = []
    Pext_log = []
    qd_tau_sat_log = []
    drift_log = []

    total_steps = 0
    S_prev = None
    t_prev = None
    Sref_int = 0.0
    Ekin_prev = None

    # μcomp settings
    ENABLE_TAU_COMP = True
    MUCOMP_WARMUP_SEC = 0.5
    MUCOMP_LPF_BETA = 0.9
    tau_comp_prev = np.zeros(7)
    Jbar_full_prev = None

    # nullspace tracking
    Z2_track = None
    S_prev = None
    print_every = 10  # print every N sim steps

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

            # Jacobian
            J_pos_full = np.zeros((3, model.nv), dtype=float)
            J_rot_full = np.zeros((3, model.nv), dtype=float)
            mujoco.mj_jacSite(model, data, J_pos_full, J_rot_full, ee_site_id)
            J = J_pos_full[:, :7].copy()

            # desired
            if t < t_hold:
                y_d = p0.copy()
                yd_d = np.zeros(3)
            else:
                y_d, yd_d = s_curve_pingpong_traj(t - t_hold, p0=p0, p1=p1, T=T_s)
            # y_d, yd_d都是0

            # 任务1 y_d = ？设置一个y_d做阻抗控制

            # 主任务 没问题 yd_d = 0
            y = data.site_xpos[ee_site_id].copy()
            e = y - y_d
            xdot1 = J @ qd
            gradV = Kp_task * e
            F1 = -gradV - Kd_task * xdot1
            tau_task = J.T @ F1

            # dynamics
            M = full_mass_matrix(model, data, nv_use=7)
            J1Mplus, _ = dyn_consistent_projector(J, M, eps=1e-8)
            J1bar = J
            # Z2, J2bar
            J2bar, Z2 = compute_J2bar(J, M, Z2_prev=Z2_track, tol=1e-10, eps=1e-8)
            if Z2 is None or Z2.shape[0] != m2:
                print("warning")
                tau_raw = np.clip(tau_task, ctrl_lo, ctrl_hi)
                data.ctrl[:7] = tau_raw
                mujoco.mj_step(model, data)
                viewer.sync()
                total_steps += 1
                continue
            Z2_track = Z2

            # w1/w2
            w1 = J2 @ J1Mplus @ xdot1  # (4,)
            a = J2 @ qd
            v2 = J2bar @ qd
            w2 = J2 @ Z2.T @ v2

            # 引入能量池
            x2 = J2 @ q

            if not x2_hat_inited:
                x2_hat = x2.copy()
                x2_hat_inited = True

            w1_ref = w1 + (Kp_drift * (x2 - x2_hat))
            allow_tap = (E_tank > E_thresh) and (s > s_eps)

            if allow_tap:
                n = w1_ref / s
            else:

                n = np.array([0, 0, 0, 0])

            # n = w1_ref / s

            u = n * s

            x2hat_dot = u + w2
            x2_hat = x2_hat + x2hat_dot * dt

            x2_nom = J2 @ q_nominal  # x2的标准点就是q_nominal
            Kp2 = Kp_null[:m2]
            Kd2 = Kd_null[:m2]
            F2 = (Kp2 * (x2_nom - x2_hat)) - (Kd2 * x2hat_dot)

            # 副任务
            tau_null = J2bar.T @ Z2 @ J2.T @ F2

            # 能量变化
            Pd1 = float(np.dot(xdot1, Kd_task * xdot1))
            Pd2 = float(np.dot(x2hat_dot, Kd2 * x2hat_dot))
            Pdiss = Pd1 + Pd2
            P_ex = float(np.dot(u, F2))

            if s > s_eps:
                s_dot = (alpha * Pdiss + P_ex) / s
            else:
                s_dot = (alpha * Pdiss) / max(s_eps, 1e-12)

            s = float(np.clip(s + s_dot * dt, 0.0, np.sqrt(2.0 * E_max)))
            E_tank = 0.5 * s * s

            """
            alpha = 1
            E_tank_dot = alpha * float(np.dot(xdot1, Kd_task * xdot1) +
                  alpha * np.dot(x2hat_dot, Kd2 * x2hat_dot) +
                   np.dot(u, F2))
            E_tank = float(np.clip(E_tank + E_tank_dot * dt, 0.0, E_max))
            s = float(np.sqrt(2.0 * E_tank))
            """

            # ============================================================
            # μcomp 用两个任务分量算偏置效应，只拿对各自任务的影响
            # ============================================================
            tau_comp = np.zeros(7)

            # 只算一次 g（避免后面又 compute_g_only 一次）
            g_7 = compute_g_only(model, data, nv_use=7)

            if ENABLE_TAU_COMP and (t > MUCOMP_WARMUP_SEC):
                Jbar_full = np.vstack([J, J2bar])

                if Jbar_full_prev is None:
                    Jbar_dot = np.zeros((7, 7))
                else:
                    Jbar_dot = (Jbar_full - Jbar_full_prev) / dt
                Jbar_full_prev = Jbar_full.copy()

                Minv = np.linalg.inv(M)

                Lambda1_inv = J @ Minv @ J.T
                Lambda1 = np.linalg.inv(Lambda1_inv + 1e-8 * np.eye(m1))
                Lambda2_inv = Z2 @ M @ Z2.T
                Lambda2_blk = np.linalg.inv(Lambda2_inv + 1e-8 * np.eye(m2))

                Lambda = np.zeros((7, 7))
                Lambda[:m1, :m1] = Lambda1
                Lambda[m1:, m1:] = Lambda2_blk

                qd_a = (J1Mplus @ xdot1).copy()
                qd_b = (Z2.T @ v2).copy()

                Cqd_a = compute_Cqd_given_qd(model, data, qd_a, g_7)
                Cqd_b = compute_Cqd_given_qd(model, data, qd_b, g_7)

                Minv_cqd_a = np.linalg.solve(M, Cqd_a)
                Minv_cqd_b = np.linalg.solve(M, Cqd_b)

                Jbar_dot = Jbar_dot * 0

                term_a = (Jbar_full @ Minv_cqd_a) - (Jbar_dot @ qd_a)
                term_b = (Jbar_full @ Minv_cqd_b) - (Jbar_dot @ qd_b)

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

            # ============================================================
            # 补齐 debug 输出需要的量 + tau_comp
            # ============================================================
            tau_raw = tau_task + tau_null + g_7 + tau_comp

            tau = np.clip(tau_raw, ctrl_lo, ctrl_hi)

            sat_amount = float(norm(tau_raw - tau))  # 是否有饱和
            P_act = float(np.dot(qd, tau))  # qd·tau_sat
            P_ext = float(np.dot(qd, data.qfrc_applied[:7]))  # 如果没施加外力，一般就是 0

            data.ctrl[:7] = tau

            # ============================================================
            # Logging
            # ============================================================
            if total_steps % log_every == 0:
                V1 = 0.5 * float(np.dot(e, Kp_task * e))
                qerr = float(norm(q - q_nominal))

                #  V2_hat 应该在 x2 空间算：用 (x2_nom - x2_hat) 和 Kp2
                V2_hat = 0.5 * float(np.dot((x2_nom - x2_hat), Kp2 * (x2_nom - x2_hat)))

                Ekin = 0.5 * float(qd.T @ M @ qd)
                P_passive = qd @ data.qfrc_passive[:7]  # 阻尼 摩擦  <0
                P_bias = qd @ data.qfrc_bias[:7]
                Stotal = Ekin + V1 + V2_hat + float(E_tank)  # +  P_passive

                # numerical dS/dt
                if (S_prev is None) or (t_prev is None) or (t <= t_prev):
                    dSdt = 0.0
                else:
                    dSdt = (Stotal - S_prev) / (t - t_prev)

                S_prev = Stotal
                t_prev = t

                ##debug
                P_ctrl = qd @ tau  # 施加到系统的控制功率
                P_bias = qd @ data.qfrc_bias[:7]  # bias 的功率
                P_passive = qd @ data.qfrc_passive[:7]  # 阻尼 摩擦  <0
                P_contact = qd @ data.qfrc_constraint[:7]  # =0

                if Ekin_prev is None:
                    Ekindot_fd = 0.0
                else:
                    Ekindot_fd = (Ekin - Ekin_prev) / dt

                Ekin_prev = Ekin

                Ekindot_power = P_ctrl - (P_bias + P_passive + P_contact)

                error = Ekindot_fd - Ekindot_power
                # print(error)

                Pref1 = -float((Kp_task * e) @ yd_d)
                Sref_int += Pref1 * dt
                dSdt_corr = dSdt - Pref1
                Stotal_corr = Stotal - Sref_int
                drift = float(norm(x2 - x2_hat))

                T_log.append(t)
                y_log.append(y.copy())
                yd_log.append(y_d.copy())
                V1_log.append(V1)
                V2_log.append(V2_hat)
                qerr_log.append(qerr)
                Etank_log.append(float(E_tank))
                # Stotal =  Stotal_corr
                Stotal_log.append(Stotal)

                Pdiss_log.append(Pdiss)
                Pex_log.append(P_ex)
                dS_log.append(dSdt)
                sat_log.append(sat_amount)
                Pext_log.append(P_ext)
                qd_tau_sat_log.append(P_act)
                drift_log.append(drift)

            if total_steps % 200 == 0:
                print("true 副任务正常；false副任务有截断", allow_tap)

            mujoco.mj_step(model, data)
            viewer.sync()

            total_steps += 1

    print("\n===== Final Joint Angles =====")
    q_final = data.qpos[:7].copy()
    print("Full q vector:", q_final)

    # ============================================================
    # Plotting
    # ============================================================

    if len(T_log) == 0:
        print("[WARN] No data logged; skip plotting.")
        return

    T_log = np.asarray(T_log)
    y_log = np.asarray(y_log)
    yd_log = np.asarray(yd_log)
    V1_log = np.asarray(V1_log)
    V2_log = np.asarray(V2_log)
    qerr_log = np.asarray(qerr_log)
    Etank_log = np.asarray(Etank_log)
    Stotal_log = np.asarray(Stotal_log)
    drift_log = np.asarray(drift_log)

    """

    import os

    desktop_path = r"C:/Users/BCJ/Desktop/plots"
    os.makedirs(desktop_path, exist_ok=True)

    # ============================================================
    # Plot 1 : 3D End-effector
    # ============================================================
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot(y_log[:, 0], y_log[:, 1], y_log[:, 2], label="EE actual")
    ax1.plot(yd_log[:, 0], yd_log[:, 1], yd_log[:, 2], label="EE equilibrium")
    ax1.set_title("End-effector (3D)")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
    ax1.legend()
    fig1.savefig(os.path.join(desktop_path, "1_end_effector.png"))
    plt.close(fig1)

    # ============================================================
    # Plot 2 : V1
    # ============================================================
    fig2 = plt.figure()
    plt.plot(T_log, V1_log)
    plt.title("Main task potential energy V1")
    plt.xlabel("time [s]")
    plt.ylabel("V1 [J]")
    plt.grid(True)
    fig2.savefig(os.path.join(desktop_path, "2_V1.png"))
    plt.close(fig2)

    # ============================================================
    # Plot 3 : Joint error
    # ============================================================
    fig3 = plt.figure()
    plt.plot(T_log, qerr_log)
    plt.title("Subtask joint error")
    plt.xlabel("time [s]")
    plt.ylabel("error [rad]")
    plt.grid(True)
    fig3.savefig(os.path.join(desktop_path, "3_joint_error.png"))
    plt.close(fig3)

    # ============================================================
    # Plot 4 : Energy tank
    # ============================================================
    fig4 = plt.figure()
    plt.plot(T_log, Etank_log)
    plt.title("Energy tank")
    plt.xlabel("time [s]")
    plt.ylabel("Etank [J]")
    plt.grid(True)
    fig4.savefig(os.path.join(desktop_path, "4_Etank.png"))
    plt.close(fig4)

    # ============================================================
    # Plot 5 : Total energy
    # ============================================================
    fig5 = plt.figure()
    plt.plot(T_log, Stotal_log)
    plt.title("Total energy S")
    plt.xlabel("time [s]")
    plt.ylabel("S [J]")
    plt.grid(True)
    fig5.savefig(os.path.join(desktop_path, "5_Stotal.png"))
    plt.close(fig5)

    # ============================================================
    # Plot 6 : Drift
    # ============================================================
    fig6 = plt.figure()
    plt.plot(T_log, drift_log)
    plt.title("Drift norm")
    plt.xlabel("time [s]")
    plt.ylabel("drift [rad]")
    plt.grid(True)
    fig6.savefig(os.path.join(desktop_path, "6_drift.png"))
    plt.close(fig6)

    # ============================================================
    # Plot 7 : V2
    # ============================================================
    fig7 = plt.figure()
    plt.plot(T_log, V2_log)
    plt.title("Subtask potential energy V2")
    plt.xlabel("time [s]")
    plt.ylabel("V2 [J]")
    plt.grid(True)
    fig7.savefig(os.path.join(desktop_path, "7_V2.png"))
    plt.close(fig7)

    print("All plots saved to desktop.")


    """
    fig = plt.figure(figsize=(20, 10))

    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    ax1.plot(y_log[:, 0], y_log[:, 1], y_log[:, 2], label="EE actual")
    ax1.plot(yd_log[:, 0], yd_log[:, 1], yd_log[:, 2], label="EE equilibrium (setpoint)")
    ax1.set_title("End-effector (3D): actual vs equilibrium")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.set_ylim(-0.06, 0.06)
    ax1.set_yticks(np.arange(-0.05, 0.051, 0.02))
    ax1.legend()

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(T_log, V1_log)
    ax2.set_title("Main task potential energy V1")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("V1 [J]")
    ax2.grid(True)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(T_log, qerr_log)
    ax3.set_title("Subtask joint error ||q - q_nom||")
    ax3.set_xlabel("time [s]")
    ax3.set_ylabel("error [rad]")
    ax3.grid(True)

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(T_log, Etank_log)
    ax4.set_title("Energy tank Etank")
    ax4.set_xlabel("time [s]")
    ax4.set_ylabel("Etank [J]")
    ax4.grid(True)

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(T_log, Stotal_log)
    ax5.set_title("Total energy S = Ekin + V1 + V2(x_hat) + Etank")
    ax5.set_xlabel("time [s]")
    ax5.set_ylabel("S [J]")
    ax5.grid(True)

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(T_log, drift_log)
    ax6.set_title("Drift norm ||x2 - x2_hat||")
    ax6.set_xlabel("time [s]")
    ax6.set_ylabel("drift [rad]")
    ax6.grid(True)

    plt.figure()
    plt.plot(T_log, V2_log)
    plt.title("Subtask potential energy V2 (using x2_hat)")
    plt.xlabel("time [s]")
    plt.ylabel("V2 [J]")
    plt.grid(True)

    plt.show()

    print("=" * 70)
    print("Simulation finished + plots generated.")


if __name__ == "__main__":
    main()
