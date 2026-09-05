import mujoco  # 这版本尝试画图，末端执行器的问题可能依旧存在
import numpy as np
import mujoco.viewer
import pinocchio
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 中文字体设置
rcParams["font.sans-serif"] = ["SimHei"]
rcParams["axes.unicode_minus"] = False


def circle_traj(t, center=np.array([0.45, 0.0, 0.5]), r=0.08, omega=0.5):
    """圆轨迹：在XY平面内画圆"""
    return np.array(
        [center[0] + r * np.cos(omega * t), center[1] + r * np.sin(omega * t), center[2]],
        dtype=float,
    )


class IKPinocchio:
    def __init__(self, urdf_filename: str, joint_id: int = 7):
        self.model = pinocchio.buildModelFromUrdf(urdf_filename)
        self.data = self.model.createData()
        self.JOINT_ID = joint_id

    def solve(self, q_seed, R_des, p_des, eps=1e-4, it_max=100, dt=0.1, damp=1e-6, verbose=False):
        q = q_seed.copy()
        oMdes = pinocchio.SE3(R_des, np.asarray(p_des, dtype=float))

        for i in range(it_max):
            pinocchio.forwardKinematics(self.model, self.data, q)
            iMd = self.data.oMi[self.JOINT_ID].actInv(oMdes)
            err = pinocchio.log(iMd).vector

            if norm(err) < eps:
                if verbose:
                    print(f"Converged in {i} iterations, ||err||={norm(err):.6f}")
                return q

            J = pinocchio.computeJointJacobian(self.model, self.data, q, self.JOINT_ID)
            J = -pinocchio.Jlog6(iMd.inverse()) @ J
            v = -J.T @ solve(J @ J.T + damp * np.eye(6), err)
            q = pinocchio.integrate(self.model, q, v * dt)

            if verbose and (i % 20 == 0):
                print(f"iter {i}, ||err||={norm(err):.6f}")

        if verbose:
            print(f"Warning: Max iterations reached, ||err||={norm(err):.6f}")
        return q


# ---- MuJoCo load
mj_model = mujoco.MjModel.from_xml_path("model/franka_emika_panda/panda.xml")
mj_data = mujoco.MjData(mj_model)

# ---- Pinocchio IK solver
ik = IKPinocchio("model/franka_panda_urdf/robots/panda_arm.urdf", joint_id=7)


class CustomViewer:
    def __init__(self, model, data, sim_duration=10.0):
        self.model = model
        self.data = data
        self.handle = mujoco.viewer.launch_passive(model, data)
        self.sim_duration = sim_duration

        # 初始关节角（7 DoF）
        self.initial_q = data.qpos[:7].copy()

        # 末端朝下：绕 X 轴转 π
        theta = np.pi
        self.R_des = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ],
            dtype=float,
        )

        # 获取关节限位
        self.arm_joint_names = [f"joint{i}" for i in range(1, 8)]
        self.jnt_qmin = np.zeros(7)
        self.jnt_qmax = np.zeros(7)
        for k, name in enumerate(self.arm_joint_names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"Cannot find joint '{name}' in MuJoCo model.")
            self.jnt_qmin[k] = model.jnt_range[jid, 0]
            self.jnt_qmax[k] = model.jnt_range[jid, 1]

        # 力矩限幅
        self.tau_min = model.actuator_ctrlrange[:7, 0].copy()
        self.tau_max = model.actuator_ctrlrange[:7, 1].copy()

        # PD控制增益
        self.Kp = np.array([100, 100, 80, 70, 50, 30, 20], dtype=float)
        self.Kd = 2.0 * np.sqrt(self.Kp)

        # 轨迹平滑
        self.q_des_prev = self.initial_q.copy()

        # 数据记录初始化
        self.dt = self.model.opt.timestep if self.model.opt.timestep > 0 else 0.002
        self.num_steps = int(self.sim_duration / self.dt)

        # 预分配存储空间
        self.time_axis = []
        self.target_positions = []
        self.actual_positions = []
        self.joint_angles = []
        self.control_signals = []

        # 查找末端执行器site
        self.ee_site_id = self._find_ee_site()

    def _find_ee_site(self):
        """查找末端执行器的site ID"""
        # 尝试几个常见的site名称
        possible_names = ["attachment_site", "link_tcp", "ee_site", "end_effector"]
        for name in possible_names:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id >= 0:
                print(f"找到末端执行器site: {name}")
                return site_id

        # 如果没找到，使用最后一个site
        if self.model.nsite > 0:
            print(f"使用默认site (ID: {self.model.nsite - 1})")
            return self.model.nsite - 1
        else:
            print("警告：未找到末端执行器site，将使用关节位置")
            return -1

    def is_running(self):
        return self.handle.is_running()

    def sync(self):
        self.handle.sync()

    @property
    def cam(self):
        return self.handle.cam

    def run_loop(self):
        t = 0.0
        q_seed = self.initial_q.copy()

        print("=" * 50)
        print("开始画圆轨迹...")
        print(f"仿真时长: {self.sim_duration}s")
        print(f"时间步长: {self.dt:.4f}s")
        print(f"预计步数: {self.num_steps}")
        print("=" * 50)

        step_count = 0
        while self.is_running() and t < self.sim_duration:
            # 1) Forward dynamics
            mujoco.mj_forward(self.model, self.data)

            # 2) 目标末端位置（圆轨迹）
            target_pos = circle_traj(t, center=np.array([0.45, 0.0, 0.5]), r=0.08, omega=0.5)

            # 3) IK求解
            q_des = ik.solve(
                q_seed, self.R_des, target_pos, it_max=100, dt=0.1, damp=1e-6, verbose=False
            )
            q_des = np.clip(q_des, self.jnt_qmin, self.jnt_qmax)

            # 轨迹平滑
            max_joint_vel = 0.5
            q_diff = q_des - self.q_des_prev
            q_diff = np.clip(q_diff, -max_joint_vel * self.dt, max_joint_vel * self.dt)
            q_des = self.q_des_prev + q_diff
            self.q_des_prev = q_des

            # 4) PD控制 + 重力补偿
            q = self.data.qpos[:7].copy()
            qd = self.data.qvel[:7].copy()
            bias = self.data.qfrc_bias[:7].copy()

            tau = self.Kp * (q_des - q) - self.Kd * qd + bias
            tau = np.clip(tau, self.tau_min, self.tau_max)

            self.data.ctrl[:7] = tau

            # 5) 获取实际末端位置
            if self.ee_site_id >= 0:
                actual_pos = self.data.site_xpos[self.ee_site_id].copy()
            else:
                # 如果没有site，使用Pinocchio前向运动学
                pinocchio.forwardKinematics(ik.model, ik.data, q)
                actual_pos = ik.data.oMi[ik.JOINT_ID].translation

            # 6) 记录数据
            self.time_axis.append(t)
            self.target_positions.append(target_pos.copy())
            self.actual_positions.append(actual_pos.copy())
            self.joint_angles.append(q.copy())
            self.control_signals.append(tau.copy())

            # 7) Step simulation
            mujoco.mj_step(self.model, self.data)
            self.sync()

            # 更新
            q_seed = q_des
            t += self.dt
            step_count += 1

            # 每秒打印一次进度
            if step_count % int(1.0 / self.dt) == 0:
                pos_err = norm(actual_pos - target_pos)
                print(f"t={t:.2f}s, 位置误差: {pos_err*1000:.2f}mm")

        print("\n仿真完成！开始绘制图表...")
        self.plot_results()

    def plot_results(self):
        """绘制仿真结果"""
        # 转换为numpy数组
        time_axis = np.array(self.time_axis)
        target_positions = np.array(self.target_positions)
        actual_positions = np.array(self.actual_positions)
        joint_angles = np.array(self.joint_angles)
        control_signals = np.array(self.control_signals)

        # 创建图表
        fig = plt.figure(figsize=(14, 10))

        # 1. 三维轨迹跟踪结果
        ax1 = fig.add_subplot(2, 2, 1, projection="3d")
        ax1.plot(
            target_positions[:, 0],
            target_positions[:, 1],
            target_positions[:, 2],
            "b--",
            label="目标轨迹",
            linewidth=2,
        )
        ax1.plot(
            actual_positions[:, 0],
            actual_positions[:, 1],
            actual_positions[:, 2],
            "r-",
            label="实际轨迹",
            alpha=0.7,
            linewidth=1.5,
        )
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_zlabel("Z (m)")
        ax1.set_title("末端执行器轨迹跟踪")
        ax1.legend()
        ax1.grid(True)

        # 2. 关节角度变化
        ax2 = fig.add_subplot(2, 2, 2)
        for j in range(7):
            ax2.plot(time_axis, np.degrees(joint_angles[:, j]), label=f"关节 {j+1}", linewidth=1.2)
        ax2.set_xlabel("时间 (s)")
        ax2.set_ylabel("关节角度 (°)")
        ax2.set_title("关节运动状态")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax2.grid(True)

        # 3. 控制信号
        ax3 = fig.add_subplot(2, 2, 3)
        for j in range(7):
            ax3.plot(time_axis, control_signals[:, j], label=f"关节 {j+1}", linewidth=1.2)
        ax3.set_xlabel("时间 (s)")
        ax3.set_ylabel("控制力矩 (N·m)")
        ax3.set_title("控制信号输出")
        ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax3.grid(True)

        # 4. 跟踪误差分析
        ax4 = fig.add_subplot(2, 2, 4)
        position_error = np.linalg.norm(actual_positions - target_positions, axis=1)
        ax4.plot(time_axis, position_error * 1000, "r-", linewidth=1.5)
        ax4.set_xlabel("时间 (s)")
        ax4.set_ylabel("跟踪误差 (mm)")
        ax4.set_title("位置跟踪误差")
        ax4.grid(True)

        # 统计信息
        mean_error = np.mean(position_error) * 1000
        max_error = np.max(position_error) * 1000
        ax4.text(
            0.02,
            0.98,
            f"平均误差: {mean_error:.2f}mm\n最大误差: {max_error:.2f}mm",
            transform=ax4.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.savefig("franka_circle_trajectory.png", dpi=150, bbox_inches="tight")
        print("图表已保存为 'franka_circle_trajectory.png'")
        plt.show()


# 主程序
if __name__ == "__main__":
    viewer = CustomViewer(mj_model, mj_data, sim_duration=10.0)
    viewer.cam.distance = 2.5
    viewer.cam.azimuth = 45
    viewer.cam.elevation = -20
    viewer.run_loop()
