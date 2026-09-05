# RCM Velocity Passivity / RCM 速度控制无源性

[中文](#中文说明) · [English below](#english)

## 中文说明

本目录包含基于 MuJoCo 的 RCM（Remote Center of Motion，远程运动中心）速度控制、投影式零空间柔顺控制及能量罐无源化实验。系统以 RCM 点为主任务，以杆端/器械末端运动为次任务，并记录储能、外部功、参考轨迹功率、控制器阻尼耗散以及 MuJoCo 被动力等能量通道。

### 重要说明：耗散实验中的 α = 0

`rcm_energy_tank_depletion_intervention_demo.py` 明确设置：

```python
ALPHA = 0.0
```

能量罐内部功率关系为

```text
dE_tank/dt ≈ α P_diss_controller + uᵀF₂
```

因此在这个耗散/干预实验中，`α = 0` 表示控制器阻尼所耗散的能量不会被回收到能量罐：阻尼能量只耗散、不回收（tank without damping refill）。此设置会有意地让能量罐在次任务输出能量时逐步下降；当 `E_tank` 到达 `E_THRESH` 后，次任务通道被关闭或削弱，以维持无源性预算。

这里的“只耗散、不回收”专指阻尼耗散项 `P_diss_controller` 不用于给能量罐充能。能量罐仍通过 `uᵀF₂` 与次任务端口交换能量。

其他验证脚本保留各自的实验配置，其中部分脚本使用 `ALPHA = 1.0` 来研究阻尼能量回收；请勿将耗散演示中的 `α = 0` 结论泛化到所有文件。

### 代码结构

| 文件 | 用途 |
| --- | --- |
| `rcm_energy_tank_depletion_intervention_demo.py` | 推荐的能量耗尽与阈值干预演示；`ALPHA = 0.0`，阻尼能量不回收。 |
| `rcm_energy_tank_passivity_verification.py` | 无源性与能量平衡验证，支持固定参考、外部脉冲和圆轨迹模式。 |
| `rcm_energy_tank_with_mujoco_dissipation.py` | 在能量平衡中显式诊断 MuJoCo 关节阻尼等被动耗散。 |
| `rcm_rod_circle_tank.py` | 带能量罐的杆端圆轨迹控制实验。 |
| `final_ver_null_tank_w_force.py` | 含外力情形的零空间能量罐版本。 |
| `final_ver_null_tank_wo_force.py` | 无外力情形的零空间能量罐版本。 |
| `test_mobe.py` | 早期/辅助实验入口。 |
| `model/` | Franka Emika Panda 的 MuJoCo XML、网格与许可证。 |
| `code_passvity/` | 与根目录代码对应的镜像副本；修改实验参数时应同步维护。 |

### 运行

建议使用 Python 3.10 或更高版本，并安装：

```bash
pip install numpy matplotlib mujoco
```

从本目录运行脚本，以保证相对模型路径能够正确解析：

```bash
cd rcm_velocity_passvity
python rcm_energy_tank_depletion_intervention_demo.py
```

可在脚本顶部选择 `TEST_MODE`：

- `fixed_passivity`：固定主、次任务参考，用于基础无源性检查；
- `external_pulse`：固定参考并施加外部力矩脉冲；
- `circle_tracking`：移动次任务参考，参考轨迹作为附加功率端口。

主要可调参数包括任务刚度/阻尼、`E_TANK_INIT`、`E_MAX`、`E_THRESH`、轨迹半径与频率，以及外部脉冲参数。运行时会打开 MuJoCo viewer，并在实验结束后绘制轨迹、能量、功率和残差等诊断结果。

### 理论来源与引用

本项目的投影式零空间控制能量罐无源化方法参考以下论文。使用或扩展本代码时，请引用原作者：

> A. Dietrich, C. Ott and S. Stramigioli, “Passivation of Projection-Based Null Space Compliance Control Via Energy Tanks,” *IEEE Robotics and Automation Letters*, vol. 1, no. 1, pp. 184–191, Jan. 2016, doi: [10.1109/LRA.2015.2512937](https://doi.org/10.1109/LRA.2015.2512937).

```bibtex
@article{Dietrich2016Passivation,
  author  = {Alexander Dietrich and Christian Ott and Stefano Stramigioli},
  title   = {Passivation of Projection-Based Null Space Compliance Control Via Energy Tanks},
  journal = {IEEE Robotics and Automation Letters},
  volume  = {1},
  number  = {1},
  pages   = {184--191},
  month   = jan,
  year    = {2016},
  doi     = {10.1109/LRA.2015.2512937}
}
```

关键词：Null space；Manipulator dynamics；Damping；Aerospace electronics；Asymptotic stability；Compliance and Impedance Control；Redundant Robots。

## English

This directory contains MuJoCo experiments for RCM (Remote Center of Motion) velocity control, projection-based null-space compliance control, and energy-tank passivation. The RCM point is treated as the primary task and the rod/instrument tip motion as the secondary task. The scripts log stored energy, external work, reference-port power, controller damping dissipation, and MuJoCo passive-force channels.

### Important: α = 0 in the dissipation experiment

`rcm_energy_tank_depletion_intervention_demo.py` explicitly sets:

```python
ALPHA = 0.0
```

The internal tank-power relation is

```text
dE_tank/dt ≈ α P_diss_controller + uᵀF₂
```

Consequently, in this depletion/intervention experiment, `α = 0` means that energy dissipated by controller damping is not recovered into the energy tank: it is dissipated only, with no damping-energy refill. This intentionally allows the tank level to fall when the secondary task supplies energy. Once `E_tank` reaches `E_THRESH`, the secondary-task channel is disabled or attenuated to preserve the passivity budget.

“Dissipated only, not recovered” refers specifically to excluding `P_diss_controller` from tank charging. The tank can still exchange energy with the secondary-task port through `uᵀF₂`.

Other verification scripts retain their own experimental settings; some use `ALPHA = 1.0` to study damping-energy recovery. The `α = 0` statement for the depletion demo should therefore not be generalized to every script.

### Repository layout

| File | Purpose |
| --- | --- |
| `rcm_energy_tank_depletion_intervention_demo.py` | Recommended depletion and threshold-intervention demo; `ALPHA = 0.0`, with no damping-energy recovery. |
| `rcm_energy_tank_passivity_verification.py` | Passivity and energy-balance checks for fixed-reference, external-pulse, and circular-tracking modes. |
| `rcm_energy_tank_with_mujoco_dissipation.py` | Explicit diagnostics for MuJoCo passive dissipation, including joint damping. |
| `rcm_rod_circle_tank.py` | Circular rod-tip tracking with an energy tank. |
| `final_ver_null_tank_w_force.py` | Null-space energy-tank variant with external force. |
| `final_ver_null_tank_wo_force.py` | Null-space energy-tank variant without external force. |
| `test_mobe.py` | Earlier/supporting experimental entry point. |
| `model/` | Franka Emika Panda MuJoCo XML files, meshes, and license. |
| `code_passvity/` | Mirror of the root scripts; keep both copies synchronized when changing experiment settings. |

### Running the experiments

Python 3.10 or newer is recommended. Install the runtime dependencies with:

```bash
pip install numpy matplotlib mujoco
```

Run scripts from this directory so that the relative model path resolves correctly:

```bash
cd rcm_velocity_passvity
python rcm_energy_tank_depletion_intervention_demo.py
```

Select `TEST_MODE` near the top of a script:

- `fixed_passivity`: fixed primary and secondary references for a baseline passivity check;
- `external_pulse`: fixed references with external torque pulses;
- `circle_tracking`: a moving secondary reference treated as an additional power port.

Key parameters include task stiffness/damping, `E_TANK_INIT`, `E_MAX`, `E_THRESH`, circle radius/frequency, and external-pulse settings. A MuJoCo viewer opens during execution; trajectory, energy, power, and residual diagnostics are plotted afterward.

### Theoretical basis and citation

The energy-tank passivation used for projection-based null-space control in this project follows the work below. Please cite the original authors when using or extending this code:

> A. Dietrich, C. Ott and S. Stramigioli, “Passivation of Projection-Based Null Space Compliance Control Via Energy Tanks,” *IEEE Robotics and Automation Letters*, vol. 1, no. 1, pp. 184–191, Jan. 2016, doi: [10.1109/LRA.2015.2512937](https://doi.org/10.1109/LRA.2015.2512937).

```bibtex
@article{Dietrich2016Passivation,
  author  = {Alexander Dietrich and Christian Ott and Stefano Stramigioli},
  title   = {Passivation of Projection-Based Null Space Compliance Control Via Energy Tanks},
  journal = {IEEE Robotics and Automation Letters},
  volume  = {1},
  number  = {1},
  pages   = {184--191},
  month   = jan,
  year    = {2016},
  doi     = {10.1109/LRA.2015.2512937}
}
```

Keywords: Null space; Manipulator dynamics; Damping; Aerospace electronics; Asymptotic stability; Compliance and Impedance Control; Redundant Robots.
