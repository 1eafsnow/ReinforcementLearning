# MuJoCo 24自由度六足机器人 SAC 行走训练

本项目通过 `mjcf/scene.xml` 加载带地面的完整场景，并由场景相对引用 `mjcf/hexapod_model.xml`。每条腿4关节、共24动作，5个STL文件保持不变，MJCF中的模型和网格路径均为工程相对路径，不会在运行时写回模型文件。

## 环境与动作

- MuJoCo物理步长：`0.001s`（1000Hz）。
- `frame_skip=10`，策略控制周期：`0.01s`（100Hz）。
- 动作：24维归一化关节位置偏移，顺序为每条腿的 `coxa/femur/tibia/tarsus`。
- 观测：固定95维，包含机体姿态/速度、`command_x/y/pitch`、Yaw目标误差、24关节角和速度、6足接触、上一滤波动作及步态相位。
- 第一阶段训练80%任意方向移动和20%静止站立；Yaw目标保持reset时朝向，Pitch目标保持水平，但观测与奖励已经使用最终姿态Command结构。
- 静止Command只有在六只脚均与地面接触时才获得完整静止支撑奖励，且静止状态冻结步态相位。
- Tripod完整周期频率按目标平移速度从`1.10Hz`线性增加到`2.20Hz`；对应支撑组切换间隔从约`0.455s`缩短到`0.227s`。

## Command定义

- `command_x`：机器人实时Yaw坐标系中的纵向目标速度，正值前进。
- `command_y`：机器人实时Yaw坐标系中的横向目标速度，正值向左。
- `command_yaw`：世界坐标系目标Yaw角，使用 `sin(yaw_error), cos(yaw_error)` 输入策略。
- `command_pitch`：相对水平面的目标Pitch角，Pitch不参与水平速度坐标系转换。

平移速度每一步都按当前实际Yaw解释，因此Yaw改变后，即使 `command_x/y` 不变，机器人在世界坐标系中的移动方向也会同步改变。`command_x/y=0` 时支持原地转向、原地调整Pitch和静止站立。

所有环境、奖励、SAC和训练参数集中在 `config.py`。修改参数时不需要进入 `env.py`、`sac.py` 或 `train.py`。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell激活命令：

```powershell
.venv\Scripts\Activate.ps1
```

## 先做动态测试

```bash
python smoke_test.py --steps 1000
python smoke_test.py --steps 1000 --randomized-reset
python smoke_test.py --steps 5000 --random-actions
python smoke_test.py --steps 1000 --render
```

零动作1000步对应10秒。测试会检查观测、奖励、物理状态、非足端触地、基座高度和姿态终止。

## 训练

```bash
python train.py
```

默认训练200万步，方向在 `[-pi, pi]` 内均匀采样，速度课程为：

| 训练步数 | 平移速度范围 |
|---:|---:|
| 0–300k | 0.03–0.07 m/s |
| 300k–800k | 0.04–0.12 m/s |
| 800k–1.4M | 0.05–0.18 m/s |
| 1.4M以后 | 0.06–0.25 m/s |

常用命令：

```bash
python train.py --device cuda --total-steps 2000000
python train.py --resume checkpoints/sac_hexapod_1000000.pt --total-steps 2000000
```

训练默认不渲染。新任务日志写入 `logs/episodes_pose_command.csv` 和 `logs/evaluation_pose_command.csv`，已有历史CSV保持不变；周期模型、最佳模型和最终模型保存在 `checkpoints/`。续训会恢复网络、温度和optimizer；Replay Buffer不写入checkpoint，恢复后先重新收集10000条transition才继续更新。

## 播放模型

```bash
python play.py --checkpoint checkpoints/sac_hexapod_best.pt --command-x 0.10 --command-y 0.00
python play.py --checkpoint checkpoints/sac_hexapod_best.pt --command-x 0.00 --command-y 0.00 --command-yaw-offset-deg 15
python play.py --checkpoint checkpoints/sac_hexapod_best.pt --command-x 0.00 --command-y 0.00 --command-pitch-deg 5
```

## 关键配置

- `EnvConfig`：模型路径、24关节映射、PD、动作缩放、Command采样、速度自适应步态频率、Yaw/Pitch平滑速度、运行时armature/damping、初始扰动和终止条件。
- `RewardConfig`：平移速度、Yaw/Pitch/Roll姿态、姿态角速度、静止六足支撑和步态奖励，以及停滞、滑脚、动作变化、关节加速度、扭矩和功率惩罚。
- `SACConfig`：95维观测、24维动作、网络宽度、学习率、熵目标、Replay Buffer和训练更新参数。
- `TrainConfig`：训练步数、平移速度课程、评估/保存频率和warm-up动作幅度。

后续加入姿态训练时只需要逐步扩大 `command_yaw_offset_range`、`command_pitch_range`，并提高 `turn_in_place_probability`、`pitch_in_place_probability`。如需每2～4秒自动更新Command，再将 `resample_commands_during_episode` 设为 `True`；网络观测维度仍保持95。

旧版18动作、75/77观测的checkpoint与本项目不兼容，必须重新训练。

## 文件说明

- `config.py`：全部可调参数。
- `env.py`：Gymnasium环境、模型内存加载、PD控制、观测、奖励、接触与终止。
- `sac.py`：双Q SAC、自动熵调节、经验回放和checkpoint。
- `train.py`：课程训练、评估、CSV日志、保存与续训。
- `play.py`：确定性策略可视化。
- `smoke_test.py`：站立、随机扰动和随机动作动态测试。
- `CHANGES.md`：逐项修改记录和验证结果。
