# Snake Tracked Robot SAC Obstacle Avoidance

基于 MuJoCo + Soft Actor-Critic（SAC）的蛇形履带机器人点目标避障训练程序。

该目录中的 SAC 网络结构与 `hexapod_control/sac.py` 保持一致，环境代码结构也按照 `hexapod_control/env.py` 的组织方式实现。履带采用 MuJoCo 3.11+ 的 `geom.surfacevel`，Python 只计算虚拟履带电机状态，实际地面接触与摩擦由 MuJoCo 求解。

## 1. 文件结构

```text
snake_control/
├── config.py                 # 环境、奖励、SAC、训练参数
├── env.py                    # SnakeAvoidEnv 环境
├── sac.py                    # SAC 网络、ReplayBuffer、checkpoint
├── train.py                  # SAC 训练入口
├── play.py                   # 加载 checkpoint 可视化测试
├── smoke_test.py             # 环境快速检查
├── requirements.txt          # Python 依赖
├── pd_control_slider.py      # surfacevel 履带手动控制/调试
├── pd_control_capsule.py     # 旧 capsule 版本控制程序
└── README.md
```

机器人模型使用：

```text
snake_description/mjcf/scene_slider.xml
snake_description/mjcf/snake_robot_tracks_slider.xml
```

## 2. 环境要求

推荐 Python 3.10+。

安装依赖：

```bash
cd ReinforcementLearning/snake_control
pip install -r requirements.txt
```

当前依赖包括：

```text
numpy >= 1.26
Gymnasium >= 1.0
MuJoCo >= 3.11
PyTorch >= 2.3
```

必须使用 MuJoCo 3.11 或更高版本，因为履带驱动使用 `geom_surfacevel`。

检查 MuJoCo 版本：

```bash
python -c "import mujoco; print(mujoco.__version__)"
```

检查 CUDA：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 3. 先运行 Smoke Test

正式训练前建议先运行：

```bash
python smoke_test.py
```

Smoke test 会检查：

- MJCF 是否能够正常加载；
- observation/action 维度是否正确；
- `surfacevel` 履带是否能够执行；
- LiDAR 是否输出有限值；
- 随机动作下 MuJoCo 是否出现 NaN/Inf；
- 环境能否正常 reset/step。

如果这里报错，应先解决环境或模型问题，再开始长时间训练。

## 4. 动作空间

SAC 输出 6 维连续动作，范围均为 `[-1, 1]`：

```text
action[0]  front_joint1 位置目标
action[1]  front_joint2 位置目标
action[2]  back_joint1  位置目标
action[3]  back_joint2  位置目标
action[4]  front_track   履带目标速度
action[5]  back_track    履带目标速度
```

前四维经过动作滤波后转换成关节 `q_des`，由 PD 控制器输出关节力矩。

默认关节参数：

```text
Kp = 20
Kd = 5
q_nominal = 0
joint_action_scale = 0.9 rad
```

后两维转换成虚拟履带角速度目标：

```python
track_target = track_speed_center + track_speed_scale * action
```

默认：

```text
track_speed_center = 4 rad/s
track_speed_scale  = 6 rad/s
```

因此：

```text
action = -1  -> -2 rad/s
action =  0  ->  4 rad/s
action = +1  -> 10 rad/s
```

履带内部采用：

```text
SAC target omega
      ↓
虚拟电机速度控制
      ↓
虚拟电机惯量 / 阻尼 / 负载力矩
      ↓
实际 track omega
      ↓
surfacevel = radius * omega
      ↓
MuJoCo Contact Solver
      ↓
地面摩擦与牵引力
```

## 5. 观测空间

当前 observation 共 510 维：

```text
480  LiDAR（4 × 120）
  3  projected gravity
  3  base angular velocity
  3  base linear velocity
  2  goal position in robot heading frame (x, y)
  1  goal distance
  2  heading error (sin, cos)
  4  joint position
  4  joint velocity
  2  virtual track omega
  6  previous filtered action
----
510
```

LiDAR 默认参数：

```text
Rows          = 4
Columns       = 120
Horizontal FOV = 120 deg
Vertical FOV   = 45 deg
Max range      = 30 m
Scan rate      = 10 Hz
```

物理仿真 timestep 为 1 ms，`frame_skip=20`，因此策略频率约为 50 Hz。LiDAR 为 10 Hz，所以多次策略 step 之间会保持最近一帧雷达结果。

训练环境会把 `floor` 放到 LiDAR 不扫描的 geom group，LiDAR 主要用于检测 `group=0` 的障碍物。

## 6. 目标点与障碍物

每个 episode 会重新随机目标点和障碍物位置。

默认目标距离：

```text
2.5 ~ 4.0 m
```

默认目标横向偏移：

```text
-0.60 ~ +0.60 m
```

当机器人进入目标点 `0.30 m` 半径内时判定成功。

环境会自动寻找名称以 `obstacle` 开头的 geom，例如：

```xml
<geom name="obstacle1" .../>
<geom name="obstacle2" .../>
<geom name="obstacle3" .../>
```

因此后续在 `scene_slider.xml` 中增加更多 `obstacle*` 障碍物后，不需要修改训练环境中的障碍物 ID 列表。

训练中带有障碍物位置 curriculum。随着训练步数增加，障碍物横向偏移逐渐减小，使障碍物更靠近起点到目标点的直接路径。

## 7. Reward

奖励主要由以下部分组成：

```text
+ 向目标前进的 progress
+ 朝向目标
+ 朝目标方向的速度

- 距离障碍物过近
- 停滞
- 动作变化过快
- 动作二阶变化
- 关节偏离中立姿态
- 关节速度
- 关节力矩
- 履带电机力矩
- 每步时间成本
```

终止奖励/惩罚：

```text
到达目标       + success_reward
撞击障碍物     - collision_penalty
翻倒/越界等    - termination_penalty
仿真出现 NaN   - invalid_physics_penalty
```

所有权重都集中在 `config.py -> RewardConfig` 中。

## 8. 开始训练

从头训练：

```bash
python train.py
```

显式使用 CUDA：

```bash
python train.py --device cuda
```

使用 CPU：

```bash
python train.py --device cpu
```

指定总训练步数：

```bash
python train.py --device cuda --total-steps 2000000
```

短时间测试训练流程：

```bash
python train.py --device cuda --total-steps 20000
```

训练时打开 MuJoCo Viewer：

```bash
python train.py --device cuda --render
```

注意：渲染会明显降低训练速度，正式训练建议不要使用 `--render`。

## 9. 从 Checkpoint 继续训练

`train.py` 已支持 `--resume`。

例如从 500000 step 的 checkpoint 继续训练到默认总步数：

```bash
python train.py --device cuda --resume checkpoints/sac_snake_500000.pt
```

继续训练并指定新的总步数：

```bash
python train.py --device cuda --resume checkpoints/sac_snake_500000.pt --total-steps 3000000
```

也可以从中断保存文件恢复：

```bash
python train.py --device cuda --resume checkpoints/sac_snake_interrupted.pt --total-steps 3000000
```

或者从 best checkpoint 继续：

```bash
python train.py --device cuda --resume checkpoints/sac_snake_best.pt --total-steps 3000000
```

当前 checkpoint 会恢复：

- Actor；
- Critic；
- Target Critic；
- Actor optimizer；
- Critic optimizer；
- alpha optimizer；
- SAC temperature `log_alpha`；
- checkpoint 中记录的 global step；
- best evaluation reward；
- best success rate。

注意：当前 ReplayBuffer 不写入 checkpoint。使用 `--resume` 后 ReplayBuffer 会重新从空状态收集数据，达到 `update_after` 条 transition 后才重新进行网络更新。这不会影响网络参数恢复，但恢复训练后的前一段时间主要用于重新填充经验池。

`--total-steps` 表示最终训练总步数，不是“再训练多少步”。例如 checkpoint 位于 500000 step：

```bash
python train.py --resume checkpoints/sac_snake_500000.pt --total-steps 2000000
```

实际会继续训练：

```text
500001 -> 2000000
```

## 10. Checkpoint

训练模型默认保存在：

```text
snake_control/checkpoints/
```

周期 checkpoint：

```text
sac_snake_100000.pt
sac_snake_200000.pt
...
```

最佳模型：

```text
sac_snake_best.pt
```

正常训练结束：

```text
sac_snake_final.pt
```

Ctrl+C 中断训练时：

```text
sac_snake_interrupted.pt
```

默认保存周期：

```text
checkpoint_every = 100000 steps
```

可通过命令行修改：

```bash
python train.py --checkpoint-every 50000
```

## 11. Evaluation

训练过程中默认每 50000 step 做一次评估。

可修改：

```bash
python train.py --eval-every 25000 --eval-episodes 10
```

评估指标包括：

```text
reward
success_rate
collision_rate
final_goal_distance
progress
path_length
mean_lidar_min
```

最佳模型优先按照 `success_rate` 判断；成功率相同时再比较 evaluation reward。

## 12. 日志

日志默认保存到：

```text
snake_control/logs/
```

Episode 日志：

```text
episodes_avoidance.csv
```

Evaluation 日志：

```text
evaluation_avoidance.csv
```

SAC 网络更新日志：

```text
updates.csv
```

## 13. 测试训练好的模型

测试 best checkpoint：

```bash
python play.py --checkpoint checkpoints/sac_snake_best.pt
```

测试指定 checkpoint：

```bash
python play.py --checkpoint checkpoints/sac_snake_1000000.pt
```

`play.py` 默认使用 deterministic action，即使用 Actor 输出分布的均值，而不是继续随机采样。

Viewer 中会显示目标位置和目标方向，便于观察机器人是否能够利用 LiDAR 绕开障碍物并到达目标。

## 14. 常用训练命令

```bash
# 安装
pip install -r requirements.txt

# 环境检查
python smoke_test.py

# 快速训练测试
python train.py --device cuda --total-steps 20000

# 正式训练
python train.py --device cuda --total-steps 2000000

# 从 checkpoint 继续训练到 3M step
python train.py --device cuda --resume checkpoints/sac_snake_1000000.pt --total-steps 3000000

# 可视化训练（较慢）
python train.py --device cuda --render

# 测试最佳模型
python play.py --checkpoint checkpoints/sac_snake_best.pt
```

## 15. 主要参数位置

环境与训练参数统一放在：

```text
config.py
```

主要配置类：

```text
RewardConfig   reward 权重
EnvConfig      MuJoCo、动作、LiDAR、目标、障碍物参数
SACConfig      SAC 网络及优化参数
TrainConfig    总步数、保存、评估、curriculum 参数
```

如需调训练行为，优先修改 `config.py`，避免把实验参数散落到 `env.py` 和 `train.py` 中。
