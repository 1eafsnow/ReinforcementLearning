# Snake Tracked Robot SAC Obstacle Traversal

基于 MuJoCo + Soft Actor-Critic（SAC）的蛇形履带机器人越障训练程序。当前任务不是绕开障碍物，而是让机器人从障碍物上方爬过去，并到达障碍物后的目标点。

SAC 网络结构与 `hexapod_control/sac.py` 保持一致，环境结构沿用 hexapod 的组织方式。履带使用 MuJoCo 3.11+ 的 `geom.surfacevel`。

## 1. 文件结构

```text
snake_control/
├── config.py
├── env.py
├── sac.py
├── train.py
├── play.py
├── smoke_test.py
├── requirements.txt
└── README.md
```

机器人模型：

```text
snake_description/mjcf/scene_slider.xml
snake_description/mjcf/snake_robot_tracks_slider.xml
```

## 2. 障碍物不再写在 scene_slider.xml 中

`scene_slider.xml` 现在只保留基础场景、地面和机器人，不包含训练障碍物。

训练环境启动时，`env.py` 会读取 `scene_slider.xml` 文本，在 Python 中动态加入：

```text
training_obstacle_hfield
training_obstacle
```

然后通过临时 MJCF 编译 MuJoCo 模型。临时文件在模型加载完成后立即删除，不会写回仓库。

之后每次 `reset()` 都会重新修改 heightfield 数据，因此每个 episode 都会得到新的梯形障碍物。

## 3. Python 随机梯形障碍

障碍横截面为：

```text
             ┌────────────┐
            /              \
───────────/                \───────────
          上坡      平台       下坡
```

当前每个 episode 随机：

```text
障碍高度          obstacle_min_height ~ 当前 curriculum 上限
障碍起点距离      1.05 ~ 1.35 m
单侧坡长          0.35 ~ 0.55 m
顶部平台长度      0.25 ~ 0.45 m
目标点距障碍末端  1.00 ~ 1.35 m
目标横向偏移      -0.08 ~ +0.08 m
```

heightfield 在 Y 方向横跨训练区域，因此机器人不能简单从障碍物侧面绕过去，必须学习越障。

如需在测试时固定障碍高度：

```bash
python play.py --checkpoint checkpoints/sac_snake_traverse_best.pt --fixed-obstacle-height 0.08
```

## 4. 履带与障碍接触

地面使用碰撞 bit `1`。

履带 surfacevel pad 使用碰撞 bit `2`。

Python 动态生成的 heightfield 障碍使用 `contype=3`、`conaffinity=3`，也就是同时包含 bit `1` 和 bit `2`。

因此：

```text
履带 pad <-> floor
    仍然使用 snake_robot_tracks_slider.xml 中原来的显式 <pair>

履带 pad <-> training obstacle
    自动产生接触，并保留 surfacevel 牵引

结构碰撞体 <-> training obstacle
    正常产生物理碰撞
```

这样机器人爬坡时不是只有外壳撞上障碍，而是履带 pad 可以真正对坡面产生驱动力。

## 5. 动作空间

SAC 输出 6 维连续动作，范围均为 `[-1, 1]`：

```text
action[0]  front_joint1 q_des
action[1]  front_joint2 q_des
action[2]  back_joint1  q_des
action[3]  back_joint2  q_des
action[4]  front_track   target omega
action[5]  back_track    target omega
```

前 4 维由 PD 控制关节，后 2 维控制虚拟履带电机目标速度。

履带控制链：

```text
SAC target omega
      ↓
virtual track motor
      ↓
track omega
      ↓
surfacevel = radius * omega
      ↓
MuJoCo contact solver
```

## 6. 观测空间

保持 510 维：

```text
480  LiDAR (4 x 120)
  3  projected gravity
  3  base angular velocity
  3  base linear velocity
  2  goal local x/y
  1  goal distance
  2  heading error sin/cos
  4  joint position
  4  joint velocity
  2  track omega
  6  previous filtered action
----
510
```

LiDAR：

```text
Horizontal FOV = 120 deg
Vertical FOV   = 45 deg
Rows x Cols    = 4 x 120
Max range      = 30 m
Scan rate      = 10 Hz
```

物理 timestep 为 1 ms，`frame_skip=20`，策略频率约 50 Hz。

## 7. 越障阶段奖励

环境会记录机器人越障进度：

```text
front_track_stage   前履带到达障碍顶部区域
base_stage          中央机体到达障碍顶部区域
back_track_stage    后履带到达障碍顶部区域
obstacle_cleared    后履带完全越过障碍
```

Reward 主要包括：

```text
+ 沿训练路线向前 progress
+ 朝向最终目标
+ 向前速度
+ front_track_stage 奖励
+ base_stage 奖励
+ back_track_stage 奖励
+ obstacle_cleared 奖励
+ 最终到达目标奖励

- 长时间停滞
- 后退
- 横向偏离训练路线
- 动作变化过快
- 动作二阶变化
- 关节姿态/速度
- 关节和履带扭矩
- 时间成本
```

成功条件：

```text
obstacle_cleared == True
并且
进入目标半径
```

因此机器人不能只接近目标而不完整越过障碍。

## 8. 障碍高度 Curriculum

默认：

```text
0 step        max height = 0.040 m
250k          max height = 0.055 m
600k          max height = 0.075 m
1.00M         max height = 0.095 m
1.50M         max height = 0.120 m
```

每个 episode 会在：

```text
obstacle_min_height ~ curriculum 当前 max height
```

之间随机采样。

配置位置：

```text
config.py -> TrainConfig.obstacle_height_curriculum
```

## 9. 安装

```bash
cd ReinforcementLearning/snake_control
pip install -r requirements.txt
```

MuJoCo 必须为 3.11+。

## 10. Smoke Test

```bash
python smoke_test.py
```

会检查：

```text
动态 heightfield 障碍生成
observation/action 维度
LiDAR
surfacevel
随机动作 step
NaN/Inf
reset
```

## 11. 从头训练

```bash
python train.py --device cuda --total-steps 2000000
```

快速测试训练流程：

```bash
python train.py --device cuda --total-steps 20000
```

打开 Viewer：

```bash
python train.py --device cuda --render
```

正式训练建议关闭渲染。

## 12. Checkpoint 与继续训练

越障模型默认命名：

```text
checkpoints/sac_snake_traverse_100000.pt
checkpoints/sac_snake_traverse_best.pt
checkpoints/sac_snake_traverse_final.pt
checkpoints/sac_snake_traverse_interrupted.pt
```

继续训练：

```bash
python train.py --device cuda --resume checkpoints/sac_snake_traverse_500000.pt --total-steps 2000000
```

`--total-steps` 是最终 global step，不是额外训练步数。

旧避障 checkpoint 的网络输入输出维度仍然相同，所以技术上可以加载，但任务、奖励和环境已经改变，越障任务建议优先从头训练。

## 13. 测试模型

随机障碍：

```bash
python play.py --checkpoint checkpoints/sac_snake_traverse_best.pt
```

限制测试最大障碍高度：

```bash
python play.py --checkpoint checkpoints/sac_snake_traverse_best.pt --max-obstacle-height 0.10
```

固定为 8 cm：

```bash
python play.py --checkpoint checkpoints/sac_snake_traverse_best.pt --fixed-obstacle-height 0.08
```

## 14. 日志

```text
logs/episodes_traversal.csv
logs/evaluation_traversal.csv
logs/updates.csv
```

Episode 重点字段：

```text
obstacle_height
obstacle_ramp_length
obstacle_platform_length
front_track_stage
base_stage
back_track_stage
obstacle_cleared
success
reward
```

Evaluation：

```text
success_rate
obstacle_clear_rate
mean_obstacle_height
final_goal_distance
reward
```

## 15. 常用命令

```bash
python smoke_test.py
python train.py --device cuda --total-steps 2000000
python train.py --device cuda --resume checkpoints/sac_snake_traverse_500000.pt --total-steps 2000000
python play.py --checkpoint checkpoints/sac_snake_traverse_best.pt
```
