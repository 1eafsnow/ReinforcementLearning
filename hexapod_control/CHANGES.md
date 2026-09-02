# 修改清单

## 静止支撑与速度自适应步态

1. 静止Command继续与移动Command分支独立计算，零速度时不产生进度奖励或停滞惩罚。
2. 新增 `standing_contact` 奖励，只有静止Command成立且六只脚均与地面接触时取1；权重由 `standing_contact_weight` 配置。
3. 足端接触只统计足端碰撞体与地面的接触，避免把足端碰到机器人自身或其他几何体误判为着地。
4. 完全静止时 `gait_gate=0` 且冻结步态相位；移动Command下相位始终推进，原地Yaw/Pitch调整时由姿态运动门控推进。
5. 固定 `gait_period` 改为 `gait_frequency_range` 和 `gait_frequency_speed_range`，使用目标Command速度线性插值完整Tripod周期频率。
6. 默认在 `0.03～0.25m/s` 内将完整周期频率从 `1.10Hz` 提高到 `2.20Hz`，Tripod支撑组每半周期切换一次。
7. 观测仍为95维、动作仍为24维；checkpoint结构兼容，版本标记更新为5。

## Command语义

1. `command_x`、`command_y`改为机器人实时Yaw坐标系中的纵向、横向目标速度，不再使用reset时固定坐标系。
2. 世界坐标速度先读取为全局速度，再只按当前Yaw旋转到水平朝向坐标系；Roll和Pitch不会改变水平移动方向或混合Z轴速度。
3. `command_yaw`改为世界坐标系目标Yaw角，`command_pitch`改为相对水平面的目标Pitch角，二者不再作为角速度使用。
4. 新增 `requested_command_yaw/pitch` 和Yaw/Pitch平滑更新，可通过 `set_command()` 在运行中改变目标。
5. 新增速度大小与方向角采样，默认80%任意方向移动、20%零速度静止站立。
6. 预留原地转向、原地Pitch调整以及每2～4秒自动重采样配置，第一阶段默认关闭姿态变化和轮内重采样。

## 观测

1. 观测维度保持95，动作维度保持24。
2. Command观测固定为归一化的 `command_x`、`command_y`、`command_pitch`。
3. Yaw目标通过 `sin(yaw_error)`、`cos(yaw_error)`表达，避免角度在正负pi处跳变。
4. 当前Pitch和Roll仍由完整机身坐标系中的投影重力表达，后续扩大姿态Command范围时无需改变网络shape。

## Reward与终止

1. 平移速度奖励改为二维速度误差，允许前后、左右及斜向移动，不再惩罚合法横向速度。
2. 零平移速度时关闭进度奖励和停滞惩罚，改用更严格的零速度跟踪奖励。
3. 新增Yaw、Pitch、Roll目标姿态奖励以及由姿态误差生成目标角速度的跟踪奖励。
4. 新增静止站立奖励，同时利用动作变化、功率和关节运动惩罚抑制原地抖动。
5. 平移或姿态调整时允许Tripod和抬腿奖励；完全静止时关闭步态门控，防止原地抬腿刷奖励。
6. 删除固定直立奖励、固定Roll/Pitch角速度惩罚和Yaw误差终止，避免阻碍合法Pitch和大角度Yaw目标。
7. 行进距离改为沿实时Yaw坐标系瞬时Command方向逐步累计，不再投影到reset时固定世界方向。

## 工程与工具

1. 训练环境默认加载带地面的 `mjcf/scene.xml`；带include的场景使用 `MjModel.from_xml_path()` 保留源文件目录，使 `hexapod_model.xml` 和模型meshdir的相对路径均能正确解析。
2. `train.py`课程改为速度大小课程，移动方向在 `[-pi, pi]` 内采样，并增加Pitch误差和完整Command日志。
3. 新日志写入 `episodes_pose_command.csv` 和 `evaluation_pose_command.csv`，保留已有历史CSV不变。
4. `play.py`增加 `command-x/y`、Yaw偏移和Pitch目标参数，可直接测试移动、原地转向或静止站立。
5. `smoke_test.py`显式使用零速度站立模式，并检查最终95维观测。
6. README中的物理步长修正为XML实际值 `0.001s`，`frame_skip=10`，策略周期仍为 `0.01s`。

## Checkpoint兼容性

- 网络维度仍为95维观测和24维动作，当前95/24 checkpoint可以加载。
- 新保存的checkpoint版本号更新为4，用于区分本次Command和Reward语义。
- Command第三维由旧的零Yaw角速度占位改为Pitch目标；第一阶段Pitch仍为0，因此输入shape和默认数值不变。
- Reward和速度坐标系语义已经改变，建议将旧checkpoint用于微调并重新观察训练曲线，不建议把旧结果直接视为新任务的收敛结果。

## 验证结果

- Python语法、AST解析以及95维观测/24维动作配置一致性检查通过。
- MuJoCo 3.11零动作无扰动站立1000步通过，最终基座高度约 `0.1316m`。
- 随机初始位置、姿态和任意世界Yaw站立1000步通过，最终Yaw误差约 `-0.041deg`。
- 随机动作5000步通过，观测与奖励始终有限，未出现异常终止。
- 定向测试确认Yaw为90度时，世界X正向速度转换为实时Yaw坐标系Y负向速度；增加10度Pitch后转换结果和Z轴速度保持不变。
- 本次修改使用MuJoCo 3.3.7重新完成无扰动站立1000步、随机初始状态站立1000步和随机动作5000步测试。
- 静止专项测试确认六足接触数为6、`standing_contact=1`、`gait_gate=0`且相位保持不变。
- 频率专项测试确认目标速度0.03和0.25m/s时完整周期频率分别为1.10和2.20Hz，相位累计值与理论值一致。
