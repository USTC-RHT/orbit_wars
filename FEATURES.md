# RL Feature Definitions

`src/features.py` exposes three fixed input sizes:

```python
SELF_DIM = 18
CANDIDATE_DIM = 25
GLOBAL_DIM = 11
```

## self_features

One row per owned source planet.

```python
# 0  bias: 固定 1.0，给 MLP 一个常量偏置输入。
# 1  src_x: 来源星球 x 坐标 / board_size。
# 2  src_y: 来源星球 y 坐标 / board_size。
# 3  src_radius: 来源星球半径 / 5.0。
# 4  src_ships: 来源星球当前船数 / max_ships。
# 5  src_production: 来源星球产量 / max_production。
# 6  src_is_rotating: 来源星球是否在内圈旋转轨道。
# 7  src_dist_to_sun: 来源星球到太阳中心距离 / board_size。
# 8  my_planet_count: 我方星球数量 / max_planets。
# 9  enemy_planet_count: 敌方星球数量 / max_planets。
# 10 my_total_ships_on_planets: 我方星球上总船数归一化。
# 11 fleet_pressure: 敌方飞行中船数 - 我方飞行中船数，归一化。
# ---- inbound 特征（预测未来 60 步击中 source 星球的舰队）----
# 12 my_inbound_ships: 我方增援舰队总船数 / max_ships。
# 13 enemy_inbound_ships: 敌方来袭舰队总船数 / max_ships。
# 14 my_inbound_eta_min: 我方最早到达 ETA / horizon，无则为 1。
# 15 my_inbound_eta_mean: 我方到达 ETA 平均 / horizon，无则为 1。
# 16 enemy_inbound_eta_min: 敌方最早到达 ETA / horizon，无则为 1。
# 17 enemy_inbound_eta_mean: 敌方到达 ETA 平均 / horizon，无则为 1。
```

## candidate_features

`slot 0` is no-op; `slot 1..max_planets` are current planets sorted by id. Empty slots are zero padding.

```python
# 0  is_noop: 这个槽位是否是不发兵动作。
# 1  is_planet: 这个槽位是否是真实星球；padding 槽位为 0。
# 2  is_source: 这个槽位是否就是当前来源星球自己。
# 3  is_neutral: 目标是否中立。
# 4  is_friend: 目标是否我方其它星球，不包括 source 自己。
# 5  is_enemy: 目标是否敌方星球。
# 6  tgt_x: 目标 x 坐标 / board_size。
# 7  tgt_y: 目标 y 坐标 / board_size。
# 8  dx_to_src: 目标 x - 来源 x，再除以 board_size。
# 9  dy_to_src: 目标 y - 来源 y，再除以 board_size。
# 10 dist_to_src: 来源到目标距离 / board_size。
# 11 tgt_radius: 目标半径 / 5.0。
# 12 tgt_ships: 目标当前船数 / max_ships。
# 13 tgt_production: 目标产量 / max_production。
# 14 tgt_is_rotating: 目标是否是普通旋转星球。
# 15 tgt_is_comet: 目标是否是彗星。
# 16 target_ship_ratio: (目标船数 + 1) / 来源船数，用于估算攻占压力。
# 17 src_ships_again: 来源船数 / max_ships，让目标编码也知道本次发兵能力。
# ---- inbound 特征（预测未来 60 步到达该目标的舰队）----
# 18 my_inbound_ships: 我方舰队总船数 / max_ships。
# 19 enemy_inbound_ships: 敌方舰队总船数 / max_ships。
# 20 my_inbound_eta_min: 我方最早到达 ETA / horizon，无则为 1。
# 21 my_inbound_eta_mean: 我方到达 ETA 平均 / horizon，无则为 1。
# 22 enemy_inbound_eta_min: 敌方最早到达 ETA / horizon，无则为 1。
# 23 enemy_inbound_eta_mean: 敌方到达 ETA 平均 / horizon，无则为 1。
# 24 proto_score: Proto-V15 公式得分 / 100，给网络一个手工设计的目标价值参考信号。
```

`proto_score` uses:

```text
score = (100 - dist) + (15 * production) + (10 * enemy_bonus) - (0.7 * total_ships) - (2 * eta)
```

The model still only chooses target and ship bucket. `src/opponents.py` calls `src/physics.py` to compute the actual angle, ETA, and collision-valid route.

## global_features

Repeated for each source planet in the same turn.

```python
# 0  game_progress: 当前 step / episode_steps。
# 1  angular_velocity: 旋转星球角速度 / 0.05。角速度决定未来位置。
# 2  my_planet_count: 我方星球数量 / max_planets。
# 3  enemy_planet_count: 敌方星球数量 / max_planets。
# 4  neutral_planet_count: 中立星球数量 / max_planets。
# 5  my_ships_on_planets: 我方星球上总船数归一化。
# 6  enemy_ships_on_planets: 敌方星球上总船数归一化。
# 7  my_ships_in_fleets: 我方飞行中舰队总船数归一化。
# 8  enemy_ships_in_fleets: 敌方飞行中舰队总船数归一化。
# 9  my_total_production: 我方总产量归一化。
# 10 enemy_total_production: 敌方总产量归一化。
```

## inbound collision features

`precompute_inbound()` uses `INBOUND_HORIZON = 60` and assigns each fleet only to its first collision target.

```python
# 1. 舰队移动阶段：fleet[t-1] -> fleet[t] 线段撞 planet[t-1]。
# 2. 星球 sweep 阶段：planet[t-1] -> planet[t] 线段扫过 fleet[t]。
# 3. 同一 tick 内，舰队移动碰撞早于星球 sweep 碰撞。
```
