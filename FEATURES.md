# RL 特征定义

`jax_agent/features_jax.py` 当前训练使用的特征维度：

```python
PLANET_DIM = 26
RELATIVE_DIM = 8
GLOBAL_DIM = 11
FLEET_DIM = 28
MAP_CHANNELS = 6
MAP_GRID_SIZE = 8
```

`encode_batch(state, params)` 返回一个 `EncodedBatch`：

```python
planet_features    # [B, P, 26]
relative_features  # [B, P, P + 1, 8]
global_features    # [B, P, 11]
map_features       # [B, P, 6, 8, 8]
candidate_mask     # [B, P, P + 1]
amount_mask        # [B, P, P + 1, ship_bucket_count]
source_ids         # [B, P]
source_ships       # [B, P]
candidate_ids      # [B, P, P + 1]
ship_buckets       # [B, P, ship_bucket_count]
source_mask        # [B, P]
fleet_features     # [B, P, max_fleet_tokens, 28]
fleet_mask         # [B, P, max_fleet_tokens]
```

基础约定：

```python
# 训练和模型内部统一从 learner 视角看局面，learner 永远是 owner == 0。
# source_mask 是静态 source 所属权 mask：
#   source_mask = planet_alive & (owner == 0)
#
# candidate slot 0 是 no-op / padding 类目标位。
# candidate slot i + 1 对应 planet slot i。
# autoregressive target head 实际选择 target 时会把 slot 0 关掉；
# 停止生成命令由 end head 负责，不靠 target no-op。
#
# relative_features 只保存 source-target 相对关系；target 的绝对属性在 planet_features 里。
```

## planet_features

`planet_features` 是每个 planet slot 的统一星球特征，会为所有星球槽位构建，
包括我方、中立、敌方和 padding/dead 星球。当前 autoregressive Transformer 的星球
token 直接来自这组特征；`relative_features` 只用于 target/amount 的 source-target 相对关系。

```python
# 0  is_planet: 固定为 1.0，占位标记。
# 1  is_alive: 星球是否存活。
# 2  is_neutral: 星球是否为中立。
# 3  is_mine: 星球是否属于我方 learner。
# 4  is_enemy: 星球是否属于敌方。
# 5  x: 星球 x 坐标 / BOARD_SIZE。
# 6  y: 星球 y 坐标 / BOARD_SIZE。
# 7  radius: 星球半径 / 5.0。
# 8  ships: clip01(星球当前船数 / max_ships)。
# 9  production: 星球生产力 / max_production。
# 10 is_static_planet: 静止星球。
# 11 is_rotating_planet: 普通旋转星球，不含彗星。
# 12 is_comet_planet: 彗星星球。
# 13 dist_to_sun: 星球到太阳中心距离 / BOARD_SIZE。
#
# 局部拓扑特征，所有星球都会计算，包括我方/敌方/中立星球：
# 14 nearest_my_planet_dist: 最近我方星球距离 / BOARD_SIZE；没有则为 1。
# 15 nearest_enemy_planet_dist: 最近敌方星球距离 / BOARD_SIZE；没有则为 1。
# 16 nearest_neutral_planet_dist: 最近中立星球距离 / BOARD_SIZE；没有则为 1。
# 17 near_my_planet_count_r25: R=25 内我方星球数量 / max_planets。
# 18 near_enemy_planet_count_r25: R=25 内敌方星球数量 / max_planets。
# 19 near_neutral_planet_count_r25: R=25 内中立星球数量 / max_planets。
#
# 飞向该星球的 inbound 特征：
# 20 inbound_my_ships: 预计飞向该星球的我方船数 / max_ships。
# 21 inbound_enemy_ships: 预计飞向该星球的敌方船数 / max_ships。
# 22 inbound_my_eta_min: 我方 inbound 最短 ETA / INBOUND_HORIZON；没有则为 1。
# 23 inbound_my_eta_mean: 我方 inbound 平均 ETA / INBOUND_HORIZON；没有则为 1。
# 24 inbound_enemy_eta_min: 敌方 inbound 最短 ETA / INBOUND_HORIZON；没有则为 1。
# 25 inbound_enemy_eta_mean: 敌方 inbound 平均 ETA / INBOUND_HORIZON；没有则为 1。
```

## relative_features

`relative_features[b, src, tgt, :]` 描述“从某个 source 看某个 target”的关系。
`tgt=0` 是 no-op，`tgt=i+1` 对应 planet slot `i`。

```python
# 0  is_source: target 是否就是当前 source 自己。
# 1  dx: (target.x - source.x) / BOARD_SIZE。
# 2  dy: (target.y - source.y) / BOARD_SIZE。
# 3  dist: source 到 target 的距离 / BOARD_SIZE。
# 4  target_ship_ratio: clip01((target 船数 + 1) / max(source 船数, 1))。
# 5  source_ships: clip01(source 船数 / max_ships)，复制到每个 target 行。
# 6  proto_score: Proto-style 目标评分 / 100.0。
# 7  eta_max_bucket: 使用最大派兵档速度估算的 ETA / INBOUND_HORIZON。
```

`proto_score` 只是软提示，不是硬规则。actor 最终仍然通过 policy logits 选择 source、target 和 amount。

## global_features

同一个 state 下，`global_features` 会复制到每个 source 行。

```python
# 0  game_progress: 当前 step / episode_steps。
# 1  angular_velocity: 星球角速度 / 0.05。
# 2  my_planet_count: 我方存活星球数 / max_planets。
# 3  enemy_planet_count: 敌方存活星球数 / max_planets。
# 4  neutral_planet_count: 中立存活星球数 / max_planets。
# 5  my_ships_on_planets: 我方星球总船数，归一化。
# 6  enemy_ships_on_planets: 敌方星球总船数，归一化。
# 7  my_ships_in_fleets: 我方飞行舰队总船数，归一化。
# 8  enemy_ships_in_fleets: 敌方飞行舰队总船数，归一化。
# 9  my_total_production: 我方总生产力 / (max_planets * max_production)。
# 10 enemy_total_production: 敌方总生产力 / (max_planets * max_production)。
```

## fleet_features

`fleet_features` 保留 top-K 飞行舰队作为 Transformer token。
当前实现是 per-state top-K，然后复制到 source 维；不是每个 source 单独筛一遍。

舰队排序规则：

```text
sort_key = -ships + 0.01 * dist_to_center
```

也就是优先保留大舰队。

```python
# 0  is_my_fleet: 舰队是否属于我方。
# 1  fleet_x: 舰队 x 坐标 / BOARD_SIZE。
# 2  fleet_y: 舰队 y 坐标 / BOARD_SIZE。
# 3  cos_angle: 舰队飞行角度的 cos。
# 4  sin_angle: 舰队飞行角度的 sin。
# 5  fleet_ships: clip01(舰队船数 / max_ships)。
# 6  fleet_speed: 舰队速度 / 6.0。
# 7  launch_age: 估算已飞行时间 / episode_steps。
# 8  from_x: 来源星球 x 坐标 / BOARD_SIZE。
# 9  from_y: 来源星球 y 坐标 / BOARD_SIZE。
# 10 from_is_mine: 来源星球当前是否属于我方。
# 11 from_ships: clip01(来源星球当前船数 / max_ships)。
# 12 dist_from_origin: 舰队离来源星球的距离 / BOARD_SIZE。
# 13 nearest_enemy_planet_dist: 到最近敌方星球距离 / BOARD_SIZE。
# 14 nearest_neutral_planet_dist: 到最近中立星球距离 / BOARD_SIZE。
# 15 nearest_my_planet_dist: 到最近我方星球距离 / BOARD_SIZE。
# 16 target_valid: 未来 INBOUND_HORIZON 步内是否预测会命中某个存活星球。
# 17 target_is_mine: 预测目标当前是否属于我方；target_valid=0 时为 0。
# 18 target_is_enemy: 预测目标当前是否属于敌方；target_valid=0 时为 0。
# 19 target_is_neutral: 预测目标当前是否为中立；target_valid=0 时为 0。
# 20 target_x: 预测目标当前 x 坐标 / BOARD_SIZE；target_valid=0 时为 0。
# 21 target_y: 预测目标当前 y 坐标 / BOARD_SIZE；target_valid=0 时为 0。
# 22 target_ships: clip01(预测目标当前船数 / max_ships)；target_valid=0 时为 0。
# 23 target_production: 预测目标生产力 / max_production；target_valid=0 时为 0。
# 24 target_dx: (target.x - fleet.x) / BOARD_SIZE；target_valid=0 时为 0。
# 25 target_dy: (target.y - fleet.y) / BOARD_SIZE；target_valid=0 时为 0。
# 26 target_dist: 舰队到预测目标当前距离 / BOARD_SIZE；target_valid=0 时为 0。
# 27 target_eta: 预测命中时间 / INBOUND_HORIZON；target_valid=0 时为 0。
```

`fleet_mask` 标记哪些 fleet token 有效。无效 token 的特征全为 0。

`target_valid` 使用与 inbound 统计相同的轨迹预测：沿舰队当前方向向前模拟
`INBOUND_HORIZON=60` 步，取最先碰撞的存活星球。若 60 步内没有有效命中，
目标相关维度全部置 0。

## map_features

`map_features[b, src]` 是 8x8 的 source 视角小地图，当前主要给 autoregressive actor
的 source 选择 CNN 分支使用。

```python
# channel 0 source_planet: 当前 source 星球所在格子。
# channel 1 my_planets: 我方非 source 星球。
# channel 2 enemy_planets: 敌方星球。
# channel 3 neutral_planets: 中立星球。
# channel 4 my_ships: 我方飞行舰队船数强度，累加 clip01(ships / max_ships)。
# channel 5 enemy_ships: 敌方飞行舰队船数强度，累加 clip01(ships / max_ships)。
```

星球通道写入 `1.0`。飞船强度通道按格子累加归一化船数，并 clip 到 `[0, 1]`。

## Masks

```python
# source_mask[src]:
#   我方存活星球。
#
# src_mask[src]:
#   哪些源星球可以被选。
#   src_mask = source_mask & (remaining_ships[src] >= MIN_LAUNCH)
#   remaining_ships 是动态预算；同一 env step 内前面几个 token
#   已派出去的船会扣掉，避免超发。
#
# tgt_mask_t[tgt]:
#   选定 src 后，哪些目标可以被攻击/增援。
#   tgt_mask_t = target_alive & (target != src)
#   autoregressive target head 会把 slot 0 关掉；是否停止由 end head 决定。
#
# amt_mask_t[bucket]:
#   选定 src + tgt 后，哪些兵力档位合法。
#   amt_mask_t = (ship_buckets[bucket] >= MIN_LAUNCH)
#              & (ship_buckets[bucket] <= remaining_ships[src])
#
# MIN_LAUNCH = 10
# 下界：必须至少派 MIN_LAUNCH 艘。
# 上界：不能超过当前 source 的动态预算。
```

实现上，`candidate_mask[src, tgt]` 是 `tgt_mask_t` 的静态编码版本；
`amount_mask[src, tgt, bucket]` 提供静态下界，decoder 内再叠加
`remaining_ships[src]` 得到最终 `amt_mask_t`。

## ship_buckets

默认派兵比例档位：

```python
SHIP_RATIOS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
```

低于 `MIN_LAUNCH = 10` 的 bucket 会被视为 0/非法。实际 bucket 数量由 `ship_bucket_count` 控制。

## inbound collision features

`precompute_inbound()` 使用 `INBOUND_HORIZON = 60`，并把每支舰队分配给预测中第一个会碰撞的目标。

```python
# 1. 舰队移动碰撞：
#    fleet[t-1] -> fleet[t] 线段检测 planet[t-1]。
#
# 2. 星球 sweep 碰撞：
#    planet[t-1] -> planet[t] 线段检测 fleet[t]。
#
# 3. 同一个 tick 内，先检查舰队移动碰撞，再检查星球 sweep 碰撞。
```

## 模型使用方式

当前 autoregressive actor 大致这样使用特征：

```python
# State encoder:
#   planet tokens = planet_features，前面补一个 no-op token 后进入 Transformer
#   fleet tokens  = fleet_features[:, 0]
#   global ctx    = MLP(global_features[:, 0])
#
# Source selection:
#   使用 Transformer planet keys + source CNN map_features + source_mask。
#   实际 src_logits 还会叠加 remaining_ships >= MIN_LAUNCH 的动态约束。
#
# Target selection:
#   使用已选 source、planet tokens、relative_features、candidate_mask。
#
# Amount selection:
#   使用已选 source、已选 target、amount_mask，以及动态剩余船数 mask。
```

重要易错点：

```python
# source_mask 只表示“我方存活星球”，真正的 source head 还会检查当前剩余兵力。
# target 绝对属性永远看 planet_features / planet_out，不再从 relative_features 里取。
```
