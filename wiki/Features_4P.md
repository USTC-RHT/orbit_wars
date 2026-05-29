# RL 四人局特征定义

本文档只描述 `jax_agent/features_jax.py` 的四人局 canonical 特征。旧版二人局特征仍保留在 `FEATURES.md`，两份文档不要混用。

当前四人局训练使用的维度：

```python
PLANET_DIM = 38
RELATIVE_DIM = 13
GLOBAL_DIM = 22
FLEET_DIM = 39
MAP_CHANNELS = 10
MAP_GRID_SIZE = 8
```

`encode_batch(state, params)` 返回：

```python
planet_features    # [B, P, 38]
relative_features  # [B, P, P + 1, 13]
global_features    # [B, P, 22]
map_features       # [B, P, 10, 8, 8]
candidate_mask     # [B, P, P + 1]
amount_mask        # [B, P, P + 1, ship_bucket_count]
source_ids         # [B, P]
source_ships       # [B, P]
candidate_ids      # [B, P, P + 1]
ship_buckets       # [B, P, ship_bucket_count]
source_mask        # [B, P]
fleet_features     # [B, P, max_fleet_tokens, 39]
fleet_mask         # [B, P, max_fleet_tokens]
```

## 基础约定

训练和推理都先把任意真实玩家视角 canonicalize 成 player 0 视角：

```python
# learner/self 永远是 owner == 0。
# 三个对手分别是 owner == 1 / 2 / 3。
# 中立星球是 owner == -1。
```

星球、目标和来源星球 owner 使用 5 类 one-hot：

```python
owner_class = [neutral, self, opp1, opp2, opp3]
```

舰队 owner 使用 4 类 one-hot：

```python
player_owner = [self, opp1, opp2, opp3]
```

`candidate slot 0` 是 no-op / padding 类目标位；`candidate slot i + 1` 对应 `planet slot i`。

## planet_features

每个 planet slot 都会构建一条星球 token，包含我方、中立、三名敌方以及 dead/padding 槽位。

```python
# 0     is_planet: 固定为 1.0。
# 1     is_alive: 星球是否存活。
# 2-6   owner_class one-hot: neutral / self / opp1 / opp2 / opp3。
# 7     x: 星球 x 坐标 / BOARD_SIZE。
# 8     y: 星球 y 坐标 / BOARD_SIZE。
# 9     radius: 星球半径 / 5.0。
# 10    ships: clip01(星球当前船数 / max_ships)。
# 11    production: 星球生产力 / max_production。
# 12    is_static_planet: 静止星球。
# 13    is_rotating_planet: 普通旋转星球，不含彗星。
# 14    is_comet_planet: 彗星星球。
# 15    dist_to_sun: 星球到太阳中心距离 / BOARD_SIZE。
# 16-20 nearest_owner_dist: 最近 neutral/self/opp1/opp2/opp3 星球距离 / BOARD_SIZE；没有则为 1。
# 21-25 near_owner_count_r25: R=25 内 neutral/self/opp1/opp2/opp3 星球数量 / max_planets。
# 26-29 inbound_ships: 预计飞向该星球的 self/opp1/opp2/opp3 船数 / max_ships。
# 30-33 inbound_eta_min: self/opp1/opp2/opp3 inbound 最短 ETA / INBOUND_HORIZON；没有则为 1。
# 34-37 inbound_eta_mean: self/opp1/opp2/opp3 inbound 平均 ETA / INBOUND_HORIZON；没有则为 1。
```

## relative_features

`relative_features[b, src, tgt, :]` 描述“从某个 source 看某个 target”的关系。

```python
# 0     is_source: target 是否就是当前 source 自己。
# 1-5   target owner_class one-hot: neutral / self / opp1 / opp2 / opp3。
# 6     dx: (target.x - source.x) / BOARD_SIZE。
# 7     dy: (target.y - source.y) / BOARD_SIZE。
# 8     dist: source 到 target 的距离 / BOARD_SIZE。
# 9     target_ship_ratio: clip01((target 船数 + 1) / max(source 船数, 1))。
# 10    source_ships: clip01(source 船数 / max_ships)，复制到每个 target 行。
# 11    proto_score: Proto-style 目标评分 / 100.0。
# 12    eta_max_bucket: 使用最大派兵档速度估算的 ETA / INBOUND_HORIZON。
```

目标合法性：

```python
target_valid =
    alive
    & (target != source)
    & (
        is_friend
        | ((is_enemy | is_neutral) & (source.ships > target.ships))
    )
```

其中 `is_enemy = owner > 0`，所以三个敌方玩家都会被视为可攻击敌人，但特征里仍保留具体 enemy owner。

## global_features

同一个 state 下，`global_features` 会复制到每个 source 行。

```python
# 0      game_progress: 当前 step / episode_steps。
# 1      angular_velocity: 星球角速度 / 0.05。
# 2-6    planet_count: neutral/self/opp1/opp2/opp3 存活星球数 / max_planets。
# 7-11   planet_ships: neutral/self/opp1/opp2/opp3 星球总船数，归一化。
# 12-16  production: neutral/self/opp1/opp2/opp3 总生产力，归一化。
# 17-21  fleet_ships: neutral/self/opp1/opp2/opp3 飞行舰队总船数，归一化；neutral 恒为 0。
```

## fleet_features

`fleet_features` 保留 top-K 飞行舰队作为 Transformer token。当前实现是 per-state top-K，然后复制到 source 维。

排序规则：

```text
sort_key = -ships + 0.01 * dist_to_center
```

特征定义：

```python
# 0-3   fleet owner one-hot: self / opp1 / opp2 / opp3。
# 4     fleet_x: 舰队 x 坐标 / BOARD_SIZE。
# 5     fleet_y: 舰队 y 坐标 / BOARD_SIZE。
# 6     cos_angle: 舰队飞行角度的 cos。
# 7     sin_angle: 舰队飞行角度的 sin。
# 8     fleet_ships: clip01(舰队船数 / max_ships)。
# 9     fleet_speed: 舰队速度 / 6.0。
# 10    launch_age: 估算已飞行时间 / episode_steps。
# 11    from_x: 来源星球 x 坐标 / BOARD_SIZE。
# 12    from_y: 来源星球 y 坐标 / BOARD_SIZE。
# 13-17 from owner_class one-hot: neutral / self / opp1 / opp2 / opp3。
# 18    from_ships: clip01(来源星球当前船数 / max_ships)。
# 19    dist_from_origin: 舰队离来源星球的距离 / BOARD_SIZE。
# 20-24 nearest_owner_dist: 到最近 neutral/self/opp1/opp2/opp3 星球距离 / BOARD_SIZE。
# 25    target_valid: 未来 INBOUND_HORIZON 步内是否预测会命中某个存活星球。
# 26-30 target owner_class one-hot: neutral / self / opp1 / opp2 / opp3。
# 31    target_x: 预测目标当前 x 坐标 / BOARD_SIZE；target_valid=0 时为 0。
# 32    target_y: 预测目标当前 y 坐标 / BOARD_SIZE；target_valid=0 时为 0。
# 33    target_ships: clip01(预测目标当前船数 / max_ships)；target_valid=0 时为 0。
# 34    target_production: 预测目标生产力 / max_production；target_valid=0 时为 0。
# 35    target_dx: (target.x - fleet.x) / BOARD_SIZE；target_valid=0 时为 0。
# 36    target_dy: (target.y - fleet.y) / BOARD_SIZE；target_valid=0 时为 0。
# 37    target_dist: 舰队到预测目标当前距离 / BOARD_SIZE；target_valid=0 时为 0。
# 38    target_eta: 预测命中时间 / INBOUND_HORIZON；target_valid=0 时为 0。
```

`fleet_mask` 标记哪些 fleet token 有效。无效 token 的特征全为 0。

## map_features

`map_features[b, src]` 是 8x8 的 source 视角小地图，主要给 autoregressive actor 的 source 选择 CNN 分支使用。

```python
# channel 0 source_planet: 当前 source 星球所在格子。
# channel 1 self_planets: 我方非 source 星球。
# channel 2 opp1_planets: 对手 1 星球。
# channel 3 opp2_planets: 对手 2 星球。
# channel 4 opp3_planets: 对手 3 星球。
# channel 5 neutral_planets: 中立星球。
# channel 6 self_fleets: 我方飞行舰队船数强度。
# channel 7 opp1_fleets: 对手 1 飞行舰队船数强度。
# channel 8 opp2_fleets: 对手 2 飞行舰队船数强度。
# channel 9 opp3_fleets: 对手 3 飞行舰队船数强度。
```

星球通道写入 `1.0`。飞船强度通道按格子累加 `clip01(ships / max_ships)`，并 clip 到 `[0, 1]`。

## Normalization

归一化只处理连续数值列，owner one-hot、存活、类型等离散语义保持原值。

```python
# planet normalized: 7-11, 15-37
# relative normalized: 6-12
# global normalized: 0-21
```

旧 checkpoint 和旧 normalizer 的特征维度与当前四人局特征不兼容，需要重新训练或重新初始化对应统计量。

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

## ship_buckets

默认派兵比例档位：

```python
SHIP_RATIOS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
```

低于 `MIN_LAUNCH = 10` 的 bucket 会被视为 0/非法。实际 bucket 数量由 `ship_bucket_count` 控制。
