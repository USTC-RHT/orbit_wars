# JAX Agent 特征定义

本文档对应当前 `jax_agent/features_jax.py`、`jax_agent/model_jax.py`、
`jax_agent/normalization_jax.py` 和主训练配置。

当前主训练配置（`cfg_jax_fixed_pool_4090.yaml`、`cfg_jax_RL_1.yaml`、
`cfg_jax_RL_2.yaml`、`cfg_jax_RL_3.yaml`）的关键开关：

```yaml
env.max_planets: 45
env.target_filter_enabled: true
env.attack_top_k: 10
env.defense_top_k: 5
model.use_cnn: true
model.num_transformer_layers: 2
model.num_heads: 4
model.max_fleet_tokens: 40
model.fleet_selection: impact
model.max_action_tokens: 12
state_norm.enabled: true
```

注意：`features_jax.py` 的默认 `MAX_FLEET_TOKENS = 40`。旧的
`cfg_jax_fixed_pool_0525.yaml` 使用 `max_fleet_tokens: 16`，且没有显式设置
`fleet_selection`，因此会走默认 `size` 选择。

## 维度和返回值

`jax_agent/features_jax.py` 中的固定特征维度：

```python
PLANET_DIM = 28
RELATIVE_DIM = 22
GLOBAL_DIM = 11
FLEET_DIM = 28
MAP_CHANNELS = 6
MAP_GRID_SIZE = 8
INBOUND_HORIZON = 60.0
MIN_LAUNCH = 10.0
LOCAL_PLANET_RADIUS = 25.0
```

`encode_batch(state, params)` 返回 `EncodedBatch`：

```python
planet_features    # [B, P, 28]
relative_features  # [B, P, P + 1, 22]
global_features    # [B, P, 11]
map_features       # [B, P, 6, 8, 8]
candidate_mask     # [B, P, P + 1]
source_ids         # [B, P]
source_ships       # [B, P]
candidate_ids      # [B, P, P + 1]
source_mask        # [B, P]
owned_mask         # [B, P]
fleet_features     # [B, P, max_fleet_tokens, 28]
fleet_mask         # [B, P, max_fleet_tokens]
```

基础索引约定：

```python
# 训练和模型内部统一从 learner 视角看局面，learner 永远是 owner == 0。
# 如果 learner 实际是 player 1，train_jax.flip_state() 会先做 owner/坐标/角度/星球槽位镜像。
#
# source slot 直接对应 planet slot，动作 head 中的 source 索引会额外前置一个 no-op：
#   src_action = 0      表示无效/占位，不允许真正选择。
#   src_action = i + 1  对应 planet slot i。
#
# target slot 也前置 no-op：
#   target slot 0      是 no-op / padding 类目标位。
#   target slot i + 1  对应 planet slot i。
#
# autoregressive target head 实际选择 target 时会把 slot 0 关掉。
# 停止继续生成命令由 end head 负责，不靠 target no-op。
```

## planet_features

`planet_features[b, i]` 是 planet slot `i` 的星球 token 特征。它会为所有星球槽位构建，
包括我方、中立、敌方、dead/padding 星球。模型编码时会在最前面补一个 no-op planet token，
并用 `is_alive` mask 掉 dead/padding planet token。

```python
# 0  is_planet: 固定为 1.0，占位标记。
# 1  is_alive: 星球是否存活。
# 2  is_neutral: 星球是否为中立 owner == -1。
# 3  is_mine: 星球是否属于 learner owner == 0。
# 4  is_enemy: 星球是否属于敌方 owner == 1。
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
# 局部拓扑特征，排除自己，只统计存活星球：
# 14 nearest_my_planet_dist: 最近我方星球距离 / BOARD_SIZE；没有则为 1。
# 15 nearest_enemy_planet_dist: 最近敌方星球距离 / BOARD_SIZE；没有则为 1。
# 16 nearest_neutral_planet_dist: 最近中立星球距离 / BOARD_SIZE；没有则为 1。
# 17 near_my_planet_count_r25: R=25 内我方星球数量 / max_planets。
# 18 near_enemy_planet_count_r25: R=25 内敌方星球数量 / max_planets。
# 19 near_neutral_planet_count_r25: R=25 内中立星球数量 / max_planets。
#
# 飞向该星球的短期 inbound 特征，来自 garrison_jax 的 18 步到达表：
# 20 inbound_my_ships: 预计飞向该星球的我方船数 / max_ships。
# 21 inbound_enemy_ships: 预计飞向该星球的敌方船数 / max_ships。
# 22 inbound_my_eta_min: 我方 inbound 最短 ETA / DEFAULT_GARRISON_HORIZON；没有则为 1。
# 23 inbound_my_eta_mean: 我方 inbound 船数加权平均 ETA / DEFAULT_GARRISON_HORIZON；没有则为 1。
# 24 inbound_enemy_eta_min: 敌方 inbound 最短 ETA / DEFAULT_GARRISON_HORIZON；没有则为 1。
# 25 inbound_enemy_eta_mean: 敌方 inbound 船数加权平均 ETA / DEFAULT_GARRISON_HORIZON；没有则为 1。
# 26 safe_drain_ratio: producer-style safe_drain / max(current ships, 1)。
# 27 full_send_risk_ratio: max(current ships - safe_drain, 0) / max(current ships, 1)。
```

旋转判定来自 `ROTATION_RADIUS_LIMIT`：`distance(initial_planet, center) + radius < limit`。
彗星星球由 `comet_planet_mask(state)` 覆盖，不算普通旋转星球。

## relative_features

`relative_features[b, src, tgt, :]` 描述从 source planet slot `src` 看 target slot `tgt`
的相对关系。`tgt=0` 是 no-op 行，特征全 0；`tgt=i+1` 对应 planet slot `i`。

```python
# 0  is_source: target 是否就是当前 source 自己。
# 1  dx: (target.x - source.x) / BOARD_SIZE。
# 2  dy: (target.y - source.y) / BOARD_SIZE。
# 3  dist: source 到 target 的距离 / BOARD_SIZE。
# 4  target_ship_ratio: clip01((target 船数 + 1) / max(source 船数, 1))。
# 5  source_ships: clip01(source 船数 / max_ships)，复制到每个 target 行。
# 6  proto_score: Proto-style 目标评分 / 100.0。
# 7  physics_eta_norm: full-send 物理层命中 tick / INBOUND_HORIZON；无命中则为 (INBOUND_HORIZON + 1) / INBOUND_HORIZON。
# 8  target_owner_at_eta_is_neutral: 按 ceil(physics_eta) 从 garrison pre-combat 表取样。
# 9  target_owner_at_eta_is_mine: 同上。
# 10 target_owner_at_eta_is_enemy: 同上。
# 11 target_ships_at_eta: 目标在 ETA 步生产后、到达结算前的驻军 / max_ships。
# 12 my_arrivals_through_eta: garrison 到达表中 <= ETA 步的我方到达船数 / max_ships。
# 13 enemy_arrivals_through_eta: garrison 到达表中 <= ETA 步的敌方到达船数 / max_ships。
# 14 eta_out_of_garrison_horizon: physics_eta 是否超过 DEFAULT_GARRISON_HORIZON。
# 15 pressure_delta: producer-style enemy_pressure[target] - enemy_pressure[source]，再除以 max_ships 并裁剪到 [-1, 1]。
# 16 target_enemy_pressure: producer-style enemy_pressure[target] / max_ships。
# 17 capture_margin_at_eta: (full_send_ships - target_ships_at_eta) / max_ships，裁剪到 [-1, 1]。
# 18 arrival_balance_at_eta: (my_arrivals_through_eta - enemy_arrivals_through_eta) / max_ships，裁剪到 [-1, 1]。
# 19 contestedness_at_eta: (my_arrivals_through_eta + enemy_arrivals_through_eta) / max_ships。
# 20 capture_floor_at_eta: producer-style ETA 到达时最低派兵门槛 / max_ships；友方目标为 1 船，非友方为 ceil(projected_defenders + 1)。
# 21 capture_surplus_at_eta: (full_send_ships - capture_floor_at_eta) / max_ships，裁剪到 [-1, 1]。
```

`proto_score(src, tgt)` 的实现：

```python
dist = distance(src, tgt)
min_ships = tgt.ships + 1
speed = fleet_speed(min_ships)
eta = dist / speed
enemy_produced = eta * tgt.production if tgt.owner != -1 else 0
enemy_bonus = tgt.production if tgt.owner != -1 else 0
total_ships = min_ships + enemy_produced
proto_score = (100 - dist) + 15 * tgt.production + 10 * enemy_bonus - 0.7 * total_ships - 2 * eta
```

这个分数是软提示。actor 最终仍通过 policy logits 选择 source 和 target；
当 `target_filter_enabled=true` 时，它还会参与候选目标 top-K 过滤。

## global_features

同一个 state 下，`global_features` 会复制到每个 source 行，模型实际读取
`global_features[:, 0]` 作为全局上下文。

```python
# 0  game_progress: 当前 step / episode_steps。
# 1  angular_velocity: 星球角速度 / 0.05。
# 2  my_planet_count: 我方存活星球数 / max_planets。
# 3  enemy_planet_count: 敌方存活星球数 / max_planets。
# 4  neutral_planet_count: 中立存活星球数 / max_planets。
# 5  my_ships_on_planets: 我方星球总船数 / (max_planets * max_ships)。
# 6  enemy_ships_on_planets: 敌方星球总船数 / (max_planets * max_ships)。
# 7  my_ships_in_fleets: 我方飞行舰队总船数 / (max_planets * max_ships)。
# 8  enemy_ships_in_fleets: 敌方飞行舰队总船数 / (max_planets * max_ships)。
# 9  my_total_production: 我方总生产力 / (max_planets * max_production)。
# 10 enemy_total_production: 敌方总生产力 / (max_planets * max_production)。
```

## fleet_features

`fleet_features` 是 per-state top-K 飞行舰队 token，然后复制到 source 维；
模型实际使用 `fleet_features[:, 0]` 和 `fleet_mask[:, 0]`，不是每个 source 单独筛一次。

当前主训练配置使用：

```yaml
model.max_fleet_tokens: 40
model.fleet_selection: impact
```

支持两种舰队排序方式，都是 `argsort(score)[:max_fleet_tokens]`，score 越小越优先保留：

```python
# fleet_selection == "size"
score = -fleet.ships + 0.01 * dist_to_center

# fleet_selection == "impact"
importance = (
    3.0 * enemy_fleet_to_my_planet
  + 1.6 * my_fleet_to_enemy_or_neutral
  + 0.8 * enemy_fleet_to_neutral
  + 0.7 * my_fleet_to_my_planet
  + 1.2 * urgency
  + 1.0 * ships_norm
  + 0.8 * target_production_norm
  + 0.4 * target_ships_norm
)
score = -importance + 1e-4 * dist_to_center / BOARD_SIZE
```

`urgency = 1 - target_eta_norm`，只在预测目标有效时有意义。无效或 dead fleet 的 score 为
`inf`，特征全 0，`fleet_mask=False`。

```python
# 0  is_my_fleet: 舰队是否属于我方；敌方舰队为 0。
# 1  fleet_x: 舰队 x 坐标 / BOARD_SIZE。
# 2  fleet_y: 舰队 y 坐标 / BOARD_SIZE。
# 3  cos_angle: 舰队飞行角度的 cos。
# 4  sin_angle: 舰队飞行角度的 sin。
# 5  fleet_ships: clip01(舰队船数 / max_ships)。
# 6  fleet_speed: fleet_speed(ships) / 6.0。
# 7  launch_age: 估算已飞行时间 / episode_steps。
# 8  from_x: 来源星球当前 x 坐标 / BOARD_SIZE；找不到来源时为 0。
# 9  from_y: 来源星球当前 y 坐标 / BOARD_SIZE；找不到来源时为 0。
# 10 from_is_mine: 来源星球当前是否属于我方。
# 11 from_ships: clip01(来源星球当前船数 / max_ships)；找不到来源时为 0。
# 12 dist_from_origin: 舰队离来源星球当前距离 / BOARD_SIZE；找不到来源时为 0。
# 13 nearest_enemy_planet_dist: 到最近敌方星球距离 / BOARD_SIZE。
# 14 nearest_neutral_planet_dist: 到最近中立星球距离 / BOARD_SIZE。
# 15 nearest_my_planet_dist: 到最近我方星球距离 / BOARD_SIZE。
# 16 target_valid: 未来 INBOUND_HORIZON 步内是否预测会命中存活星球。
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

`target_valid` 使用与 inbound 统计相同的轨迹预测：沿舰队当前方向向前模拟
`INBOUND_HORIZON=60` 步，取第一个会碰撞的存活星球。

## map_features

`map_features[b, src]` 是 8x8 全局棋盘栅格，不是围绕 source 裁剪的局部窗口。
它随 source 变化的部分主要是 channel 0 会标记当前 source slot。当前主训练配置
`use_cnn=true`，source head 会把这张图经过 CNN 后融合进 source key。

```python
# channel 0 source_planet: 当前 source 星球所在格子。
# channel 1 my_planets: 我方非 source 星球。
# channel 2 enemy_planets: 敌方星球。
# channel 3 neutral_planets: 中立星球。
# channel 4 my_ships: 我方飞行舰队船数强度，累加 clip01(ships / max_ships)。
# channel 5 enemy_ships: 敌方飞行舰队船数强度，累加 clip01(ships / max_ships)。
```

星球通道写入 `1.0`，但只有 `planet_alive=True` 的星球会实际写入。飞船强度通道按格子累加，
最后 clip 到 `[0, 1]`。格子坐标来自全局 `x / BOARD_SIZE`、`y / BOARD_SIZE`。

## Masks 和候选过滤

`source_mask[src]` 是 source pre-mask：

```python
source_mask = planet_alive
            & (owner == 0)
            & (ships >= MIN_LAUNCH)
            & candidate_mask[src, 1:].any()
```

也就是说，一个星球只有在“能发射且至少有一个非 no-op target 可选”时才会进入 source head。
这会提前过滤掉 target 全 mask 的 source，避免靠 rollout 里的 invalid token 事后跳过。

source head 的实际可选 mask 还会叠加动态剩余兵力：

```python
src_mask = [False] + source_mask
src_mask &= remaining_ships >= MIN_LAUNCH
src_mask[0] = False
```

`owned_mask[src]` 是更宽的我方星球统计 mask：

```python
owned_mask = planet_alive & (owner == 0)
```

target 的 pre-mask 来自 `candidate_mask[src, tgt]`，先做静态/几何粗筛，再做物理精筛：

```python
attackable_by_ships = source.ships > target.ships

target_valid = target_alive
             & (target != source)
             & reachable_by_full_send_fleet
             & (
                  target_is_friend
                  | ((target_is_enemy | target_is_neutral) & attackable_by_ships)
               )

candidate_mask[src] = [True] + target_valid
```

target head 会再把 `tgt=0` 关掉：

```python
tgt_mask = candidate_mask[src] & (target_slot != 0)
```

粗筛后，target 会进入 `physics_target_hit_eta_jax()`，用同一套物理层计算 full-send 是否可命中以及精确 ETA：

```python
angle, hit, eta_turns = aim_with_prediction_eta_jax(source, target, target_idx, state, full_send_ships)
eta_feature = eta_turns / INBOUND_HORIZON
candidate_mask[src, target] = target_valid & hit
```

这一步使用 Newton 连续拦截时间 + 相邻整数 tick 验证，并检查太阳、出界、首次碰撞对象和旋转/彗星 sweep。
ETA 相关 relative features、garrison-at-ETA 取样、友方 defense/pressure 的 ETA 衰减都使用这个物理层 ETA。

当 `target_filter_enabled=true` 时，物理精筛后的 `candidate_mask` 会继续被 `apply_target_filter()` 剪枝：

```python
# no-op slot 继续保留。
# attack targets: 在敌方/中立候选里按 proto_score 保留 attack_top_k 个。
# defense targets: 在友方候选里按 will_flip/pressure_delta/ETA 衰减后的防守增援分数保留 defense_top_k 个。
```

当前主训练配置 `attack_top_k=10`、`defense_top_k=5`，因此 target 候选不是“所有静态合法目标”，
而是已经经过 reachability、物理命中、Proto/防守增援 top-K 过滤后的目标集合。

因此 target head 在采样前已经看不到物理层打不到的目标。`source_mask` 也会叠加
`candidate_mask[src, 1:].any()`，没有任何可选 target 的 source 在 source head 阶段就会被关闭。
如果当前局面没有任何可执行 source，rollout 会强制生成 end token。

## full-send amount

当前 no-amount-head 分支不再训练 amount head，也不再生成 `amount_mask`。
一旦某个 source-target 动作通过 target pre-mask，派兵量由规则决定：

```python
ships = floor(remaining_ships[source])
ships = ships if ships >= MIN_LAUNCH else 0
```

因此 decoder 每执行一条命令后，该 source 的 remaining 会被扣到 0，同一 env step 中不会再次从该 source 发射。
target reachability 粗筛也使用 full-send 舰队速度判断目标是否物理可达。

## inbound collision features

`precompute_inbound_from_garrison()` 使用 `DEFAULT_GARRISON_HORIZON = 18`，并复用
`garrison_jax.GarrisonStatus.arrivals_by_owner`。这张表已经按 planet / step / owner 聚合当前存活舰队的首次碰撞到达量。
返回 `[P, 6]`：

```python
my_ships
enemy_ships
my_eta_min
my_eta_mean
enemy_eta_min
enemy_eta_mean
```

碰撞预测与 `predict_fleet_targets()` 一致：

```python
# 1. 舰队移动碰撞：
#    fleet[t-1] -> fleet[t] 线段检测 planet[t-1]。
#
# 2. 星球 sweep 碰撞：
#    planet[t-1] -> planet[t] 线段检测 fleet[t]。
#
# 3. 同一个 tick 内，move hit 的优先级高于 sweep hit。
#    move key = 2 * t，sweep key = 2 * t + 1。
```

旋转星球按当前 `angular_velocity` 前推；彗星星球按 `state.comet_paths` 和
`state.comet_path_index` 前推。若舰队在 60 步内没有命中任何存活星球，则不计入 inbound。

## 运行时归一化

`normalization_jax.py` 只归一化 `planet_features`、`relative_features` 和 `global_features`
中的连续列；布尔/类别列、mask、`map_features`、`fleet_features` 不参与这层 running statistics。

归一化列：

```python
PLANET_NORMALIZE_INDICES = [5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
RELATIVE_NORMALIZE_INDICES = [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21]
GLOBAL_NORMALIZE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

统计量更新 mask：

```python
# planet stats:
#   用 owned_mask=True 的我方存活星球行更新统计量。
#
# relative stats:
#   只用 source_mask=True 的 source 行，并排除 candidate_ids < 0 的 no-op target 行。
#
# global stats:
#   用 owned_mask=True 的行更新；同一个 state 会按我方存活星球数重复计入。
```

归一化公式：

```python
normalized = clip((x - running_mean) / sqrt(running_var + eps), -clip, clip)
```

当前主训练配置：

```yaml
state_norm.enabled: true
state_norm.clip: 5.0
state_norm.eps: 1.0e-8
```

checkpoint 和 ONNX 导出会一起携带 normalizer 状态；评估/推理必须使用同一份统计量。

## 模型使用方式

当前 actor 是状态级 autoregressive actor-critic：

```python
# Encoder:
#   1. planet_features 前置 no-op planet token。
#   2. fleet_features 使用 shared_fleet_features = fleet_features[:, 0]。
#   3. CLS token、planet tokens、fleet tokens 加 type embedding 后进入 Transformer。
#   4. global_features[:, 0] 经过 MLP，和 CLS 输出拼成 state_emb。
#   5. relative_features 单独经过 MLP，供 target head 使用。
#
# Source head:
#   使用 planet_out/source keys。
#   use_cnn=true 时融合 map_features 的 CNN 表示。
#   mask = source_mask & remaining_ships >= MIN_LAUNCH，且 src slot 0 永远关闭。
#
# Target head:
#   使用已选 source、planet_out、relative_h、candidate_mask。
#   target slot 0 永远关闭。
#
# End head:
#   每个 decoder token 先判断是否停止生成；停止动作不依赖 target no-op。
```

`model.use_transformer` 是配置里的兼容字段，当前 `AutoregressiveActorCritic` 实现本身总是按
`num_transformer_layers` 构建 Transformer 编码层。设为 0 层才会实质跳过 Transformer block。

rollout 时，模型输出的是 `(end, src, tgt)` token；真正提交给环境的动作是：

```python
[source_planet_id, launch_angle, ships]
```

`launch_angle` 由同一套 Newton aiming 逻辑根据 source、target、ships 计算。由于 target head 已经经过
target pre-mask，已选 target 的物理可命中性在采样前确定；rollout 阶段只重新计算发射角，不再用 `aim_hit`
做事后否决。若没有合法 source 会强制 end；若采样到 source/target pre-mask 外的编号，该 token 仍会标为
`invalid`，不会真实发射，decoder 下一步会收到 `invalid_token`。
