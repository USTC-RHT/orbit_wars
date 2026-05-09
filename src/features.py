"""特征工程和全图目标槽位生成。

把 Kaggle observation 转成固定形状的神经网络输入；同时保存
DecisionContext，用于把网络输出的 target/amount 还原成真实舰队动作。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import EnvConfig
from .state import GameState, Planet, parse_observation

CENTER = (50.0, 50.0)
ROTATION_LIMIT = 50.0
INBOUND_HORIZON = 60  # 预测未来 60 回合内舰队与星球的碰撞
SELF_DIM = 18
CANDIDATE_DIM = 26
GLOBAL_DIM = 11
MAP_GRID_SIZE = 8
MAP_CHANNELS = 6



@dataclass(slots=True)
class DecisionContext:
    """一行网络输出对应的真实游戏上下文。

    网络只输出 target_index 和 amount_bucket。这里保存 source_id、
    candidate_ids 和 ship_buckets。candidate_ids[0] 固定为 no-op，
    后续槽位对应当前全图星球；真正发射角在动作解码阶段根据
    src + target + ships 调 physics 计算。
    """

    source_id: int
    source_ships: int
    candidate_ids: list[int]
    candidate_mask: np.ndarray
    amount_mask: np.ndarray
    ship_buckets: list[int]


@dataclass(slots=True)
class TurnBatch:
    """一次 obs 编码后的批数据。

    self_features: 每颗我方星球一行。
    candidate_features: 每行对应 no-op + 全图星球槽位。
    global_features: 全局局势，按来源星球重复。
    map_features: 以当前来源星球为第 0 通道的小地图特征，形状 [N, 6, 8, 8]。
    candidate_mask: 哪些目标槽位可选。
    amount_mask: 轻量合法性 mask；物理可达性在动作解码阶段只检查已选结果。
    contexts: 把网络动作映射回游戏动作所需的信息。
    """

    self_features: np.ndarray
    candidate_features: np.ndarray
    global_features: np.ndarray
    map_features: np.ndarray
    candidate_mask: np.ndarray
    amount_mask: np.ndarray
    contexts: list[DecisionContext]
    state: GameState


# 详细特征定义见 RL_new/FEATURES.md。


def target_count(cfg: EnvConfig) -> int:
    """网络可见的目标槽位数：0 是 no-op，后面是最多 max_planets 个星球。"""

    return cfg.max_planets + 1


def encode_turn(observation: Any, cfg: EnvConfig) -> TurnBatch:
    """把 Kaggle observation 编码成网络输入。

    Orbit Wars 初始星球数量每局不同，彗星也会中途加入，所以神经网络
    需要固定维度输入。这里把每颗我方星球拆成一行，并把全图当前星球
    padding 到固定槽位里。

    槽位 0 永远是 no-op；槽位 1..max_planets 是当前全图星球，按 id
    排序填入。这样网络不再依赖手写候选目标规则，而是自己从全图选目标。
    """

    state = observation if isinstance(observation, GameState) else parse_observation(observation)
    n_targets = target_count(cfg)
    sources = sorted([p for p in state.planets if p.owner == state.player], key=lambda p: p.id)
    if not sources:
        return TurnBatch(
            self_features=np.zeros((0, SELF_DIM), dtype=np.float32),
            candidate_features=np.zeros((0, n_targets, CANDIDATE_DIM), dtype=np.float32),
            global_features=np.zeros((0, GLOBAL_DIM), dtype=np.float32),
            map_features=np.zeros((0, MAP_CHANNELS, MAP_GRID_SIZE, MAP_GRID_SIZE), dtype=np.float32),
            candidate_mask=np.zeros((0, n_targets), dtype=bool),
            amount_mask=np.zeros((0, n_targets, cfg.ship_bucket_count), dtype=bool),
            contexts=[],
            state=state,
        )

    global_features = build_global_features(state, cfg)
    inbound = precompute_inbound(state)
    self_rows = []
    candidate_rows = []
    map_rows = []
    masks = []
    amount_masks = []
    contexts = []
    for src in sources:
        cand_features, mask, amount_mask, ship_buckets, candidate_ids = build_candidate_features(src, state, cfg, inbound)
        self_rows.append(build_self_features(src, state, cfg, inbound))
        candidate_rows.append(cand_features)
        map_rows.append(build_map_features(src, state, cfg))
        masks.append(mask)
        amount_masks.append(amount_mask)
        contexts.append(
            DecisionContext(
                source_id=src.id,
                source_ships=int(src.ships),
                candidate_ids=candidate_ids,
                candidate_mask=mask,
                amount_mask=amount_mask,
                ship_buckets=ship_buckets,
            )
        )

    return TurnBatch(
        self_features=np.asarray(self_rows, dtype=np.float32),
        candidate_features=np.asarray(candidate_rows, dtype=np.float32),
        global_features=np.repeat(global_features[None, :], len(self_rows), axis=0),
        map_features=np.asarray(map_rows, dtype=np.float32),
        candidate_mask=np.asarray(masks, dtype=bool),
        amount_mask=np.asarray(amount_masks, dtype=bool),
        contexts=contexts,
        state=state,
    )


def build_self_features(
    src: Planet,
    state: GameState,
    cfg: EnvConfig,
    inbound: dict[int, tuple[float, float, float, float, float, float]] | None = None,
) -> np.ndarray:
    """构建来源星球自身特征。

    一颗我方星球对应一行决策。这里描述“我从哪里发兵”以及这颗星球
    所处的粗略局势。具体每一维见 RL_new/FEATURES.md。
    """

    my_planets = [p for p in state.planets if p.owner == state.player]
    enemy_planets = [p for p in state.planets if p.owner not in {-1, state.player}]
    my_fleets = [f for f in state.fleets if f.owner == state.player]
    enemy_fleets = [f for f in state.fleets if f.owner != state.player]
    inb = (inbound or {}).get(src.id, (0.0, 0.0, 1.0, 1.0, 1.0, 1.0))
    return np.asarray(
        [
            1.0,
            src.x / cfg.board_size,
            src.y / cfg.board_size,
            src.radius / 5.0,
            clip01(src.ships / cfg.max_ships),
            src.production / cfg.max_production,
            1.0 if is_rotating(src) else 0.0,
            distance_to_sun(src) / cfg.board_size,
            len(my_planets) / cfg.max_planets,
            len(enemy_planets) / cfg.max_planets,
            total_ships(my_planets) / (cfg.max_planets * cfg.max_ships),
            (sum(f.ships for f in enemy_fleets) - sum(f.ships for f in my_fleets))
            / (cfg.max_planets * cfg.max_ships),
            clip01(inb[0] / cfg.max_ships),
            clip01(inb[1] / cfg.max_ships),
            inb[2],
            inb[3],
            inb[4],
            inb[5],
        ],
        dtype=np.float32,
    )


def build_candidate_features(
    src: Planet,
    state: GameState,
    cfg: EnvConfig,
    inbound: dict[int, tuple[float, float, float, float, float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int]]:
    """构建全图目标槽位特征，以及目标 mask 和派兵档位。

    注意：所有真实星球都会被 encode 到 features 里，包括来源星球自己。
    mask 只影响动作是否能选，不影响目标进入 target_encoder / pooling。

    target mask 只过滤 padding、来源星球自身、当前不能发兵等基础非法项；
    网络选中 target + amount 后，再在 action_from_indices 中只检查这一条
    轨迹是否能真实命中。
    """

    n_targets = target_count(cfg)
    features = np.zeros((n_targets, CANDIDATE_DIM), dtype=np.float32)
    mask = np.zeros((n_targets,), dtype=bool)
    amount_mask = np.zeros((n_targets, cfg.ship_bucket_count), dtype=bool)
    ship_buckets = build_ship_buckets(src, cfg)
    candidate_ids = [-1 for _ in range(n_targets)]

    # [slot 0] 固定为 no-op（不发兵）。
    # 这个槽位没有星球坐标，所以除了 is_noop 外其它特征保持 0。
    # mask[0]=True 保证网络始终能选择“本来源星球不行动”。
    mask[0] = True
    amount_mask[0, :] = True
    features[0, 0] = 1.0

    valid_buckets = [b for b in ship_buckets if b > 0]
    max_bucket_ships = max(valid_buckets, default=1)
    log_r_max = math.log(max(1, max_bucket_ships)) / math.log(1000.0)
    spd_max = min(1.0 + 5.0 * (log_r_max ** 1.5), 6.0)
    comet_ids = set(state.comet_planet_ids)
    planets = sorted(state.planets, key=lambda p: p.id)[: cfg.max_planets]
    for idx, tgt in enumerate(planets, start=1):
        dx = tgt.x - src.x
        dy = tgt.y - src.y
        is_source = tgt.id == src.id
        can_send = (not is_source) and src.ships > 0 and bool(valid_buckets)
        owner_is_neutral = tgt.owner == -1
        owner_is_friend = tgt.owner == state.player and not is_source
        owner_is_enemy = tgt.owner not in {-1, state.player}
        inb = (inbound or {}).get(tgt.id, (0.0, 0.0, 1.0, 1.0, 1.0, 1.0))
        raw_eta_max_fleet = distance(src, tgt) / max(spd_max, 1e-6)
        eta_max_fleet = raw_eta_max_fleet / INBOUND_HORIZON

        # [slot idx] 一个真实星球目标。即使它是 source 自己，也会被编码；
        # 只是 candidate_mask 会禁止把自己选为目标。
        features[idx] = np.asarray(
            [
                0.0,  # is_noop
                1.0,  # is_planet
                1.0 if is_source else 0.0,  # is_source
                1.0 if owner_is_neutral else 0.0,  # is_neutral
                1.0 if owner_is_friend else 0.0,  # is_friend
                1.0 if owner_is_enemy else 0.0,  # is_enemy
                tgt.x / cfg.board_size,  # tgt_x
                tgt.y / cfg.board_size,  # tgt_y
                dx / cfg.board_size,  # dx_to_src
                dy / cfg.board_size,  # dy_to_src
                distance(src, tgt) / cfg.board_size,  # dist_to_src
                tgt.radius / 5.0,  # tgt_radius
                clip01(tgt.ships / cfg.max_ships),  # tgt_ships
                tgt.production / cfg.max_production,  # tgt_production
                1.0 if is_rotating(tgt) else 0.0,  # tgt_is_rotating
                1.0 if tgt.id in comet_ids else 0.0,  # tgt_is_comet
                clip01((tgt.ships + 1) / max(src.ships, 1)),  # target_ship_ratio
                clip01(src.ships / cfg.max_ships),  # src_ships_again
                clip01(inb[0] / cfg.max_ships),  # my_inbound_ships
                clip01(inb[1] / cfg.max_ships),  # enemy_inbound_ships
                inb[2],  # my_inbound_eta_min
                inb[3],  # my_inbound_eta_mean
                inb[4],  # enemy_inbound_eta_min
                inb[5],  # enemy_inbound_eta_mean
                _proto_score(src, tgt) / 100.0,  # proto_score
                eta_max_fleet,  # 以最大档兵力出发到达该目标的实际 ETA / INBOUND_HORIZON
            ],
            dtype=np.float32,
        )
        if can_send:
            if owner_is_friend:
                # 增援友方星球：任意正数均可
                amount_mask[idx] = np.asarray([ships > 0 for ships in ship_buckets], dtype=bool)
                mask[idx] = True
            elif owner_is_enemy:
                # 攻击敌方星球：必须发送足够多的船才能攻占（ships > 驻军数）。
                # 若所有档位都不够，则屏蔽整个目标，强制 agent 先积攒兵力。
                viable = [ships > tgt.ships for ships in ship_buckets]
                amount_mask[idx] = np.asarray(viable, dtype=bool)
                mask[idx] = any(viable)
            else:
                # 中立星球：与敌方相同，必须发送足够多的船才能攻占（ships > 驻军数）。
                viable = [ships > tgt.ships for ships in ship_buckets]
                amount_mask[idx] = np.asarray(viable, dtype=bool)
                mask[idx] = any(viable)
        candidate_ids[idx] = tgt.id

    return features, mask, amount_mask, ship_buckets, candidate_ids


def build_global_features(state: GameState, cfg: EnvConfig) -> np.ndarray:
    """构建全局局势特征。

    这部分描述整局游戏的粗略态势，按来源星球重复给网络。
    """

    my_planets = [p for p in state.planets if p.owner == state.player]
    enemy_planets = [p for p in state.planets if p.owner not in {-1, state.player}]
    neutral_planets = [p for p in state.planets if p.owner == -1]
    my_fleets = [f for f in state.fleets if f.owner == state.player]
    enemy_fleets = [f for f in state.fleets if f.owner != state.player]
    my_prod = sum(p.production for p in my_planets)
    enemy_prod = sum(p.production for p in enemy_planets)
    return np.asarray(
        [
            state.step / cfg.episode_steps,
            state.angular_velocity / 0.05,
            len(my_planets) / cfg.max_planets,
            len(enemy_planets) / cfg.max_planets,
            len(neutral_planets) / cfg.max_planets,
            total_ships(my_planets) / (cfg.max_planets * cfg.max_ships),
            total_ships(enemy_planets) / (cfg.max_planets * cfg.max_ships),
            sum(f.ships for f in my_fleets) / (cfg.max_planets * cfg.max_ships),
            sum(f.ships for f in enemy_fleets) / (cfg.max_planets * cfg.max_ships),
            my_prod / (cfg.max_planets * cfg.max_production),
            enemy_prod / (cfg.max_planets * cfg.max_production),
        ],
        dtype=np.float32,
    )


def build_map_features(src: Planet, state: GameState, cfg: EnvConfig) -> np.ndarray:
    """构建以当前来源星球为参照的小地图特征。

    通道含义：
    0 当前星球；1 友方星球（不含当前星球）；2 敌方星球；3 中立星球；
    4 友方飞行舰队船数；5 敌方飞行舰队船数。
    """

    grid = np.zeros((MAP_CHANNELS, MAP_GRID_SIZE, MAP_GRID_SIZE), dtype=np.float32)
    for planet in state.planets:
        if planet.id == src.id:
            _accum_grid(grid, 0, planet.x, planet.y, 1.0, cfg)
        elif planet.owner == state.player:
            _accum_grid(grid, 1, planet.x, planet.y, 1.0, cfg)
        elif planet.owner == -1:
            _accum_grid(grid, 3, planet.x, planet.y, 1.0, cfg)
        else:
            _accum_grid(grid, 2, planet.x, planet.y, 1.0, cfg)

    for fleet in state.fleets:
        channel = 4 if fleet.owner == state.player else 5
        value = clip01(float(fleet.ships) / cfg.max_ships)
        _accum_grid(grid, channel, fleet.x, fleet.y, value, cfg)
    return grid


def _accum_grid(grid: np.ndarray, channel: int, x: float, y: float, value: float, cfg: EnvConfig) -> None:
    if x < 0.0 or y < 0.0 or x > cfg.board_size or y > cfg.board_size:
        return
    ix = min(MAP_GRID_SIZE - 1, max(0, int((x / cfg.board_size) * MAP_GRID_SIZE)))
    iy = min(MAP_GRID_SIZE - 1, max(0, int((y / cfg.board_size) * MAP_GRID_SIZE)))
    grid[channel, iy, ix] = min(1.0, grid[channel, iy, ix] + float(value))


def build_ship_buckets(src: Planet, cfg: EnvConfig) -> list[int]:
    """把连续派兵数量离散成几个档位，供网络直接选择。

    不再设置最小派兵或最大派兵比例限制。只要来源星球有船，档位会覆盖
    1 艘到全部船数的范围。例如 ship_bucket_count=8、src.ships=100 时，
    档位约为 12.5%、25%、...、100%。

    [设计] 把连续动作空间离散化为 Categorical，避免使用 Gaussian policy
    （需要调 std 参数且容易数值不稳定）。代价是损失精细控制粒度，
    但对于策略游戏"多少兵"的决策，6 个档位在实践中已经足够。
    """
    if src.ships <= 0:
        return [0 for _ in range(cfg.ship_bucket_count)]
    s = int(src.ships)
    # 比例档位覆盖常见战术语义：试探/半攻/主攻/All-in。
    # 比起均匀线性档位，更容易踩中“刚好攻占 nearest 守军”的关键数量。
    MIN_LAUNCH = 5  # 绝对下限：星球兵力不足 MIN_LAUNCH 时禁止发兵（全部档位为 0）。
    if s < MIN_LAUNCH:
        return [0 for _ in range(cfg.ship_bucket_count)]
    ratios = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    raw = [max(MIN_LAUNCH, min(s, int(round(s * r)))) for r in ratios]
    # 去重后保留升序，再补齐到 ship_bucket_count（重复 last 值不影响 Categorical）。
    seen: list[int] = []
    for v in raw:
        if v not in seen:
            seen.append(v)
    while len(seen) < cfg.ship_bucket_count:
        seen.append(seen[-1])
    return seen[: cfg.ship_bucket_count]


def _proto_score(src: Planet, tgt: Planet) -> float:
    """Proto-V15 目标评分公式。

    score = (100-dist) + 15*production + 10*enemy_bonus - 0.7*total_ships - 2*eta

    enemy_bonus = tgt.production  (仅当目标有主时)
    enemy_produced = eta * tgt.production  (仅当目标有主时，中立星球不生产增援)
    total_ships = (tgt.ships+1) + enemy_produced
    eta = dist / fleet_speed(tgt.ships+1)

    返回原始分数（未归一化），调用方除以 100 后再 clip。
    """
    dist = distance(src, tgt)
    min_ships = tgt.ships + 1
    log_ratio = math.log(max(1, min_ships)) / math.log(1000.0)
    speed = 1.0 + 5.0 * (log_ratio ** 1.5)
    speed = min(speed, 6.0)
    eta = dist / speed if speed > 0 else dist
    if tgt.owner != -1:
        enemy_produced = eta * tgt.production
        enemy_bonus = tgt.production
    else:
        enemy_produced = 0.0
        enemy_bonus = 0
    total_ships = min_ships + enemy_produced
    return (100.0 - dist) + 15.0 * tgt.production + 10.0 * enemy_bonus - 0.7 * total_ships - 2.0 * eta


def is_rotating(planet: Planet) -> bool:
    return distance_to_sun(planet) + planet.radius < ROTATION_LIMIT


def precompute_inbound(
    state: GameState,
    horizon: int = INBOUND_HORIZON,
) -> dict[int, tuple[float, float, float, float, float, float]]:
    """并行预测未来 `horizon` 回合内每颗星球被舰队击中的情况。

    返回 dict[planet_id] -> (
        my_ships, enemy_ships, my_eta_min, my_eta_mean, enemy_eta_min, enemy_eta_mean
    )

    实现要点：
    - fleets 走直线（vx,vy 由 ship 数量决定速度）。
    - 静止星球：位置不变。
    - 旋转星球：参数方程 x=50+r*cos(phi+omega*T), y=50+r*sin(...)。
    - 彗星：从存好的 path 数组按 path_index+T 取位置。
    - 每个 tick 先检查舰队移动线段撞星球，再检查星球 sweep 撞舰队。
    - 一个 fleet 只归属第一次撞到的星球（避免重复计数）。

    所有重计算用 numpy 广播，规模约 F*P*horizon ≤ 30*45*60 = 81k；
    比点距离近似更贵，但和 physics.py / 引擎的碰撞阶段一致。
    """

    planet_ids = [p.id for p in state.planets]
    if not planet_ids:
        return {}
    empty = (0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
    if not state.fleets:
        return {pid: empty for pid in planet_ids}

    F = len(state.fleets)
    P = len(state.planets)
    T = horizon

    # ── 舰队参数 ──
    fx0 = np.fromiter((f.x for f in state.fleets), dtype=np.float32, count=F)
    fy0 = np.fromiter((f.y for f in state.fleets), dtype=np.float32, count=F)
    angle = np.fromiter((f.angle for f in state.fleets), dtype=np.float32, count=F)
    ships = np.fromiter((max(1, int(f.ships)) for f in state.fleets), dtype=np.float32, count=F)
    owner = np.fromiter((f.owner for f in state.fleets), dtype=np.int32, count=F)

    log_ratio = np.clip(np.log(ships) / math.log(1000.0), 0.0, 1.0)
    speed = np.minimum(1.0 + 5.0 * log_ratio ** 1.5, 6.0).astype(np.float32)
    speed = np.where(ships <= 1, 1.0, speed).astype(np.float32)
    vx = np.cos(angle) * speed
    vy = np.sin(angle) * speed

    ts = np.arange(T + 1, dtype=np.float32)  # (T+1,)
    fx_t = fx0[:, None] + vx[:, None] * ts[None, :]  # (F, T+1)
    fy_t = fy0[:, None] + vy[:, None] * ts[None, :]
    in_bounds = (
        (fx_t >= 0.0) & (fx_t <= 100.0) & (fy_t >= 0.0) & (fy_t <= 100.0)
    )

    # ── 星球位置 ──
    px_t = np.zeros((P, T + 1), dtype=np.float32)
    py_t = np.zeros((P, T + 1), dtype=np.float32)
    radii = np.fromiter((p.radius for p in state.planets), dtype=np.float32, count=P)

    initial_by_id = {ip.id: ip for ip in state.initial_planets}
    comet_set = set(state.comet_planet_ids)
    omega = state.angular_velocity

    # 彗星路径表
    comet_path: dict[int, tuple[np.ndarray, int]] = {}
    for grp in state.comets:
        pids = grp.get("planet_ids", []) if isinstance(grp, dict) else []
        pi = grp.get("path_index", 0) if isinstance(grp, dict) else 0
        paths = grp.get("paths", []) if isinstance(grp, dict) else []
        for i, pid in enumerate(pids):
            if i < len(paths) and paths[i]:
                comet_path[pid] = (np.asarray(paths[i], dtype=np.float32), pi)

    for j, p in enumerate(state.planets):
        if p.id in comet_set:
            data = comet_path.get(p.id)
            if data is None:
                px_t[j, :] = p.x
                py_t[j, :] = p.y
                continue
            path, pi = data
            for t_step in range(T + 1):
                idx = pi + t_step
                if 0 <= idx < len(path):
                    px_t[j, t_step] = path[idx][0]
                    py_t[j, t_step] = path[idx][1]
                else:
                    # 彗星离场后用远离地图的坐标，确保不会触发碰撞。
                    px_t[j, t_step] = -1000.0
                    py_t[j, t_step] = -1000.0
            continue

        ip = initial_by_id.get(p.id)
        if ip is None:
            px_t[j, :] = p.x
            py_t[j, :] = p.y
            continue
        dx = ip.x - 50.0
        dy = ip.y - 50.0
        r_orbit = math.hypot(dx, dy)
        if r_orbit + ip.radius >= ROTATION_LIMIT:  # 静止
            px_t[j, :] = p.x
            py_t[j, :] = p.y
        else:
            phi_cur = math.atan2(p.y - 50.0, p.x - 50.0)
            phi = phi_cur + omega * ts
            px_t[j, :] = 50.0 + r_orbit * np.cos(phi)
            py_t[j, :] = 50.0 + r_orbit * np.sin(phi)

    # ── 引擎同款两阶段碰撞 (F, P, T) ──
    # step=1..T。引擎每步先移动舰队，用 step-1 的星球位置做线段-圆碰撞；
    # 然后移动旋转星球/彗星，用星球 old->new 线段扫过舰队当前位置。
    eps = np.float32(1e-9)
    radius_sq = (radii[None, :, None] ** 2).astype(np.float32)
    cur_in_bounds = in_bounds[:, 1:][:, None, :]

    fx_prev = fx_t[:, :-1]
    fy_prev = fy_t[:, :-1]
    fx_cur = fx_t[:, 1:]
    fy_cur = fy_t[:, 1:]
    px_old = px_t[:, :-1]
    py_old = py_t[:, :-1]
    px_new = px_t[:, 1:]
    py_new = py_t[:, 1:]

    # 舰队移动阶段：point=planet(step-1), segment=fleet(step-1)->fleet(step)。
    seg_x = fx_cur[:, None, :] - fx_prev[:, None, :]
    seg_y = fy_cur[:, None, :] - fy_prev[:, None, :]
    den = seg_x * seg_x + seg_y * seg_y
    rel_x = px_old[None, :, :] - fx_prev[:, None, :]
    rel_y = py_old[None, :, :] - fy_prev[:, None, :]
    proj = np.clip((rel_x * seg_x + rel_y * seg_y) / np.maximum(den, eps), 0.0, 1.0)
    closest_x = fx_prev[:, None, :] + proj * seg_x
    closest_y = fy_prev[:, None, :] + proj * seg_y
    dx_move = px_old[None, :, :] - closest_x
    dy_move = py_old[None, :, :] - closest_y
    move_hit = (dx_move * dx_move + dy_move * dy_move) < radius_sq
    move_hit &= cur_in_bounds

    # 星球 sweep 阶段：point=fleet(step), segment=planet(step-1)->planet(step)。
    sweep_x = px_new - px_old
    sweep_y = py_new - py_old
    sweep_den = sweep_x * sweep_x + sweep_y * sweep_y
    sweep_active = sweep_den > eps
    rel_x = fx_cur[:, None, :] - px_old[None, :, :]
    rel_y = fy_cur[:, None, :] - py_old[None, :, :]
    proj = np.clip(
        (rel_x * sweep_x[None, :, :] + rel_y * sweep_y[None, :, :])
        / np.maximum(sweep_den[None, :, :], eps),
        0.0,
        1.0,
    )
    closest_x = px_old[None, :, :] + proj * sweep_x[None, :, :]
    closest_y = py_old[None, :, :] + proj * sweep_y[None, :, :]
    dx_sweep = fx_cur[:, None, :] - closest_x
    dy_sweep = fy_cur[:, None, :] - closest_y
    sweep_hit = (dx_sweep * dx_sweep + dy_sweep * dy_sweep) < radius_sq
    sweep_hit &= sweep_active[None, :, :]
    sweep_hit &= cur_in_bounds

    # 用 phase key 保留引擎顺序：同一 tick 内 move_hit 早于 sweep_hit。
    large_key = np.int32(2 * (T + 1) + 1)
    steps = np.arange(1, T + 1, dtype=np.int32)
    move_keys = (2 * steps)[None, None, :]
    sweep_keys = (2 * steps + 1)[None, None, :]
    first_move_key = np.where(move_hit, move_keys, large_key).min(axis=2)
    first_sweep_key = np.where(sweep_hit, sweep_keys, large_key).min(axis=2)
    first_key_fp = np.minimum(first_move_key, first_sweep_key)
    first_t_fp = (first_key_fp // 2).astype(np.int32)

    # 每个 fleet 只归属最早撞到的那一颗星球
    fleet_first_key = first_key_fp.min(axis=1)  # (F,)
    fleet_target = first_key_fp.argmin(axis=1)  # (F,)
    fleet_valid = fleet_first_key < large_key
    fleet_first_t = (fleet_first_key // 2).astype(np.int32)

    result: dict[int, tuple[float, float, float, float, float, float]] = {}
    norm = float(T)
    for j, pid in enumerate(planet_ids):
        sel = fleet_valid & (fleet_target == j)
        if not sel.any():
            result[pid] = empty
            continue
        my = sel & (owner == state.player)
        en = sel & (owner != state.player)
        my_ships = float(ships[my].sum()) if my.any() else 0.0
        en_ships = float(ships[en].sum()) if en.any() else 0.0
        my_eta = fleet_first_t[my]
        en_eta = fleet_first_t[en]
        my_min = float(my_eta.min()) / norm if my.any() else 1.0
        my_mean = float(my_eta.mean()) / norm if my.any() else 1.0
        en_min = float(en_eta.min()) / norm if en.any() else 1.0
        en_mean = float(en_eta.mean()) / norm if en.any() else 1.0
        result[pid] = (my_ships, en_ships, my_min, my_mean, en_min, en_mean)
    return result


def distance_to_sun(planet: Planet) -> float:
    return math.hypot(planet.x - CENTER[0], planet.y - CENTER[1])


def distance(a: Planet, b: Planet) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def total_ships(planets: list[Planet]) -> float:
    return float(sum(p.ships for p in planets))


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
