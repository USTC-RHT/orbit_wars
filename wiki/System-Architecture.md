# 系统架构

> **所属页面**：[Home](Home.md) > 系统架构

---

## 1. 整体架构图

```
浏览器
  │  Hash 路由 (#/quick-match, #/leaderboard …)
  │  REST 请求 (fetch /api/*)
  │  SSE 流  (EventSource /api/runs/:id/stream)
  ▼
┌─────────────────────────────────────────────────────┐
│            Vite 前端 SPA (TypeScript)               │
│  router.ts → views/* → components/*                │
│  api.ts (类型化 fetch 封装)                          │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP / SSE
                       ▼
┌─────────────────────────────────────────────────────┐
│         FastAPI 后端 (Python 3.12)                  │
│  main.py → api.py (Router)                          │
│                                                     │
│  tournament.py   ProcessPoolExecutor                │
│  match.py        kaggle-environments 引擎            │
│  trueskill_store.py  TrueSkill 评分 JSON             │
│  discovery.py    扫描 agents/ 目录                   │
│  replay_store.py 读写 runs/ 目录                    │
│  kaggle_auth.py  Kaggle API Token 管理               │
│  kaggle_submissions.py  拉取 LB 数据                 │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
              本地文件系统
          agents/   runs/   viewer/dist/
```

开发模式：前端 `:6001`（Vite HMR），后端 `:8000`（uvicorn --reload）。  
生产模式：后端直接以 `StaticFiles` 托管前端构建产物，统一 `:8000`。

---

## 2. 后端详解

### 2.1 入口：`orbit_wars_app/main.py`

```python
app = FastAPI(title="Orbit Wars Lab", lifespan=lifespan)
app.include_router(api.router)        # 挂载 /api/* 路由

# 生产模式：托管前端构建产物
if VIEWER_DIST.is_dir():
    app.mount("/", StaticFiles(directory=VIEWER_DIST, html=True))
```

`lifespan` 上下文管理器在应用关闭时优雅停止 `ProcessPoolExecutor`，防止僵尸进程。

---

### 2.2 API 路由：`orbit_wars_app/api.py`

所有接口挂载在 `/api` 前缀下：

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/api/health` | 健康检查，返回版本号 |
| GET | `/api/agents` | 列出全部 Agent（扫描 `agents/` 目录） |
| GET | `/api/agents/{id}` | 获取单个 Agent 详情 |
| GET | `/api/ratings` | 获取 TrueSkill 排行榜（2p/4p） |
| GET | `/api/runs` | 列出所有赛事（可过滤 quick match） |
| GET | `/api/runs/{id}` | 获取单个赛事详情 + 比赛结果列表 |
| POST | `/api/tournaments` | 创建并启动新赛事 |
| GET | `/api/runs/{id}/stream` | SSE 流，实时推送赛事进度 |
| DELETE | `/api/runs/{id}` | 删除赛事及其所有 replay |
| GET | `/api/replays` | 列出所有 replay（本地 + Kaggle） |
| GET | `/api/replays/{run}/{match}` | 读取单个 replay JSON |
| GET | `/api/kaggle-submissions` | 列出 Kaggle LB 提交记录 |
| POST | `/api/kaggle-settings` | 保存 Kaggle API Token |
| GET | `/api/kaggle-settings/status` | 检测 Token 是否有效 |

#### 路径穿越安全防护

```python
def _safe_subpath(parent: Path, child_name: str) -> Path:
    joined = parent / child_name
    joined.resolve().relative_to(parent.resolve())  # 拒绝 ../ 跳出
    return joined
```

所有以文件路径为参数的接口都经过此检查，防止 **目录穿越攻击（Path Traversal）**。

---

### 2.3 赛事系统：`tournament.py` + `match.py`

#### 两种执行模式

| 模式 | 实现方式 | 适用场景 |
|------|----------|----------|
| **fast（快速）** | 直接 `import` Agent 代码，在当前进程内运行 | 内置 Agent，速度快 |
| **faithful（忠实）** | 在独立子进程中运行 Agent（HTTP 子进程隔离） | 外部/不受信任 Agent，安全隔离 |

#### 并行执行

```python
with ProcessPoolExecutor(
    max_workers=workers,
    initializer=_quiet_kaggle_environments,  # 抑制多余日志
) as pool:
    futures = {
        pool.submit(_run_match_in_worker, ...): pair
        for pair in match_pairs
    }
    for future in as_completed(futures):
        result = future.result()
        # 更新 TrueSkill、保存 replay、广播 SSE 进度
```

> **为什么用 ProcessPoolExecutor 而非 ThreadPoolExecutor？**  
> Python GIL 限制多线程 CPU 并行性，`kaggle-environments` 运行 Agent 属于 CPU 密集任务，多进程可真正利用多核，实测 3–4× 加速。

#### 赛制格式

| 格式 | 说明 | 适用场景 |
|------|------|----------|
| **round-robin（循环赛）** | 每对 Agent 互打 K 局，全面评估整体实力 | 建立排行榜 |
| **gauntlet（挑战赛）** | 1 个挑战者 vs 所有其他 Agent | 快速评估新 Agent 相对排名 |

---

### 2.4 TrueSkill 评分：`trueskill_store.py`

本项目使用微软 TrueSkill 算法维护持久化排名：

| 参数 | 值 | 说明 |
|------|-----|------|
| `TS_MU_0` | 600.0 | 初始平均分 |
| `TS_SIGMA_0` | 200.0 | 初始不确定度 |
| `TS_BETA` | 100.0 | 技能差异半径 |
| `TS_TAU` | 2.0 | 动态因子（允许排名随时间漂移） |
| `TS_DRAW_PROB` | 0.05 | 平局概率 |

**保守分**（Conservative Score）= `mu - 3 * sigma`，排行榜按此排序，确保场次少的 Agent 不因运气好而虚高。

评分数据保存于 `runs/trueskill.json`，每局结束后实时更新，跨次启动持久化。

---

### 2.5 Agent 发现：`discovery.py`

后端无需配置文件，启动时直接扫描 `agents/` 目录树：

```
agents/{bucket}/{name}/main.py  → 有效 Agent
```

- `bucket` 只接受 `baselines / external / mine` 三个值
- 读取同目录 `agent.yaml` 填充元数据（名称、作者、标签等）
- `agent.yaml` 损坏时 fallback 到目录名，并记录 `last_error`

---

### 2.6 SSE 实时进度推送

赛事运行期间，前端通过 **Server-Sent Events** 实时接收进度：

| 事件类型 | 数据 | 含义 |
|----------|------|------|
| `match_result` | `{match_id, winner, scores}` | 单场比赛结束 |
| `run_complete` | `{status, matches_done}` | 整轮赛事结束 |

前端用 `EventSource` 监听，每收到事件自动刷新进度条和结果列表，无需轮询。

---

## 3. 前端详解

### 3.1 技术选型

| 技术 | 用途 |
|------|------|
| **Vite** | 构建工具 + 开发服务器（HMR 热重载） |
| **TypeScript** | 静态类型，与后端 API 接口保持一致 |
| **Vanilla DOM** | 无框架（无 React/Vue），直接操作 DOM |
| **@kaggle-environments/core** | 官方 replay 播放器（React 组件，vendored） |
| **Hash 路由** | `#/path` 形式单页应用路由 |

### 3.2 Hash 路由表

| URL Hash | 对应页面 |
|----------|----------|
| `#/` | Quick Match（快速对战） |
| `#/leaderboard` | TrueSkill 排行榜 |
| `#/tournaments` | 赛事列表 |
| `#/tournaments/:runId` | 赛事详情 |
| `#/replays` | Replay 库 |
| `#/agents` | Agent 列表 |
| `#/agent/:id` | 单个 Agent 详情 |
| `#/submissions` | Kaggle 提交记录 |
| `#/settings` | 设置（Kaggle Token） |
| `#/replay/:runId/:matchId` | 本地 Replay 播放 |
| `#/kreplay/:subId/:epId` | Kaggle Replay 播放 |

### 3.3 API 调用层：`api.ts`

所有接口都有 TypeScript 类型定义，通过统一的 `j<T>()` 函数发起请求：

```typescript
async function j<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return res.json();
}
```

开发模式下 Vite 配置代理，将 `/api/*` 转发到 `:8000`，前端代码无需区分开发与生产环境。

### 3.4 Replay 播放器扩展功能

基于官方 `@kaggle-environments/core` 播放器，本项目新增以下功能：

| 功能 | 说明 |
|------|------|
| 点击星球/舰队 | 侧边栏显示详细数据（船数、产能、在途 ETA、速度） |
| 多选支持 | 同时查看多颗星球的合计力量 |
| 亮/暗主题 | 实时切换，播放器与侧边栏同步应用 |
| 舰队轨迹叠加 | 显示在途舰队的预测飞行路径 |

---

## 4. 开发 vs 生产模式对比

| 维度 | 开发模式 | 生产模式 |
|------|----------|----------|
| 前端 | Vite `:6001`，HMR 热重载 | `pnpm build` → `viewer/dist/` |
| 后端 | uvicorn `:8000`，`--reload` | uvicorn `:8000` |
| 静态文件 | Vite 代理 `/api/*` → `:8000` | FastAPI `StaticFiles("viewer/dist")` |
| 访问地址 | http://localhost:6001 | http://localhost:8000 |

---

## 5. 关键数据文件

| 路径 | 说明 |
|------|------|
| `runs/trueskill.json` | TrueSkill 评分持久化文件 |
| `runs/{run-id}/` | 单次赛事目录，含所有场次 replay JSON |
| `runs/latest` | 软链接，指向最新赛事目录 |
| `agents/{bucket}/{name}/main.py` | Agent 核心逻辑 |
| `agents/{bucket}/{name}/agent.yaml` | Agent 元数据 |

---

## 相关页面

- [开发指南](Development-Guide.md)
- [评估与排行榜](Evaluation.md)
