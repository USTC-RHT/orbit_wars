# Kaggle BC 讨论帖中文翻译

> **所属页面**：[Home](Home.md) > Kaggle BC 讨论帖中文翻译  
> **原帖标题**：Struggling with RL? Something that helped me.  
> **说明**：本文是用户提供的 Kaggle 讨论帖粘贴文本的中文翻译，尽量保留原帖与评论区的顺序。界面按钮、点赞计数等无关 UI 文案已略去。

---

## 原帖：RL 卡住了？有个方法帮到了我

我一开始做 RL 的时候很快撞墙了。我的策略会坍缩，要么一直没有提升，要么就是到处乱撞，基本学不到什么东西。后来有一件事帮了我很多，所以我想分享一下，也许对其他人有用。

在直接上 RL 之前，我先尝试了监督学习。我训练了一个监督模型去模仿好的对局：给定游戏状态，预测应该采取的动作，数据来自强对局的 replay。没有 reward，没有 self-play，也没有探索带来的头疼问题；只是简单地问：“给定这个局面，什么是一个合理的走法？”

诀窍在于你教它什么。不要只是盲目克隆动作。把每局里的每个动作都照搬，会把坏习惯和好习惯一起学进去，最后你继承的是平均玩家，而不是强玩家。你要有意识地选择学习谁的决策，以及你真正信任什么信号。不要把它想成“模仿这局 replay”，而要想成“学习好棋长什么样”。

这件事帮到我的原因有几个：

- 它给了 RL 一个更好的起点。模型不是从随机乱撞开始，而是已经有了还不错的水平，所以 RL 可以在已有的好行为上继续打磨，而不是从零发现这些行为。
- 它是一个快速、稳定的信号。在投入很多天去调一个很敏感的 RL 循环之前，我可以很快判断我的特征、数据和网络是否足以表达不错的玩法。
- 它本身的效果也意外地不错。纯监督学习，还没有加 RL，大约 3 小时的工作就让我排到了 top 10-20 左右。这足以让我相信基础是稳的，然后再在上面叠 RL。

这不一定是最好的方法。它只是让我想通了一些事情，所以分享出来，希望能帮到其他卡在 RL 上的人。

---

## 评论区翻译

### souldrive

这几乎和我在 imitation 方向上的经验完全一致。我用当前 #1 agent 的原始 replay 克隆它，clone 的 move accuracy 一直在涨，但对 greedy Producer 的胜率卡在大约 17%。原因正是你说的，我是在盲目克隆动作，而不是学习好棋长什么样。

#1 agent 跑的是 1-ply search，盲目 clone 只能蒸馏出它的典型动作，蒸馏不出背后的搜索。所以一旦进入一些有情境差异的位置，模型就会偏离分布，然后错误滚雪球。

“有意识地选择学习谁的决策，以及你信任什么信号”正是我缺掉的那个杠杆。如果有用的话，我在这里写了失败分析：`https://www.kaggle.com/code/souldrive/why-cloning-the-1-bot-loses-to-greedy`

### datnt114

感谢分享你的方法。我有个问题：你训练的模型有多少参数？我现在的 RL 模型有 60 万参数，但扩到 500 万参数也没有看到提升。

### vkhydras（原帖作者）

我的是 2500 万参数，但只是因为我有足够多的数据能支撑这个规模。更小的模型没法吃下那么多数据。如果 replay 很少，模型变大只会过拟合。

### Luca

感谢分享，也恭喜你做出了这么强的 imitation learning 结果！你的模型里有没有使用 pairwise / edge planet features，比如距离，或者其他关系信息？

### vkhydras（原帖作者）

有一点。模型确实需要一些关系信息来判断可达性，以及哪些地方正在被争夺，所以它不是纯 per-planet 的。

### CPMP

这就是其他讨论里有人提到的 Behavioral Cloning（BC）。我同意它就是监督学习，只不过 AI 圈的人喜欢给已知技术发明新名字。这样方便发论文。

### YG JH

我最近尝试用 `producer_agent.py` 做 imitation learning。它是 Kaggle 上之前有人分享的一个挺强的 agent，LB 分数大概在 1100-1224。模型看起来学得不错，可以轻松打爆 700-800 分的 baseline agent。

但是当它对上原始的 `producer_agent.py` 时，我的模型被彻底碾压，我不知道该怎么修。另一个问题是性能天花板：因为它是基于一个大约 1200 分 teacher 的 imitation learning agent，感觉它不太可能超过那个水平。虽然我后面也尝试在它上面继续做 RL，但它还是没法打赢 `producer_agent.py`。

我不知道你有没有遇到过同样的问题，很想听听你的看法或建议。

### Billy Bradley

这是 imitation learning 的一个根本限制。它永远无法超过 demonstrator。解决方案是使用 reinforcement learning。

不过你可能在 IL 阶段把 entropy 压得太低了，导致后续 RL 很难有效学习。另外，如果你用的是 actor-critic 方法，你会从一个已经训练过的 policy 开始 RL，但 critic 还是没训练过的。这可能导致 critic 追上来之前，模型先把有用的行为“反学习”掉。

### vkhydras（原帖作者）

关于打不过 teacher 这件事，我觉得原因是：你的 clone 只见过 teacher 自己会进入的那些局面。所以一旦模型自己开始玩，稍微走到一些偏离分布的位置，错误就会滚雪球。因此 clone 基本上总是会比它模仿的对象弱一点。

如果想突破天花板，你需要某种 outcome signal，而不只是“复制这个动作”。否则它永远学不到哪些动作实际上赢了。RL 这块才是我会深入研究的地方。

有一段时间，我把 teacher 当成固定锚点，用 KL penalty 让模型靠近它，或者让模型和一个 frozen copy 做 self-play，但一直卡在它的水平附近。最后真正对我有效的是：让我训练时面对的东西也随着我一起变强，而不是一直冻结不变。

### Navneet

很棒的 RL 支持，@vickimar。

### Wilson

非常感谢你分享方法细节。我也在尝试类似的 supervised / BC pipeline，但目前只从 top-30 比赛里收集到了大约 6700 个 replay，离你提到的约 4.7 万个 replay 还差很远。

想问一下，你收集这么多数据大概花了多久？你是从每天的 episode 里跨多天收集，还是通过 episode graph 有更快的爬取方式？如果能给一点提示，我会非常感激。再次感谢你的分享！

### vkhydras（原帖作者）

我就是直接从 daily episodes datasets 下载的。

### Wilson

抱歉我太粗心了，我找到了，谢谢提醒。

### john inan

谢谢你分享你的过程和经验。我几个月前才发现 Kaggle，Orbit Wars 对我来说是一个很好的老师。一路上我感觉自己用所有可能的方式都失败过。

如果你不介意回答的话，我有几个问题，想弄清楚自己到底哪里做错了。我之前也试过 BC + RL 路线，但像你说的一样进入了平台期，后来又一直在平台和不同方法之间来回跳，每种都尝试了一点，但没有真正打磨任何一种。

当你把每个 turn 按 planet 拆开时，每个 planet 的 head 输出什么？它是分类一个 target planet，还是回归一个连续角度？`hold` 是 target head 里的一个显式 no-op 类别，还是 target 前面有一个单独的 launch/no-launch gate？每个 planet 独立决策时，你怎么保证这一回合整体是协调的，不会所有 planet 都一窝蜂打同一个 target？或者这其实不重要？

你说 features 是 normalized rows，而且比较 basic。它们是纯 per-planet 吗，比如 owner、ships、production、position、distance-to-sun、orbital phase？还是也包括 relational / pairwise 信息，比如 planet-to-planet distance 或 ETA？如果是纯 per-planet，模型怎么判断哪个 target 可达、哪个正在被争夺？

另外，BC 里有没有包含任何 lookahead / simulation 得到的信号？最后，你预先写进去了多少 physics 和 bias？也就是说，哪些东西你没有让网络自己学，比如如何瞄准飞船、角度、避开太阳等等？

### vkhydras（原帖作者）

target head 是分类一个 planet，也就是 over planets 的 pointer，不是回归连续角度。我不让网络去学角度。

`hold` 不是 target head 里的一个类别，而是在前面单独有一个 launch/no-launch gate。每个 planet 先决定是否行动，然后再决定打哪里、派多少。

关于一窝蜂打同一个 target：它们确实是独立决策的，但实践里很少变成问题。模型会从数据里学到协调模式，所以通常不会把所有东西都堆到一个目标上。

剩下的，也就是 features 里到底有什么，我现在先保持模糊，因为这是大部分 edge 所在，而且我还在用它。不过有两个方向性的建议：你说得对，纯 per-planet features 很难判断可达性，所以给模型一些 reach 的概念很重要。另外，不要让网络学习任何可以精确求解的东西。最好把那些东西放在网络外面处理，让网络专注于决策。

### john inan

感谢你。Hold / DoNothing 一直在折磨我。我会试试这个方向。

### Saheb

用 self-play RL 打败 producer / planner agent，有什么建议吗？

我把它们加进 external pool 里，但目前没什么帮助。而且我也不确定把它们加到 external pool 里是不是正确路线。

### vkhydras（原帖作者）

我没有 demonstrator agent。我只是从 daily episode data 里学习，然后用 self-play RL，也就是 lagged-self，在 plateau 前继续提升它。

### Justice Ackermann

你是怎么做 preprocess 的？又是怎么处理一个 action 里的多个 subactions 的？我做 supervised learning 的时候，agent 一直很难学会 do nothing。

### vkhydras（原帖作者）

Preprocessing 就是把 board 转成 normalized feature rows。

对于 subactions，我把一个 turn 按每个 planet 拆开，而不是把它当成一个巨大的 joint action。每个 planet 都做一个小决策。

关于 do nothing 的问题：不发射是目前最常见的 label，所以 head 会坍缩成永远 no-op。重新平衡数据，让 idle cases 不要淹没真正的 launches，就能修复这个问题。

### Justice Ackermann

很好，谢谢你。

### azxav

非常感谢你分享经验。我真的很感激。我也尝试了和你类似的方法，大概做了一周。我的情况是，我用 2500 个 replay 训练了一个基础 supervised model。但是之后用 PFSP 做 RL，大概 200 万步后遇到了 policy collapse，模型只会选择 no-operation。

现在我正在尝试修改 reward policy 和其他一些小调整。我用的是非常简单的 reward policy：2P 是 -/+1，4P 是 -/+0.5/1。

我想问问你的 reward policy 是什么？你在 RL 训练中使用了什么方法？有没有用 heuristic models 作为对手？

另外我很好奇你的 features。你的 features 是比较 basic，还是更复杂？我的 features 很复杂，包括带 ETA 的 pair features，所以训练很慢。我想知道你的经验里，这些复杂 features 是否真的有帮助，还是用更简单的 features 训练更久就够了？

### vkhydras（原帖作者）

RL 是我还没有真正搞定的部分。我其实没有太多 RL 经验，所以让我冲上榜的大部分东西来自 supervised model。我的 RL 一开始确实提升了一点，但那里面很多也是运气，后来我也撞到了同样的墙。不过那一点 RL 确实帮 supervised model 提升到了大约第 7-10 名。

我的 reward 也很简单，基本就是 +/-1。对手主要是我自己更早的 checkpoints，不是 heuristic bots。我的 features 也相当 basic。

### azxav

你的 RL 训练了多少 steps？

### 原文折叠回复

原文此处显示还有 3 条回复，但用户提供的粘贴文本里没有展开具体内容。

### sue124

感谢你分享方法。当你用纯 supervised 达到 top 10-20 时，神经网络是完整 policy 吗？也就是 action head 直接输出所有 launches？还是它被用作 heuristic search 上面的 prior / feature，比如像 “The Producer” agent 那样？

我想弄清楚，纯 NN-driven policy 自己是否就能达到那里，还是仍然有 heuristic planner 在做主要工作。

---

## 译文中的关键词对照

| 英文 | 译法 |
|------|------|
| Behavioral Cloning / BC | 行为克隆 / BC |
| supervised learning | 监督学习 |
| imitation learning / IL | 模仿学习 / IL |
| demonstrator / teacher | 示范者 / teacher |
| replay | 对局回放 / replay |
| policy collapse | 策略坍缩 |
| no-op / do nothing / hold | 不行动 / 不发兵 / hold |
| launch gate | 发兵门控 |
| target head | 目标头 |
| amount head | 派兵量头 |
| pointer over planets | 在星球集合上的 pointer 分类 |
| off-distribution | 偏离训练分布 |
| lagged-self | 滞后自我对弈 / 历史自身 checkpoint |
