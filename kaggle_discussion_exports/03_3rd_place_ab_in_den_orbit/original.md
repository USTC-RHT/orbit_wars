[3rd Place] Ab in den Orbit

Self-Play RL with PPO, transformers and JAX environment/feature engineering

Orbit Wars

Solution Writeup · 3rd place · Jul 8, 2026

TL;DR

Orbit Wars is a real-time strategy game where players conquer a solar system in 2-player or 4-player mode. They launch ships from their starting planets to capture new worlds, which in turn produce more ships. The winner is whoever has the most ships at the end of the game (or eliminates all opponents by capturing their planets).

Approach: pure self-play RL (PPO + PFSP), no imitation learning. I rewrote the full environment and feature engineering in JAX, and the model itself is a 6.2M-parameter transformer in Torch.
Key idea: the 'reachability tensor' - an object storing, for every source/target planet pair, how long a fleet of a given size would take to travel (and whether it can reach at all). Computing it efficiently was almost as much work as the model itself.
Biggest levers: the action representation (semantic actions like hold / sortie / kill-at-arrival instead of raw ship fractions), a suitable action head design and the entropy schedule - the single most important training knob for me.
Introduction

Hi Everyone! This was a great competition. The combination of long term planning and continuous action space made for an extremely interesting challenge. Thanks to the organisers @bovard et al. for such a thoughtful design. While not my first simulation competition, it was my very first attempt at RL and it was pretty cool to compete with a lot of the legends from the writeups I studied during the course of the competition, so shoutout to @ferdinandlimburg, @pressman1 and @tonyk98! I also got early (technical) validation from seeing lots of parallels to my own early attempts in the competition, when not a lot was working on my side yet. Super helpful, @lightmk.

Architecture

My architecture is a standard pre-norm transformer with a custom encoder module and action head. It processes 48 tokens per forward pass, 4 player tokens and 44 planet tokens, the last of which are reserved for the comets. The model has 6.2M params. 8 transformer layers form the trunk. On top, the critic consists of 2 transformer layers followed by a small MLP. Only the player tokens pass through the critic, giving a per-player value baseline (one value per player token, so the same forward pass produces the ego value and the opponents' values). The actor head also consists of 2 transformer layers that only process the planet tokens, followed by the action head, which is a bit more involved. I used dim=192 and an expansion factor of 4.

For most of the competition, I had 4 actions, sending fixed fractions of 0.25, 0.5, 0.75 and 1.0 of a planet's garrison. In the last week I switched to semantic actions: each action encodes an intent per (source, target) pair, and the actual ship count is then computed from the reachability tensor to satisfy that intent, rather than being a fixed fraction. I did this after watching lots of @jakewwill 's games 🙂 This increased learning speed by a lot (defined as how many training steps are needed to beat certain reference agents). The four actions were:

Send all - launch the entire garrison.
Sortie - send away as much as possible without losing this planet to fleets already in flight against it.
Hold - send exactly enough to conquer the target and hold it against all already-launched fleets for 8 rounds.
Kill at arrival - send exactly enough to conquer the target on arrival, accounting for all already-launched fleets.
Reachability tensor

I designed the architecture and features mostly around the 'reachability tensor', a tensor of shape (B, P, P, S, 3): for every (source planet, target planet) pair and every one of the S actions (send-all / sortie / hold / kill-at-arrival), it stores the three numbers that fully define a launch - the number of ships, the launch angle and the arrival time. It is used for features, the action head, the semantic ship counts and action masking, so almost everything downstream reads from it. I introduce it up front because the rest of the writeup keeps referring back to it.

The hard part was computing it efficiently every turn. I implemented the environment and the feature engineering in JAX. The latter caused massive headaches, since this meant (a) vectorized collision checks (very complex) and (b) long compile times, which scale with the size of the XLA output. Especially with 4p, some games time out due to compilation overhead, and checking longer time horizons for collisions - thereby allowing longer flights - increases the compilation time enormously. In the future I would probably do this part in Rust or C++: while JIT is great for speed, compilation on the Kaggle servers just seems risky, since you can't control or budget for it within the per-turn time limit.

Still, it was worth obsessing over. Most of the code was written by Claude, but I kept coming back to it conceptually a lot, since finding better aiming angles and reaching that one high-production planet 1 tick earlier made a real difference. Same with the horizon of the collision check: on some maps, not being able to fly longer than e.g. 16 turns simply means losing. On the other hand, optimising this kernel and going through different ways of finding the launch angles with Newton-Raphson or simple fixed-point iteration was also a lot of fun.

One of the early challenges I faced was estimating the quality of this tensor. I settled on a forward simulation that gives me, averaged over 24 seeds, the rate of ships arriving at their target plus the total number of available routes. With the arrival rate pinned at 100% when no comets are in play, maximizing the number of routes makes a great objective for an autoresearch-style agent loop.

Feature Engineering

Features are processed in p0 perspective. In order to help the transformer with the math, I decided to go for embeddings rather than scalar features for the garrisons and fleet sizes, which were exact up to 384 ships and then binned with sqrt-sized bins up to 768 (anything above is clamped into the top bin). I did not run rigorous ablations on the different embeddings. Early on, the model kept failing to capture planets by a slim margin, and given the general notion that transformers are 'not great' at math, I decided to invest more in this crucial representation. For the incoming fleets I built a small grid ('arrival calendar') with one row per player holding the incoming ship counts, plus the post-combat survivors and the resulting owner. It spans the full horizon (24 in 2p and 16 in 4p mode) in the time dimension (columns). Every cell in this calendar is embedded and projected to obtain a scalar weight, which is then (after normalization) used to compute a weighted sum of all cells, and then concatenated to the other features after another projection. Here @simjeg's solution feels much more elegant, kudos! I added further features in graphormer style that capture information about which planets ships can be sent to, how long it takes, etc.

(Tables created after the nice example of @yijiey, read his write-up, it's excellent)

Tokens are afterwards projected to the hidden dim of 192.

FiLM for global context. The genuinely global scalars (step fraction, time until the next comet wave and how long the current one lasts, …) matter for every token, so I feed them in twice: once broadcast into the token features above, and once as a separate FiLM conditioning vector. The latter drives a small MLP that produces per-block scale/shift parameters applied inside every transformer block, letting the whole trunk modulate its behavior based on the game phase and map without spending token capacity on it.

Graph structure of the problem

The reachability tensor defines a graph, that I wanted to capture with the transformer (which by design also models a fully connected graph). Node features are easy - they just become token features. But getting information about the edges into a plain transformer is not that easy, as it turns out: standard attention only sees the two token embeddings, not the relation between them (how long a fleet takes between them, at what angle, whether the trip is even possible). I settled on graphormer-style attention biases: for each (source, target) pair I derive a handful of pairwise features from the reachability tensor and project them into additive biases on the attention logits. This lets the otherwise fully-connected attention be modulated by the actual reachability graph, so a planet can "attend" more strongly to the targets it can genuinely reach.

Action head

The action space is per source planet: each owned planet independently emits one categorical over {no-op} ∪ {target × action} - i.e. either do nothing, or pick one of the 44 targets combined with one of the 4 semantic actions. So per planet the head produces a 177-dim distribution (1 no-op + 44 × 4 launch logits), where the 4 are the semantic actions above. I compute these with two heads, a no-op head and a launch head.

For the launch head, I combined three additive terms before passing them through a 176-dim (44 planets x 4 actions) Linear layer for the logits:

a simple launch-count aware bilinear, computed as dot-product between source and target planet, where I added the number of ships to be launched to the source token. This term is by definition factorizable (it decomposes into a per-source times per-target contribution)
a full-rank edge term: a small MLP evaluated per (source, target, action) edge that injects exactly the edge-specific information the bilinear cannot recover from the two independent tokens - the per-action ETA (a pairwise quantity present in neither token), the authoritative per-edge ship count / kill-at-arrival capture cost, and the target's garrison embedding. Dense, this hidden state would be a big BPPSH intermediate (P=44, S=4, H=16), but only a small fraction of (source, target, action) edges are actually reachable. So I gather onto just the reachable edges, run the MLP on an (N, H) tensor for the N reachable edges, and scatter the resulting logits back - identical result, far cheaper. The gather/scatter breaks torch.compile, but the backward pass gets much faster.
a bias for each of the semantic actions

In the end I simply concatenated the no-op logit with the launch logits for the 177-dim categorical per source planet.

Training

I used standard PPO with GAE and pure self-play. To scale to more machines, I implemented DDP, but only ever used 2 GPUs. For 2p and 4p, I trained separate models, although the 4p was also trained on some 2p games. In both cases, I relied on terminal rewards only, although I experimented with reward shaping during the very early training phases. The terminal reward is winner-take-all: at the end of the game the player with the highest ship score gets +1 and everyone else gets -1 (-1/3 for 4p to keep rewards zero sum). Regarding hardware, I ran early experiments on my 3090, switching later to a 5090 and, after a mysterious GPU crunch on vast.ai at the end of May, to RTX 6000 Pros. On two RTX 6000 I achieved about 19k SPS (all included) for 2p and 15k for 4p, which I am quite happy with given the model size. The submitted checkpoints trained around 8.4B steps for 2p and 2.7B for 4p.

My hyperparameters were quite standard:

Hyperparameter	Value
GAE lambda	0.95
Gamma	0.993 (2p), 0.99 (4p)
lr	adaptive (KL-targeted - a controller raises/lowers the LR to keep the mean approx-KL near a target budget instead of following a fixed decay)
Rollout len	256
ppo epochs	1 (a single pass over each rollout; with fresh self-play data every update, more epochs mostly bought instability rather than sample efficiency)
mini batch size	8192
num_envs	1024
clip	0.2
Entropy

Entropy was by far the most important knob during training and I spent a lot of time trying out different entropy annealing schedules. Since the categoricals are so simple, I did not have to resort to any tricks here: simply annealing the entropy was enough for the agent to stop spamming lots of tiny fleets on its own. This works worse for 4p, where the entropy seemingly needs to stay high.

Figure 1: Entropy coefficient annealing schedule during the final 2p training run.

PFSP

The agent played exclusively against itself or past checkpoints of the same architecture. To accelerate learning I implemented PFSP (Prioritized Fictitious Self-Play - sampling past opponents in proportion to how hard they are to beat, rather than uniformly), but noticed early on that I would spend a lot of time estimating win rates during eval which are needed to prioritize strong opponent checkpoints. Switching to simply taking winrates from the rollouts introduced a subtle bug, since games are usually not played completely by the same agent when they run longer than the rollout window. I solved this by fixing the chosen opponent for 2 consecutive PPO updates (for a total of 512 steps in my case), after which I would abandon all open games. Then I would resample a new opponent that plays all envs. This gives free winrate estimations from rollouts against different checkpoints, but comes at the cost of fewer terminal rewards emitted by the environment. I would be curious to hear how others solved this.

Evaluation

During training I would periodically run against prior checkpoints and earlier models I had trained. I ran 1024 eval games in parallel. For 2p, that was no problem and I saw a near monotonically increasing win rate. 4p was harder. I evaluated the agents in a 1v3 setup, where the 3 opponents were the same agent. In hindsight I think that introducing more variety into 4p games by selecting different checkpoints for the same games would have been an obvious improvement. I settled on a T=0.1 sampling temperature, the same I use in my 4p submission.

Winrate against my then strongest 2p model during the final training run.

Winrates against my then strongest 4p model during the final phase of the competition. The jump around 1.8B occured after a critical bug fix. I stitched together multiple runs, run multiple hyperparameters in parallel and then followed up on the best ones. I only show the runs that lead to my ultimate submissions, but there are at least three times as many runs that either plateaued or crashed to 0.

Optimisation tricks

Optimising throughput was one of the most fun parts of the competition. My two favorite tricks were:

action head sparsity, already mentioned above, speeds up the backward pass
topk is actually a pretty slow operation (relevant for combat resolution). As it turns out, max -> mask -> max -> mask is a lot faster for k=2.
Supporting Infrastructure

During the course of the competition, I wrote quite a bit of surrounding infrastructure I want to briefly mention.

Checkpointing different code versions

I produced plenty of versions of models, features and training procedures. Especially the feature engineering part that happened in JAX changed from time to time, and while the contracts between the components (mostly) stayed stable, faithful evaluation required snapshotting the whole pipeline. So I kept a zoo/ folder in my repository next to a /dev folder, which allowed me to eval against older versions with all their quirks and bugs.

Resumability

Like many other contestants I rented machines on vast.ai . While in theory stable, these can just crash and I would estimate the probability this happens during a multi day span to be on the order of 5-10% . Nothing you want to happen during your final training run, so I implemented a cloud checkpointing system that contained not only the weights and optimizer states of the learner, but also the full pool of past checkpoints with their win rates and weights needed for selection. While a safety measure, this indeed came in handy multiple times.

Impact of coding agents

Like many other competitors, I wanted to use the competition for extensive agentic development. It is incredible how much work would simply not have happened without them. I relied on a Cursor Ultra subscription during the comp (old fashioned, I know, but I still look at the code from time to time 😄), that I also maxed out. It has been said before, but I have the feeling that competing at the top nowadays increasingly requires more and more resources, both for GPUs and coding agents. That being said, other contestants were much more resource-efficient, which I think deserves special appreciation, esp @sinkingpoint and @ferdinandlimburg.

I relied on no library to maximise learning (or an excuse for maximum not-invented-here syndrome, which the agents amplify a lot). One downside I noticed repeatedly - especially in RL, where rigorous ablations are hard to come by due to long training times and high run variance - is that it has become extremely easy to add bells and whistles faster than you can understand their implications. This led to a crucial bug I only unearthed 36 hours before the competition ended that severely crippled my 4p performance and ruined a few nights of sleep…

General Remarks

A few things I took away from this competition:

Self-play RL is brutally noisy. With long convergence times and high run-to-run variance, you can realistically only test whether something speeds up learning, not whether it changes the final ceiling. The effect of choices like hidden dim or model depth is thus very hard to establish rigorously, which is worth keeping in mind before reading too much into any single change (including some of mine above).
Things I'd do differently. Introduce real opponent variety into 4p evaluation and training (either different checkpoints in the same game or different sampling temperatures) instead of the 1v3-against-a-clone setup.
Start early. Schedule your final runs early, then start them a few days before that if you can manage.

Thanks for reading, and thanks to everyone who shared ideas and writeups along the way. Hope to see you in the next one!
