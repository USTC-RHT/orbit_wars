# JAX Autoregressive Transformer Policy Architecture

This note mirrors the current implementation in `jax_agent/model_jax.py`,
especially `AutoregressiveActorCritic`. The policy first encodes the board with
a Transformer, then uses a GRU decoder to generate a short sequence of commands.

Each command is one of:

```text
END
(source, target, amount)
```

The default maximum sequence length is `max_action_tokens = 12`.

## Inputs

| Tensor | Shape | Meaning |
|---|---:|---|
| `planet_features` | `[B, P, 26]` | Per-planet own features |
| `relative_features` | `[B, P, P + 1, 8]` | Source-target relative features |
| `global_features` | `[B, P, 11]` | Global context, currently read from index `0` |
| `candidate_mask` | `[B, P, P + 1]` | Valid target mask per source |
| `amount_mask` | `[B, P, P + 1, 8]` | Valid amount bucket mask |
| `source_mask` | `[B, P]` | Valid source mask |
| `fleet_features` | `[B, P, F, 28]` | Fleet features |
| `fleet_mask` | `[B, P, F]` | Fleet token mask |
| `map_features` | `[B, P, C, G, G]` | Per-source local map features |

`P = 45`. The `P + 1` target dimension includes the extra no-op/end target
slot.

## Encoder

Planet and fleet representations are encoded as Transformer tokens. Global
features are handled as a separate context path and merged with the `CLS`
output.

```python
# =====================
# Encoder
# =====================
planet_h = MLP(planet_own_features)      # each planet's own features
fleet_h  = MLP(fleet_features_global)    # each fleet/global fleet summary

tokens = [CLS, planet_h..., fleet_h...]
encoded = Transformer(tokens)

cls_out = encoded[CLS]
planet_out = encoded[planet_tokens]

# independent context
global_ctx = MLP(global_features[:, 0])
state_ctx = MLP(concat(cls_out, global_ctx))
relative_h = MLP(relative_features)
```

## Decoder

The decoder is autoregressive. At every step, the previous token embedding and
the state embedding update the GRU hidden state. The model then emits source,
target, and amount logits with conditional heads.

```python
# =====================
# Decoder
# =====================
# h: current GRU decoder hidden state
h = GRU(prev_tok, state_emb)

# src head
src_cnn = cnn(src_map_feature)
src_key_proj = MLP(concat(src_cnn, planet_out))
src_logits = pointer(
    query = MLP(h),
    keys = src_key_proj,
)

# tgt head
src_h = planet_out[src]
relative_h = MLP(relative_features[src, :])
tgt_logits = pointer(
    query = MLP(h + src_h),
    keys = MLP(planet_out) + relative_h,
)

# amount head
tgt_h = planet_out[tgt]
pair_h = relative_h[tgt]
amount_logits = MLP(concat(h, src_h, tgt_h, pair_h))
```

Masks are applied after each head:

- `source_mask` plus remaining ships for source selection.
- `candidate_mask[src]` for target selection.
- `amount_mask[src, tgt]` plus remaining ships for amount selection.

The decoder updates remaining source ships after every executed command, so
later commands in the same turn condition on earlier commands.

## Value Head

The critic reads the encoded global state:

```python
value = MLP(cls_out)
```

The value head is independent from the autoregressive command decoder and is
used by PPO for advantage and value loss computation.
