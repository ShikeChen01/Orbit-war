"""v5 target actor-critic policy + the act/evaluate wrappers (notebook cell 21).

Per-planet trunk (legacy attn+ResNet-MLP, or a stacked transformer) -> WHERE/GATE heads producing
the gated-allocation action, plus a masked-mean-pooled value head with PopArt. The runtime knobs
that were notebook globals (``DEST_HEAD``/``INIT_DEST_SCALE``/``GRAD_CHECKPOINT`` for the net;
``WHERE_DIST``/``REACH_MASK``/``ALLOC_KAPPA``/``USE_AMP``/``USE_POPART`` for the wrappers) are read
off the :class:`~orbit_wars_v12.config.Config` passed in.
"""
import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _ckpt

from .constants import BOARD_SIZE, CENTER, DTYPE, F_DIM, G_DIM, PLANET_CAP, SUN_RADIUS
from .distributions import GatedAllocDist, GatedCatDist


def _amp_ctx(cfg):
    if cfg.USE_AMP and cfg.device.type == 'cuda':
        return torch.autocast('cuda', dtype=cfg.AMP_DTYPE)
    return contextlib.nullcontext()


class PopArt:
    '''Output-preserving running normalization of value targets (Hessel et al. 2018). The value head
    predicts NORMALIZED values; raw V = denormalize(y); on each stats update val_out is rescaled so
    the raw predictions are preserved.'''
    def __init__(self, beta):
        self.mu = 0.0; self.nu = 1.0; self.sigma = 1.0; self.beta = beta; self.init = False

    def normalize(self, y):
        return (y - self.mu) / self.sigma

    def denormalize(self, yn):
        return yn * self.sigma + self.mu

    def update(self, targets, val_out):
        with torch.no_grad():
            old_mu, old_sigma = self.mu, self.sigma
            m = targets.mean().item(); v = (targets * targets).mean().item()
            if not self.init:
                self.mu, self.nu, self.init = m, v, True
            else:
                b = self.beta
                self.mu = (1 - b) * self.mu + b * m
                self.nu = (1 - b) * self.nu + b * v
            self.sigma = max((self.nu - self.mu * self.mu) ** 0.5, 1e-4)
            if old_sigma > 0:                       # preserve raw outputs of the value head
                val_out.weight.mul_(old_sigma / self.sigma)
                val_out.bias.copy_((old_sigma * val_out.bias + old_mu - self.mu) / self.sigma)


def _resolve_act(name):
    """Map an ACT_FN name to a pointwise activation fn (default relu = the historical block act)."""
    return {"relu": torch.relu, "tanh": torch.tanh, "gelu": F.gelu, "silu": F.silu}.get(name, torch.relu)


class TxEncoderLayer(nn.Module):
    '''Pre-LN transformer encoder block: x = x + MHA(LN(x)); x = x + MLP(LN(x)).
    Multi-head; masks dead-planet KEYS (matches the legacy single-head attention).'''
    def __init__(self, h, n_heads, mlp_ratio, act='relu'):
        super().__init__()
        assert h % n_heads == 0, "HIDDEN (%d) not divisible by N_HEADS (%d)" % (h, n_heads)
        self.h, self.nh, self.hd = h, n_heads, h // n_heads
        self.act = _resolve_act(act)
        self.ln1 = nn.LayerNorm(h)
        self.q = nn.Linear(h, h); self.k = nn.Linear(h, h); self.v = nn.Linear(h, h); self.o = nn.Linear(h, h)
        self.ln2 = nn.LayerNorm(h)
        self.mlp_in = nn.Linear(h, mlp_ratio * h); self.mlp_out = nn.Linear(mlp_ratio * h, h)

    def forward(self, x, entity_mask):
        B, E, _ = x.shape
        xn = self.ln1(x)
        q = self.q(xn).view(B, E, self.nh, self.hd).transpose(1, 2)        # (B,nh,E,hd)
        k = self.k(xn).view(B, E, self.nh, self.hd).transpose(1, 2)
        v = self.v(xn).view(B, E, self.nh, self.hd).transpose(1, 2)
        attn_mask = (entity_mask < 0.5).view(B, 1, 1, E).to(q.dtype) * (-1e9)   # additive key mask
        ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)      # flash/mem-efficient (B,nh,E,hd)
        ctx = ctx.transpose(1, 2).reshape(B, E, self.h)
        x = x + self.o(ctx)
        x = x + self.mlp_out(self.act(self.mlp_in(self.ln2(x))))
        return x


class TransformerTrunk(nn.Module):
    '''proj -> n_stem pre-LN ResNet-MLP -> n_layers transformer blocks -> n_head pre-LN
    ResNet-MLP -> LN. Emits per-planet features (B,E,h); drop-in for the legacy trunk.'''
    def __init__(self, F, h, n_heads, n_layers, mlp_ratio, n_stem, n_head, act='relu'):
        super().__init__()
        self.act = _resolve_act(act)
        self.proj = nn.Linear(F, h)
        self.stem_a = nn.ModuleList([nn.Linear(h, h) for _ in range(n_stem)])
        self.stem_b = nn.ModuleList([nn.Linear(h, h) for _ in range(n_stem)])
        self.stem_ln = nn.ModuleList([nn.LayerNorm(h) for _ in range(n_stem)])
        self.layers = nn.ModuleList([TxEncoderLayer(h, n_heads, mlp_ratio, act) for _ in range(n_layers)])
        self.head_a = nn.ModuleList([nn.Linear(h, h) for _ in range(n_head)])
        self.head_b = nn.ModuleList([nn.Linear(h, h) for _ in range(n_head)])
        self.head_ln = nn.ModuleList([nn.LayerNorm(h) for _ in range(n_head)])
        self.ln_out = nn.LayerNorm(h)

    def forward(self, entities, entity_mask):
        tok = self.proj(entities)
        for i in range(len(self.stem_a)):
            tok = tok + self.stem_b[i](self.act(self.stem_a[i](self.stem_ln[i](tok))))
        for layer in self.layers:
            tok = layer(tok, entity_mask)
        for i in range(len(self.head_a)):
            tok = tok + self.head_b[i](self.act(self.head_a[i](self.head_ln[i](tok))))
        return self.ln_out(tok)


def parse_trunk_spec(spec):
    """'res16,attn1,res64' -> [('res',16),('attn',1),('res',64)]. Block kinds: 'res' (pre-LN
    ResNet-MLP) and 'attn' (PURE cross-planet multi-head self-attention, no MLP)."""
    out = []
    for tok in str(spec).replace(" ", "").split(","):
        if not tok:
            continue
        i = 0
        while i < len(tok) and tok[i].isalpha():
            i += 1
        kind, n = tok[:i], int(tok[i:])
        assert kind in ("res", "attn", "seq"), "unknown block %r in TRUNK_SPEC (use res/attn/seq)" % kind
        out.append((kind, n))
    return out


class ResMLPBlock(nn.Module):
    '''One pre-LN ResNet-MLP block: x = x + W2(act(W1(LN(x)))). entity_mask arg ignored (uniform call).'''
    def __init__(self, h, act='relu'):
        super().__init__()
        self.ln = nn.LayerNorm(h); self.a = nn.Linear(h, h); self.b = nn.Linear(h, h)
        self.act = _resolve_act(act)

    def forward(self, x, entity_mask=None):
        return x + self.b(self.act(self.a(self.ln(x))))


class CrossAttnBlock(nn.Module):
    '''PURE cross-planet multi-head self-attention (pre-LN, residual, NO MLP). Masks dead-planet keys.'''
    def __init__(self, h, n_heads):
        super().__init__()
        assert h % n_heads == 0, "HIDDEN (%d) not divisible by N_HEADS (%d)" % (h, n_heads)
        self.h, self.nh, self.hd = h, n_heads, h // n_heads
        self.ln = nn.LayerNorm(h)
        self.q = nn.Linear(h, h); self.k = nn.Linear(h, h); self.v = nn.Linear(h, h); self.o = nn.Linear(h, h)

    def forward(self, x, entity_mask):
        B, E, _ = x.shape
        xn = self.ln(x)
        q = self.q(xn).view(B, E, self.nh, self.hd).transpose(1, 2)        # (B,nh,E,hd)
        k = self.k(xn).view(B, E, self.nh, self.hd).transpose(1, 2)
        v = self.v(xn).view(B, E, self.nh, self.hd).transpose(1, 2)
        am = (entity_mask < 0.5).view(B, 1, 1, E).to(q.dtype) * (-1e9)     # additive key mask
        ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        ctx = ctx.transpose(1, 2).reshape(B, E, self.h)
        return x + self.o(ctx)


class BlockSeqTrunk(nn.Module):
    '''proj -> an arbitrary ordered SEQUENCE of blocks (parsed from TRUNK_SPEC) -> LN. Block kinds:
    'res' = pre-LN ResNet-MLP; 'attn' = PURE cross-planet MHA (no MLP); 'seq' = full transformer
    encoder layer (MHA + MLP, TX_MLP_RATIO). Emits per-planet features (B,E,h); drop-in trunk.'''
    def __init__(self, F, h, spec, n_heads, act='relu', grad_checkpoint=False, mlp_ratio=4):
        super().__init__()
        self.proj = nn.Linear(F, h)
        self.grad_checkpoint = bool(grad_checkpoint)
        blocks = []
        for kind, count in spec:
            for _ in range(count):
                if kind == 'res':
                    blocks.append(ResMLPBlock(h, act))
                elif kind == 'attn':
                    blocks.append(CrossAttnBlock(h, n_heads))
                else:                                   # 'seq' = transformer encoder layer (MHA + MLP)
                    blocks.append(TxEncoderLayer(h, n_heads, mlp_ratio, act))
        self.blocks = nn.ModuleList(blocks)
        self.ln_out = nn.LayerNorm(h)

    def forward(self, entities, entity_mask):
        tok = self.proj(entities)
        for blk in self.blocks:
            if self.grad_checkpoint and tok.requires_grad:
                tok = _ckpt.checkpoint(blk, tok, entity_mask, use_reentrant=False)
            else:
                tok = blk(tok, entity_mask)
        return self.ln_out(tok)


class PolicyNet(nn.Module):
    '''v5 TARGET actor-critic (mirrors model/policy_net.cpp, target_actor branch).

    Trunk (UNCHANGED from the continuous actor): per-planet proj + masked cross-planet
    self-attention + GLU + n residual blocks + a separate board-globals embedding.
    Heads: dest_head -> (B,E,E) per-source WHERE allocation logits (-> Dirichlet rows over the
    E=PLANET_CAP dests; dest_head out dim == max_entities); gate_head -> (B,E,1) per-source launch
    gate logit. See GatedAllocDist for how WHERE x GATE compose into the (B,E,E+1) action.

    `cfg` supplies DEST_HEAD / INIT_DEST_SCALE / GRAD_CHECKPOINT (read once here); the remaining
    dims are explicit so league snapshots / invited members can rebuild at their own sizes.
    '''
    def __init__(self, cfg, F, G, hidden, d_g, max_entities, n_res_blocks, use_glu, init_gate_bias,
                 use_attention=True,
                 value_res_blocks=2, arch="trunk", n_heads=8, n_tx_layers=4, tx_mlp_ratio=4,
                 n_stem_res=1, n_head_res=1, act='relu', aux_reward_pred=False, aux_win_bet=False,
                 trunk_spec=""):
        super().__init__()
        self.F, self.G, self.h, self.d_g, self.E = F, G, hidden, d_g, max_entities
        self.use_glu = use_glu
        self.use_attention = use_attention
        self.arch = arch
        self.grad_checkpoint = bool(cfg.GRAD_CHECKPOINT)   # read once (avoids cfg on the compiled module)
        self.act_fn = _resolve_act(act)   # ResNet-MLP block activation (g_embed below stays relu)
        h = hidden
        if arch == "transformer":
            self.trunk = TransformerTrunk(F, h, n_heads, n_tx_layers, tx_mlp_ratio, n_stem_res, n_head_res, act)
        elif arch == "blockseq":                            # ordered res/attn block sequence from TRUNK_SPEC
            self.trunk = BlockSeqTrunk(F, h, parse_trunk_spec(trunk_spec), n_heads, act,
                                       self.grad_checkpoint, tx_mlp_ratio)
        else:
            self.proj = nn.Linear(F, h)
            if use_attention:                               # cross-planet self-attention (toggleable)
                self.attn_q = nn.Linear(h, h); self.attn_k = nn.Linear(h, h)
                self.attn_v = nn.Linear(h, h); self.attn_o = nn.Linear(h, h)
                self.ln_attn = nn.LayerNorm(h)
            self.ln_out = nn.LayerNorm(h)
            if use_glu:
                self.glu_gate = nn.Linear(h, h); self.glu_val = nn.Linear(h, h); self.glu_out = nn.Linear(h, h)
            self.res_a = nn.ModuleList([nn.Linear(h, h) for _ in range(n_res_blocks)])
            self.res_b = nn.ModuleList([nn.Linear(h, h) for _ in range(n_res_blocks)])
            self.ln_res = nn.ModuleList([nn.LayerNorm(h) for _ in range(n_res_blocks)])
        self.g_embed = nn.Linear(G, d_g)
        # --- gated allocation heads: WHERE (per-source dest simplex) + GATE (per-source fire logit) ---
        self.dest_mode = cfg.DEST_HEAD
        if cfg.DEST_HEAD == 'bilinear':                    # pointer score: query=source, key=dest
            self.dest_q = nn.Linear(h + d_g, h); self.dest_k = nn.Linear(h + d_g, h)
        else:
            self.dest_head = nn.Linear(h + d_g, max_entities)  # (B,E,E) per-source WHERE allocation logits
        self.gate_head = nn.Linear(h + d_g, 1)             # (B,E,1) per-source launch-gate logit
        # value head (PPO critic): input proj -> pre-LN RESIDUAL MLP (same block as the trunk) -> readout
        self.val_in = nn.Linear(h + d_g, h)
        self.val_res_a = nn.ModuleList([nn.Linear(h, h) for _ in range(value_res_blocks)])
        self.val_res_b = nn.ModuleList([nn.Linear(h, h) for _ in range(value_res_blocks)])
        self.val_ln = nn.ModuleList([nn.LayerNorm(h) for _ in range(value_res_blocks)])
        self.val_out = nn.Linear(h, 1)
        # auxiliary reward-prediction head (UNREAL): a SHALLOW head off the same pooled trunk features
        # regresses the immediate per-step reward r_t. Deliberately shallow -- the point is to pressure
        # the TRUNK, not to fit r_t with a deep head. Built only when enabled, so the state_dict (and every
        # existing checkpoint) is unchanged when off; this flag is persisted in the ckpt config blob.
        self.aux_reward_pred = bool(aux_reward_pred)
        self.aux_win_bet = bool(aux_win_bet)
        if self.aux_reward_pred or self.aux_win_bet:      # ONE shallow scalar head, two possible objectives:
            self.rew_in = nn.Linear(h + d_g, h)           #   reward-pred (raw -> MSE to r_t) OR
            self.rew_out = nn.Linear(h, 1)                #   win-bet (raw logit -> bet=tanh, maximize bet*outcome)
        # calm init: few planets fire at start (negative gate bias); WHERE small-init -> ~uniform
        with torch.no_grad():
            self.gate_head.bias.fill_(init_gate_bias)
            if cfg.DEST_HEAD == 'bilinear':
                self.dest_q.weight.mul_(cfg.INIT_DEST_SCALE); self.dest_q.bias.zero_()
                self.dest_k.weight.mul_(cfg.INIT_DEST_SCALE); self.dest_k.bias.zero_()
            else:
                self.dest_head.weight.mul_(cfg.INIT_DEST_SCALE); self.dest_head.bias.zero_()

    def _res_block(self, i, tok):                  # one pre-LN residual MLP block (trunk)
        return tok + self.res_b[i](self.act_fn(self.res_a[i](self.ln_res[i](tok))))

    def forward(self, entities, entity_mask, action_mask, globals_):
        if self.arch in ("transformer", "blockseq"):
            tok = self.trunk(entities, entity_mask)
        else:
            tok = self.proj(entities)                       # (B,E,d)
            # masked single-head self-attention over planets (pre-norm) -- the only cross-planet mixing
            if self.use_attention:
                xn = self.ln_attn(tok)
                Q = self.attn_q(xn); Kt = self.attn_k(xn); Vt = self.attn_v(xn)
                scores = torch.matmul(Q, Kt.transpose(1, 2)) / math.sqrt(self.h)   # (B,E,E)
                dead_key = (entity_mask < 0.5).unsqueeze(1)     # (B,1,E)
                scores = scores.masked_fill(dead_key, -1e9)
                attn = torch.matmul(torch.softmax(scores, -1), Vt)
                tok = tok + self.attn_o(attn)
            if self.use_glu:
                tok = tok + self.glu_out(self.glu_val(tok) * torch.sigmoid(self.glu_gate(tok)))
            for i in range(len(self.res_a)):
                if self.grad_checkpoint and tok.requires_grad:   # recompute in backward -> deep stack fits VRAM
                    tok = _ckpt.checkpoint(self._res_block, i, tok, use_reentrant=False)
                else:                                       # rollout / eval (no_grad): identical to before
                    tok = self._res_block(i, tok)
            tok = self.ln_out(tok)
        B, E = tok.shape[0], tok.shape[1]
        gp = torch.relu(self.g_embed(globals_))         # (B,d_g)
        gpb = gp.unsqueeze(1).expand(B, E, self.d_g)
        hh = torch.cat([tok, gpb], -1)                  # (B,E,d+d_g)
        # --- v5 target heads ---
        if self.dest_mode == 'bilinear':               # pointer score q.k / sqrt(h)
            qd = self.dest_q(hh); kd = self.dest_k(hh)
            dest_logits = torch.einsum('bsh,bdh->bsd', qd, kd) / math.sqrt(self.h)
        else:
            dest_logits = self.dest_head(hh)           # (B,E,E) per-source WHERE allocation logits
        gate_logits = self.gate_head(hh).squeeze(-1)   # (B,E) per-source launch-gate logit
        # value: masked-mean pool of trunk tokens over live planets + g'
        em = entity_mask.unsqueeze(-1)
        pooled = (tok * em).sum(1) / em.sum(1).clamp_min(1.0)
        vh = self.val_in(torch.cat([pooled, gp], -1))                  # (B,d)
        for i in range(len(self.val_res_a)):                           # residual blocks (skip per block)
            vh = vh + self.val_res_b[i](self.act_fn(self.val_res_a[i](self.val_ln[i](vh))))
        value = self.val_out(vh).squeeze(-1)                           # (B,)
        reward_pred = None                                             # aux scalar head off the SAME pooled features:
        if self.aux_reward_pred or self.aux_win_bet:                   # raw output -> MSE to r_t (reward-pred), or the
            reward_pred = self.rew_out(self.act_fn(self.rew_in(torch.cat([pooled, gp], -1)))).squeeze(-1)  # bet logit
        return dest_logits, gate_logits, value, reward_pred            # (win-bet: bet = tanh(reward_pred) in the loss)


def build_policy(cfg):
    _net = PolicyNet(cfg, F_DIM, G_DIM, cfg.HIDDEN, cfg.D_G, PLANET_CAP, cfg.N_RES_BLOCKS,
                     cfg.USE_GLU, cfg.INIT_GATE_BIAS, use_attention=cfg.USE_ATTENTION, act=cfg.ACT_FN,
                     value_res_blocks=cfg.VALUE_RES_BLOCKS,
                     arch=cfg.ARCH, n_heads=cfg.N_HEADS, n_tx_layers=cfg.N_TX_LAYERS,
                     tx_mlp_ratio=cfg.TX_MLP_RATIO, n_stem_res=cfg.N_STEM_RES,
                     n_head_res=cfg.N_HEAD_RES, aux_reward_pred=cfg.AUX_REWARD_PRED,
                     aux_win_bet=cfg.AUX_WIN_BET, trunk_spec=getattr(cfg, "TRUNK_SPEC", "")).to(cfg.device)
    _net.popart = PopArt(cfg.POPART_BETA)
    return _net


def compute_reach(ent, em):
    """Per-source destination reachability (B,E,E): 1 unless a straight src->dest shot is absorbed
    by the sun. Recovered from encoded planet positions so it is identical in rollout and update.
    Diagonal forced to 1; the alive-mask is applied in the distribution."""
    px = ent[..., 0] * BOARD_SIZE
    py = ent[..., 1] * BOARD_SIZE
    B, E = px.shape
    sx = px.unsqueeze(2); sy = py.unsqueeze(2)         # source (B,E,1)
    dx = px.unsqueeze(1) - sx                           # src->dest (B,E,E)
    dy = py.unsqueeze(1) - sy
    dist = torch.sqrt(dx * dx + dy * dy).clamp_min(1e-6)
    ux = dx / dist; uy = dy / dist
    cx = CENTER - sx; cy = CENTER - sy                  # source -> sun centre
    t = cx * ux + cy * uy                               # closest-approach distance along the ray
    perp2 = (cx * cx + cy * cy) - t * t
    sr = SUN_RADIUS * SUN_RADIUS
    t_sun = t - torch.sqrt((sr - perp2).clamp_min(0.0))
    blocked = (t >= 0.0) & (perp2 <= sr) & (t_sun >= 0.0) & (t_sun < dist)
    reach = (~blocked).to(DTYPE)
    eye = torch.eye(E, device=ent.device, dtype=DTYPE).unsqueeze(0)
    return reach * (1.0 - eye) + eye                    # source -> itself is always reachable


def recover_ships(ent):
    """Integer ship count per planet, read directly from feature b6 (now the RAW garrison)."""
    return torch.round(ent[..., 6].clamp_min(0.0))


def _make_dist(cfg, net, ent, em, am, gl):
    '''Build the gated-allocation actor distribution. owned = legal source planets (= action_mask),
    alive = valid destination slots (= entity_mask), reach = sun-reachability mask.'''
    reach = compute_reach(ent, em) if cfg.REACH_MASK else None
    with _amp_ctx(cfg):
        dest_logits, gate_logits, value, reward_pred = net(ent, em, am, gl)
    if cfg.WHERE_DIST == 'categorical':
        dist = GatedCatDist(cfg, dest_logits.float(), gate_logits.float(), owned=am, alive=em, reach=reach)
    else:
        dist = GatedAllocDist(cfg, dest_logits.float(), gate_logits.float(), owned=am, alive=em,
                              reach=reach, kappa=cfg.ALLOC_KAPPA)
    return dist, value.float(), (reward_pred.float() if reward_pred is not None else None)


@torch.no_grad()
def act(cfg, net, ent, em, am, gl, greedy=False, crn_group=0):
    dist, value, _ = _make_dist(cfg, net, ent, em, am, gl)   # reward_pred unused while ACTING (update only)
    # crn_group>1 couples the sampling noise across each contiguous block of crn_group envs (GRPO
    # common-random-numbers, docs/grpo_v12.tex [D]); 0 = independent draws (the default everywhere else).
    action = dist.greedy() if greedy else dist.sample(crn_group=crn_group)
    if cfg.USE_POPART and getattr(net, 'popart', None) is not None:
        value = net.popart.denormalize(value)
    return action, dist.log_prob(action), value


def evaluate(cfg, net, ent, em, am, gl, action):
    dist, value, reward_pred = _make_dist(cfg, net, ent, em, am, gl)
    return dist.log_prob(action), dist.entropy(), value, reward_pred
