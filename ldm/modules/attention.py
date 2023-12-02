from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any

from ldm.modules.diffusionmodules.util import checkpoint


try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# CrossAttn precision handling
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        # if context is not None:
        #     print(x.shape, context.shape)
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
    
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class SharedCrossAttention(CrossAttention):
    def __init__(self, shared_kv, return_kv, *args, **kwargs):
        super().__init__()
        assert isinstance(shared_kv, bool) and isinstance(return_kv, bool)
        assert shared_kv != return_kv
        self.return_kv = return_kv
        self.shared_kv = shared_kv

    def forward(self, x, shared_k=None, shared_v=None, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        if self.shared_kv:
            assert shared_k is not None and shared_v is not None
            k = torch.cat([k, shared_k], dim=-2)
            v = torch.cat([v, shared_v], dim=-2)
        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
    
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        # if self.return_kv:
        return self.to_out(out), k, v
        # else:
            # return self.to_out(out)
    

class MutualCrossAttention(nn.Module):
    def __init__(self, query_dim, context1_dim=None, context2_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context1_dim = default(context1_dim, query_dim)
        context2_dim = default(context2_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q1 = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k1 = nn.Linear(context1_dim, inner_dim, bias=False)
        self.to_v1 = nn.Linear(context1_dim, inner_dim, bias=False)

        self.to_q2 = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k2 = nn.Linear(context2_dim, inner_dim, bias=False)
        self.to_v2 = nn.Linear(context2_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x1, x2, context1=None, context2=None, mask1=None, mask2=None):
        h = self.heads

        q1 = self.to_q1(x1)
        context1 = default(context1, x1)
        k1 = self.to_k1(x1)
        v1 = self.to_v1(x1)

        q2 = self.to_q2(x2)
        context2 = default(context2, x2)
        k2 = self.to_k2(x2)
        v2 = self.to_v2(x2)

        q1, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q1, k1, v1))
        q2, k2, v2 = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q2, k2, v2))

        k = torch.cat([k1, k2], dim=-2)
        v = torch.cat([v1, v2], dim=-2)

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q1, k = q1.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q1, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q1, k) * self.scale
        
        del q1, k
    
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        # print(x.shape, context.shape)
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        # print(x.shape, context.shape)
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class BasicSharedTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": SharedCrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False, shared_kv=True, return_kv=False):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(shared_kv=shared_kv, return_kv=return_kv, query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(shared_kv=shared_kv, return_kv=return_kv, query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        self.return_kv = return_kv

    def forward(self, x, shared_k=None, shared_v=None, context=None):
        return checkpoint(self._forward, (x, shared_k, shared_v, context), self.parameters(), self.checkpoint)

    def _forward(self, x, shared_k=None, shared_v=None, context=None):
        # print(x.shape, context.shape)
        if self.return_kv:
            h, _, _ = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None)
        else:
            h = self.attn1(self.norm1(x), shared_k=shared_k, shared_v=shared_v, context=context if self.disable_self_attn else None)
        x = h + x
        # print(x.shape, context.shape)
        if self.return_kv:
            h, k, v = self.attn2(self.norm2(x), context=context)
        else:
            h = self.attn2(self.norm2(x), shared_k=shared_k, shared_v=shared_v, context=context)
        x = h + x
        x = self.ff(self.norm3(x)) + x
        if self.return_kv:
            return x, k, v
        return x
    
class BasicMutualTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": [MutualCrossAttention, CrossAttention],  # vanilla attention
        # "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context1_dim=None, context2_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        super().__init__()
        attn_mode = "softmax"
        assert attn_mode in self.ATTENTION_MODES
        mutual_attn_cls = self.ATTENTION_MODES[attn_mode][0]
        attn_cls = self.ATTENTION_MODES[attn_mode][1]
        self.disable_self_attn = disable_self_attn
        self.mutual_attn1 = mutual_attn_cls(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
            context1_dim=context1_dim if self.disable_self_attn else None,
            context2_dim=context2_dim if self.disable_self_attn else None
        )  # is a self-attention if not self.disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim, context_dim=context2_dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )
        self.mutual_ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.mutual_attn2 = mutual_attn_cls(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
            context1_dim=context1_dim if self.disable_self_attn else None,
            context2_dim=context2_dim if self.disable_self_attn else None
        )  # is a self-attention if not self.disable_self_attn
        self.attn2 = attn_cls(query_dim=dim, context_dim=context2_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
    
    def forward(self, x1, x2, context1=None, context2=None):
        return checkpoint(self._forward, (x1, x2, context1, context2), self.parameters(), self.checkpoint)
    
    def _forward(self, x1, x2, context1=None, context2=None):
        x1 = self.mutual_attn1(
            self.norm1(x1), self.norm1(x2),
            context1=context1 if self.disable_self_attn else None,
            context2=context2 if self.disable_self_attn else None
        ) + x1
        x2 = self.attn1(
            self.norm1(x2), context=context2 if self.disable_self_attn else None
        ) + x2
        x1 = self.mutual_attn2(
            self.norm2(x1), self.norm2(x2),
            context1=context1 if self.disable_self_attn else None,
            context2=context2 if self.disable_self_attn else None
        ) + x1
        x2 = self.attn2(
            self.norm2(x2), context=context2 if self.disable_self_attn else None
        ) + x2
        x1 = self.mutual_ff(self.norm3(x1)) + x1
        x2 = self.ff(self.norm3(x2)) + x2
        return x1, x2


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        # print(x.shape, context[0].shape)
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            c = context[i]
            if exists(c):
                c = rearrange(c, 'b c h w -> b (h w) c').contiguous()
            x = block(x, context=c)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class SharedSpatialTransformer(SpatialTransformer):
    def __init__(self, context_dim, depth=1, shared_kv=True, return_kv=False, *args, **kwargs):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([
            BasicSharedTransformerBlock(context_dim=context_dim[d], shared_kv=shared_kv, return_kv=return_kv, 
                                        *args, **kwargs) for d in range(depth)
        ])
        self.return_kv = return_kv
        self.shared_kv = shared_kv
    
    def forward(self, x, shared_k=None, shared_v=None, context=None):
        if not isinstance(context, list):
            context = [context]
        # print(x.shape, context[0].shape)
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        
        if self.return_kv:
            ks, vs = [], []
        for i, block in enumerate(self.transformer_blocks):
            c = context[i]
            if exists(c):
                c = rearrange(c, 'b c h w -> b (h w) c').contiguous()
            if self.return_kv:
                x, k, v = block(x)
                ks.append(k)
                vs.append(v)
            x = block(x, shared_k[i], shared_v[i], context=c)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        
        if self.return_kv:
            return x + x_in, ks, vs
        return x + x_in



class SpatialMutualTransformer(nn.Module):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.,
        context1_dim=None,
        context2_dim=None,
        disable_self_attn=False,
        use_linear=False,
        use_checkpoint=True
    ):
        super().__init__()
        if exists(context1_dim) and not isinstance(context1_dim, list):
            context1_dim = [context1_dim]
        if exists(context2_dim) and not isinstance(context2_dim, list):
            context2_dim = [context2_dim]
        
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in1 = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
            self.proj_in2 = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in1 = nn.Linear(in_channels, inner_dim)
            self.proj_in2 = nn.Linear(in_channels, inner_dim)
        
        self.transformer_blocks = nn.ModuleList(
            [BasicMutualTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context1_dim=context2_dim[d], context2_dim=context2_dim[d],
                                         disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
                for d in range(depth)]
        )

        if not use_linear:
            self.proj_out1 = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
            self.proj_out2 = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out1 = zero_module(nn.Linear(in_channels, inner_dim))
            self.proj_out2 = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x1, x2, context1=None, context2=None):
        if not isinstance(context1, list):
            context1 = [context1]
        if not isinstance(context2, list):
            context2 = [context2]
        b, c, h, w = x1.shape
        x_in1, x_in2 = x1, x2
        x1 = self.norm(x1)
        x2 = self.norm(x2)
        if not self.use_linear:
            x1 = self.proj_in1(x1)
            x2 = self.proj_in2(x2)
        x1 = rearrange(x1, 'b c h w -> b (h w) c').contiguous()
        x2 = rearrange(x2, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x1 = self.proj_in1(x1)
            x2 = self.proj_in2(x2)
        for i, block in enumerate(self.transformer_blocks):
            c1 = context1[i]
            if exists(c):
                c = rearrange(c, 'b c h w -> b (h w) c').contiguous()
            x1, x2 = block(x1, x2, context1=c1, context2=context2[i])
        if self.use_linear:
            x1 = self.proj_out1(x1)
            x2 = self.proj_out2(x2)
        x1 = rearrange(x1, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        x2 = rearrange(x2, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x1 = self.proj_out1(x1)
            x2 = self.proj_out2(x2)
        return x1 + x_in1, x2 + x_in2

