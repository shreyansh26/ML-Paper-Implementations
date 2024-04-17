import math
import sys
import torch
from torch import Tensor
import torch.nn.functional as F
import jaxtyping
from jaxtyping import Float32
from flash_attn_triton import flash_attn_func as flash_attn_func_triton
from flash_attn_interface import flash_attn_with_kvcache

torch.set_default_device("cuda")

def attention_prefix(q: Float32[Tensor, "batch qseq_len qheads dim"], k: Float32[Tensor, "batch kvseq_len kvheads dim"], v: Float32[Tensor, "batch kvseq_len kvheads dim"]):
    out, lse = flash_attn_func_triton(q, k, v, None, False, 1.0/math.sqrt(q.shape[-1])) 
    return out, lse

def attention_suffix(q: Float32[Tensor, "batch qseq_len qheads dim"], k: Float32[Tensor, "batch kvseq_len kvheads dim"], v: Float32[Tensor, "batch kvseq_len kvheads dim"]):
    out, lse = flash_attn_with_kvcache(q, k, v) 
    return out, lse

def combine_lse(out1: Float32[Tensor, "batch qseq_len qheads dim"], lse1: Float32[Tensor, "batch qheads qseq_len"], out2: Float32[Tensor, "batch qseq_len qheads dim"], lse2: Float32[Tensor, "batch qheads qseq_len"]):
# def combine_lse(out1, lse1, out2, lse2):
    print(out1.shape, out2.shape)
    print(lse1.shape, lse2.shape)
    lse1 = lse1.transpose(1, 2).to(out1.dtype)
    lse2 = lse2.transpose(1, 2).to(out2.dtype)

    max_lse = torch.maximum(lse1, lse2)
    adjust_factor1 = (lse1 - max_lse).exp()
    adjust_factor2 = (lse2 - max_lse).exp()

    new_denominator = adjust_factor1 + adjust_factor2

    aggregated = (
        out1 * adjust_factor1.unsqueeze(-1) + out2 * adjust_factor2.unsqueeze(-1)
    ) / new_denominator.unsqueeze(-1)

    return aggregated

def hydragen_attention(q: Float32[Tensor, "batch num_queries qheads dim"], prefix_k: Float32[Tensor, "prefix_len kvheads dim"], prefix_v: Float32[Tensor, "prefix_len kvheads dim"], suffix_k: Float32[Tensor, "batch suffix_len kvheads dim"], suffix_v: Float32[Tensor, "batch suffix_len kvheads dim"]):
    b, nq, hq, d = q.shape

    # inter-sequence batching: merge attention queries
    # as if they all came from the same sequence
    batched_q = q.view(1, b*nq, hq, d)

    # efficient attention over prefixes
    # prefix_out: shape [1, batch * nq, hq, dim]
    # prefix_lse: shape [1, hq, batch * nq]
    prefix_out, prefix_lse = attention_prefix(batched_q, prefix_k.unsqueeze(0), prefix_v.unsqueeze(0))

    print(prefix_out.shape)
    print(prefix_lse.shape)

    # normal attention over suffixes
    # suffix_out: shape [batch, q_len, hq, dim]
    # suffix_lse: shape [batch, hq, q_len]
    suffix_out, suffix_lse = attention_suffix(q, suffix_k, suffix_v)
    print(suffix_out.shape)
    print(suffix_lse.shape)

    aggregated = combine_lse(prefix_out.view(b, nq, hq, d), prefix_lse.view(b, hq, nq), suffix_out, suffix_lse)

    return aggregated

if __name__ == "__main__":
    b = 256
    nq = 1
    prefix_len = 512
    suffix_len = 1
    hq = 32
    hkv = 32
    d = 128

    q = torch.randn(b, nq, hq, d, dtype=torch.bfloat16)
    prefix_k = torch.randn(prefix_len, hkv, d, dtype=torch.bfloat16)
    prefix_v = torch.randn(prefix_len, hkv, d, dtype=torch.bfloat16)

    suffix_k = torch.randn(b, suffix_len, hkv, d, dtype=torch.bfloat16)
    suffix_v = torch.randn(b, suffix_len, hkv, d, dtype=torch.bfloat16)

    hydragen_out = hydragen_attention(q, prefix_k, prefix_v, suffix_k, suffix_v)

    print("Hydragen out shape:", hydragen_out.shape)


    prefix_k_expanded = prefix_k.unsqueeze(0).expand(suffix_k.size(0), -1, -1, -1)
    k = torch.cat((prefix_k_expanded, suffix_k), 1)

    prefix_v_expanded = prefix_v.unsqueeze(0).expand(suffix_v.size(0), -1, -1, -1)
    v = torch.cat((prefix_v_expanded, suffix_v), 1)

    out, _ = attention_prefix(q, k, v)
    print("Out shape:", out.shape)


    try:
        torch.testing.assert_close(hydragen_out, out, rtol=1e-1, atol=1e-1)
        print("All good attention!")
    except Exception as e:
        print(e)
