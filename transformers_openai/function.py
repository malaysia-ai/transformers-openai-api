import torch
import torch.nn.functional as F
from typing import Optional

profiler = lambda x: torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        with_flops=True,
        with_modules=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(x)
)

def prefill_attention_mask_flatten(seq_lens, dtype=torch.bfloat16, causal=True):
    masks = []
    for l in seq_lens:
        m = torch.ones(l, l)
        if causal:
            m = torch.tril(m)
        masks.append(m)
    total_size = sum(mask.size(0) for mask in masks)
    combined_mask = torch.zeros(total_size, total_size, dtype=dtype)

    current_pos = 0
    for mask in masks:
        size = mask.size(0)
        combined_mask[current_pos:current_pos + size, current_pos:current_pos + size] = mask
        current_pos += size

    min_value = torch.finfo(dtype).min if dtype.is_floating_point else torch.iinfo(dtype).min
    inverted_mask = torch.where(combined_mask == 1, torch.tensor(0, dtype=dtype), min_value)
    return inverted_mask.unsqueeze(0)

def step_attention_mask_flatten(seq_lens, dtype=torch.bfloat16):
    num_queries = len(seq_lens)
    total_kv_cache_len = sum(seq_lens)
    neg_inf = torch.finfo(dtype).min
    mask = torch.full((1, 1, num_queries, total_kv_cache_len), neg_inf, dtype=dtype)
    start = 0
    for i, length in enumerate(seq_lens):
        end = start + length
        causal_mask = torch.triu(torch.full((length, length), neg_inf, dtype=dtype), diagonal=1)
        mask[:, :, i, start:end] = causal_mask[-1, :]
        start = end

    return mask

def prefill_attention_mask(batch_size, max_len, lengths, device, dtype, causal=True):
    lengths = torch.tensor(lengths)
    if causal:
        causal_mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        batch_indices = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
        combined_mask = causal_mask.unsqueeze(0) | ~batch_indices.unsqueeze(2)
        mask = combined_mask.unsqueeze(1)
        mask = mask.type(dtype).masked_fill(mask, torch.finfo(dtype).min)
    else:
        batch_indices = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
        mask = (batch_indices.unsqueeze(1).unsqueeze(1) & batch_indices.unsqueeze(1).unsqueeze(3))
        mask = mask.type(dtype)
        
    return mask.to(device)
    

def efficient_attention_mask(batch_size, max_len, lengths, device, dtype, ones=True):
    lengths = torch.tensor(lengths)
    left = torch.arange(max_len).expand(
        batch_size, 1, 1, max_len)
    right = lengths.view(
        batch_size, 1, 1, 1)
    if ones:
        mask = left < right
        mask = mask.type(dtype)
    else:
        mask = left > right
        mask = mask.type(dtype).masked_fill(mask, torch.finfo(dtype).min)
    return mask.to(device)


def pad_attention_mask(attention_mask):
    maxlen = max([attention_mask[i].shape[1] for i in range(len(attention_mask))])
    attention_mask = [
        F.pad(
            attention_mask[i],
            (0, maxlen - attention_mask[i].shape[1], 0, 0)) for i in range(
            len(attention_mask))]
    return torch.concat(attention_mask)


def pad_hidden_encoder(out_encoder):
    maxlen = max([out_encoder[i].shape[1] for i in range(len(out_encoder))])
    out_encoder = [
        F.pad(
            out_encoder[i],
            (0, 0, 0, maxlen - out_encoder[i].shape[1], 0, 0)) for i in range(
            len(out_encoder))]
    return torch.concat(out_encoder)


def multinomial_sample_one_no_sync(probs_sort):
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(
    logits,
    mask_penalty,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
):
    logits = logits / mask_penalty
    if temperature > 0:
        logits = logits / max(temperature, 1e-5)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            pivot = v.select(-1, -1).unsqueeze(-1)
            logits = torch.where(logits < pivot, -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)

        if top_p is not None:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            indices_to_remove = cumulative_probs > top_p
            indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
            indices_to_remove[..., 0] = 0
            probs[sorted_indices[indices_to_remove]] = 0.0
            probs = probs / probs.sum(dim=-1, keepdim=True)  # renormalize

        idx_next = multinomial_sample_one_no_sync(probs)
    else:
        probs = logits
        idx_next = logits.argmax(-1, keepdim=True)

    return idx_next, probs


def sample(
    logits,
    mask_penalty,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
):
    return logits_to_probs(logits[0, -1], mask_penalty[0], temperature, top_k, top_p)


def format_timestamp(
    seconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def cleanup_cache(cache):
    try:
        if isinstance(cache, tuple) or isinstance(cache, list):
            cache = list(cache)
            for i in range(len(cache)):
                cache[i] = list(cache[i])
                for _ in range(len(cache[i])):
                    del cache[i][0]
        else:
            for _ in range(len(cache.key_cache)):
                del cache.key_cache[0]
            for _ in range(len(cache.value_cache)):
                del cache.value_cache[0]
    except Exception as e:
        logging.warning('failed to clear cache')
