from typing import List, Tuple, Optional, Dict, Any
from transformers_openai.user_queue import AsyncUserQueue, UserQueue
from transformers.cache_utils import Cache
from collections import defaultdict
import torch
import torch.nn.functional as F


def pad_kv(caches):
    """
    List[head, seq, dims]
    """

    shapes = [caches[i].shape[2] for i in range(len(caches))]
    maxlen = max(shapes)
    if all(s == maxlen for s in shapes):
        return torch.concat(caches)

    new_caches = []
    for i in range(len(caches)):
        pad_val = (0, 0, 0, maxlen - caches[i].shape[2], 0, 0, 0, 0)
        pad = F.pad(caches[i], pad_val, value=0.0)
        new_caches.append(pad)
    return torch.concat(new_caches)


class DynamicLengthDecoderCache(Cache):

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.current_uuid = []

    def batch_size(self):
        if len(self.key_cache) > 0:
            return len(self.key_cache[0])
        return 0

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        keys, values = [], []
        for i, k in enumerate(self.current_uuid):
            self.key_cache[layer_idx][k] = torch.cat(
                [self.key_cache[layer_idx][k], key_states[i: i + 1]], dim=-2)
            self.value_cache[layer_idx][k] = torch.cat(
                [self.value_cache[layer_idx][k], value_states[i: i + 1]], dim=-2)
            keys.append(self.key_cache[layer_idx][k])
            values.append(self.value_cache[layer_idx][k])

        k = pad_kv(keys)
        v = pad_kv(values)
        
        return k, v
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        if len(self.key_cache) <= layer_idx:
            return 0
        
        lengths = [self.key_cache[0][k].shape[2] for k in self.current_uuid]
        return max(lengths)

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None


class DynamicLengthEncoderDecoderCache(Cache):

    def __init__(self, whisper_mode=False) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.cross_key_cache: List[torch.Tensor] = []
        self.cross_value_cache: List[torch.Tensor] = []
        self.current_uuid = []
        self.whisper_mode = whisper_mode

    def get_cross_kv(self, layer_idx):
        if layer_idx < len(self):
            keys, values = [], []
            for k in self.current_uuid:
                keys.append(self.cross_key_cache[layer_idx][k])
                values.append(self.cross_value_cache[layer_idx][k])

            k = pad_kv(keys)
            v = pad_kv(values)
            return k, v
        else:
            raise KeyError(
                f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            if self.whisper_mode:
                return (
                    self.key_cache[layer_idx],
                    self.value_cache[layer_idx],
                    self.cross_key_cache[layer_idx],
                    self.cross_value_cache[layer_idx],
                )
            else:
                return self.key_cache[layer_idx], self.value_cache[layer_idx]
        else:
            raise KeyError(
                f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        keys, values = [], []
        for i, k in enumerate(self.current_uuid):
            
            self.key_cache[layer_idx][k] = torch.cat(
                [self.key_cache[layer_idx][k], key_states[i: i + 1]], dim=-2)
            self.value_cache[layer_idx][k] = torch.cat(
                [self.value_cache[layer_idx][k], value_states[i: i + 1]], dim=-2)

            keys.append(self.key_cache[layer_idx][k])
            values.append(self.value_cache[layer_idx][k])

        k = pad_kv(keys)
        v = pad_kv(values)
        return k, v

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        if len(self.key_cache) <= layer_idx:
            return 0
        
        lengths = [self.key_cache[0][k].shape[2] for k in self.current_uuid]
        return max(lengths)

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None


class StaticLengthEncoderDecoderCache(Cache):

    def __init__(
        self, 
        batch_size = 2, 
        encoder_max_length = 1024,
        decoder_max_length = 1024,
        encoder_head_size = 16,
        decoder_head_size = 16,
        encoder_dim_size = 768,
        decoder_dim_size = 768,
        encoder_hidden_layers = 12,
        decoder_hidden_layers = 12,
        dtype = torch.bfloat16,
        device = 'cuda',
        whisper_mode=False,
    ) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.cross_key_cache: List[torch.Tensor] = []
        self.cross_value_cache: List[torch.Tensor] = []
        self.current_uuid = []
        self.current_position = []
        self.whisper_mode = whisper_mode
        self.dtype = dtype
        self.device = device
        self.queue = UserQueue(batch_size)

        encoder_cache_shape = (
            encoder_head_size,
            encoder_max_length,
            encoder_dim_size // encoder_head_size
        )
        decoder_cache_shape = (
            decoder_head_size,
            decoder_max_length,
            decoder_dim_size // decoder_head_size
        )

        for k in range(encoder_hidden_layers):
            key_caches = []
            value_caches = []
            e_key_caches = []
            e_value_caches = []
            for i in range(batch_size):
                key_cache = torch.zeros(
                    decoder_cache_shape, dtype=self.dtype, device=self.device)
                value_cache = torch.zeros(
                    decoder_cache_shape, dtype=self.dtype, device=self.device)

                e_key_cache = torch.zeros(encoder_cache_shape, dtype=self.dtype, device=self.device)
                e_value_cache = torch.zeros(encoder_cache_shape, dtype=self.dtype, device=self.device)

                torch._dynamo.mark_static_address(key_cache)
                torch._dynamo.mark_static_address(value_cache)
                torch._dynamo.mark_static_address(e_key_cache)
                torch._dynamo.mark_static_address(e_value_cache)

                key_caches.append(key_cache)
                value_caches.append(value_cache)
                e_key_caches.append(e_key_cache)
                e_value_caches.append(e_value_cache)

                # e_key_cache[:, :, :] = existing_cache[k][2][i].clone()
                # e_value_cache[:, :, :] = existing_cache[k][3][i].clone()

            self.key_cache.append(key_caches)
            self.value_cache.append(value_caches)
            self.cross_key_cache.append(e_key_caches)
            self.cross_value_cache.append(e_value_caches)


    def get_cross_kv(self, layer_idx):
        if layer_idx < len(self):
            keys, values = [], []
            for i in range(len(self.current_position)):
                keys.append(self.key_cache[layer_idx][self.current_position[i]])
                values.append(self.value_cache[layer_idx][self.current_position[i]])
            return torch.stack(keys, 0), torch.stack(values, 0)
        else:
            raise KeyError(
                f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            if self.whisper_mode:
                return (
                    self.key_cache[layer_idx],
                    self.value_cache[layer_idx],
                    self.cross_key_cache[layer_idx],
                    self.cross_value_cache[layer_idx],
                )
            else:
                return self.key_cache[layer_idx], self.value_cache[layer_idx]
        else:
            raise KeyError(
                f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_position = cache_kwargs.get("cache_position")
        
        keys, values = [], []
        for i in range(len(self.current_position)):
            k_out = self.key_cache[layer_idx][self.current_position[i]]
            v_out = self.value_cache[layer_idx][self.current_position[i]]
            k_out[:, cache_position[i]] = key_states[i].clone()
            v_out[:, cache_position[i]] = value_states[i].clone()
            keys.append(k_out)
            values.append(v_out)
        
        return torch.stack(keys, 0), torch.stack(values, 0)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        if len(self.key_cache) <= layer_idx:
            return 0
        
        lengths = [self.key_cache[0][k].shape[1] for k in self.current_position]
        return max(lengths)

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None