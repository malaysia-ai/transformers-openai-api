from transformers_openai.env import args
from typing import Optional
import logging
import importlib
import transformers
import torch

SPIECE_UNDERLINE = "▁"

def load_hf_model():
    if 't5' in args.model_type.lower() and 'transformers_openai.models' not in args.model_type:
        raise ValueError('We only support `--model-type transformers_openai.models.T5ForConditionalGeneration` for T5.')

    if '.' in args.model_type:
        module_name, class_name = args.model_type.rsplit('.', 1)
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
    else:
        class_ = getattr(transformers, args.model_type)

    logging.info(f'loading {class_} {args.hf_model}')

    if 'auto_gptq' in args.model_type:
        return class_.from_quantized(args.hf_model)
    else:
        return class_.from_pretrained(
            args.hf_model,
            attn_implementation=args.attn_implementation,
            torch_dtype=getattr(torch, args.torch_dtype),
        ).eval().to(args.device)


def load_hf_processor():
    if '.' in args.processor_type:
        module_name, class_name = args.processor_type.rsplit('.', 1)
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
    else:
        class_ = getattr(transformers, args.processor_type)

    return class_.from_pretrained(args.hf_model)


def load_hf_tokenizer():
    tokenizer = getattr(transformers, args.tokenizer_type).from_pretrained(
        args.hf_model,
        use_fast=args.tokenizer_use_fast,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    tokenizer.padding_side = 'right'
    return tokenizer


def decode(tokenizer, ids):
    tokens = tokenizer.convert_ids_to_tokens(ids)
    if SPIECE_UNDERLINE in tokens[0]:
        prefix = ' '
    else:
        prefix = ''
    return prefix + tokenizer.convert_tokens_to_string(tokens)