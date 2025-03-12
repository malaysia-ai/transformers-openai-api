import argparse
import logging
import os
import torch

def parse_arguments():
    parser = argparse.ArgumentParser(description='Configuration parser')

    parser.add_argument(
        '--host', type=str, default=os.environ.get('HOSTNAME', '0.0.0.0'),
        help='host name to host the app (default: %(default)s, env: HOSTNAME)'
    )
    parser.add_argument(
        '--port', type=int, default=int(os.environ.get('PORT', '7088')),
        help='port to host the app (default: %(default)s, env: PORT)'
    )
    parser.add_argument(
        '--loglevel', default=os.environ.get('LOGLEVEL', 'INFO').upper(),
        help='Logging level (default: %(default)s, env: LOGLEVEL)'
    )
    parser.add_argument(
        '--model-type',
        default=os.environ.get('MODEL_TYPE', 'AutoModelForCausalLM'),
        help='Model type (default: %(default)s, env: MODEL_TYPE)'
    )
    parser.add_argument(
        '--tokenizer-type',
        default=os.environ.get('TOKENIZER_TYPE', 'AutoTokenizer'),
        help='Tokenizer type (default: %(default)s, env: TOKENIZER_TYPE)'
    )
    parser.add_argument(
        '--tokenizer-use-fast', type=lambda x: x.lower() == 'true',
        default=os.environ.get('TOKENIZER_USE_FAST', 'true').lower() == 'true',
        help='Use fast tokenizer (default: %(default)s, env: TOKENIZER_USE_FAST)'
    )
    parser.add_argument(
        '--processor-type',
        default=os.environ.get('PROCESSOR_TYPE', 'AutoTokenizer'),
        help='Processor type (default: %(default)s, env: PROCESSOR_TYPE)'
    )
    parser.add_argument(
        '--hf-model',
        default=os.environ.get('HF_MODEL', 'mesolitica/malaysian-llama2-7b-32k-instructions'),
        help='Hugging Face model (default: %(default)s, env: HF_MODEL)'
    )
    parser.add_argument(
        '--torch-dtype', default=os.environ.get('TORCH_DTYPE', 'bfloat16'),
        help='Torch data type (default: %(default)s, env: TORCH_DTYPE)'
    )
    parser.add_argument(
        '--architecture-type',
        default=os.environ.get('ARCHITECTURE_TYPE', 'decoder'),
        choices=['decoder', 'encoder-decoder'],
        help='Architecture type (default: %(default)s, env: ARCHITECTURE_TYPE)'
    )
    parser.add_argument(
        '--serving-type',
        default=os.environ.get('SERVING_TYPE', 'chat'),
        choices=['chat', 'whisper'],
        help='Serving type (default: %(default)s, env: SERVING_TYPE)'
    )
    parser.add_argument(
        '--continuous-batching-microsleep', type=float,
        default=float(os.environ.get('CONTINUOUS_BATCHING_MICROSLEEP', '1e-4')),
        help='microsleep to group continuous batching, 1 / 1e-4 = 10k steps for one second (default: %(default)s, env: CONTINUOUS_BATCHING_MICROSLEEP)'
    )
    parser.add_argument(
        '--continuous-batching-batch-size', type=int,
        default=int(os.environ.get('CONTINUOUS_BATCHING_BATCH_SIZE', '20')),
        help='maximum of batch size during continuous batching (default: %(default)s, env: CONTINUOUS_BATCHING_BATCH_SIZE)'
    )
    parser.add_argument(
        '--static-cache', type=lambda x: x.lower() == 'true',
        default=os.environ.get('STATIC_CACHE', 'false').lower() == 'true',
        help='Preallocate KV Cache for faster inference (default: %(default)s, env: STATIC_CACHE)'
    )
    parser.add_argument(
        '--static-cache-encoder-max-length',
        type=int,
        default=int(os.environ.get('STATIC_CACHE_ENCODER_MAX_LENGTH', '256')),
        help='Maximum concurrent requests (default: %(default)s, env: STATIC_CACHE_ENCODER_MAX_LENGTH)'
    )
    parser.add_argument(
        '--static-cache-decoder-max-length',
        type=int,
        default=int(os.environ.get('STATIC_CACHE_DECODER_MAX_LENGTH', '256')),
        help='Maximum concurrent requests (default: %(default)s, env: STATIC_CACHE_DECODER_MAX_LENGTH)'
    )
    parser.add_argument(
        '--accelerator-type', default=os.environ.get('ACCELERATOR_TYPE', 'cuda'),
        help='Accelerator type (default: %(default)s, env: ACCELERATOR_TYPE)'
    )
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=int(os.environ.get('MAX_CONCURRENT', '100')),
        help='Maximum concurrent requests (default: %(default)s, env: MAX_CONCURRENT)'
    )
    parser.add_argument(
        '--torch-profiling',
        type=lambda x: x.lower() == 'true',
        default=os.environ.get('TORCH_PROFILING', 'false').lower() == 'true',
        help='Use torch.autograd.profiler.profile() to profile prefill and step (default: %(default)s, env: TORCH_PROFILING)'
    )
    parser.add_argument(
        '--hqq', type=lambda x: x.lower() == 'true',
        default=os.environ.get('HQQ', 'true').lower() == 'false',
        help='int4 quantization using HQQ (default: %(default)s, env: HQQ)'
    )
    parser.add_argument(
        '--torch-compile', type=lambda x: x.lower() == 'true',
        default=os.environ.get('TORCH_COMPILE', 'true').lower() == 'false',
        help='Torch compile necessary forwards, can speed up at least 1.5X (default: %(default)s, env: TORCH_COMPILE)'
    )

    args = parser.parse_args()

    if args.hf_model is None:
        raise ValueError('must set `--hf-model` or `HF_MODEL` environment variable.')

    device = 'cpu'
    if args.accelerator_type == 'cuda':
        if not torch.cuda.is_available():
            logging.warning('CUDA is not available, fallback to CPU.')
        else:
            device = 'cuda'

    args.device = device
    return args


args = parse_arguments()
args.ready = False
logging.basicConfig(level=args.loglevel)
logging.info(f'Serving app using {args}')
