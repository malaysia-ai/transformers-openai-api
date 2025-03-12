from transformers_openai.env import args
from transformers_openai.base_model import ChatCompletionForm
from transformers_openai.middleware import InsertMiddleware, HEADERS
from fastapi import FastAPI, Request
from fastapi import File, Form, UploadFile
from sse_starlette import EventSourceResponse
from io import BytesIO
import asyncio
import uvicorn
import logging
import wave
import numpy as np
import os

if args.serving_type == 'chat':
    if args.accelerator_type == 'cuda':
        from transformers_openai.main_cuda import (
            chat_completions,
            load_model,
            prefill,
            step,
        )
if args.serving_type == 'whisper':
    if args.accelerator_type == 'cuda':
        from transformers_openai.main_whisper_cuda import (
            audio_completions,
            load_model,
            prefill,
            step,
        )


app = FastAPI()
app.add_middleware(InsertMiddleware, max_concurrent=args.max_concurrent)

if args.serving_type == 'chat':
    @app.post('/chat/completions')
    async def chat_completions_main(
        form: ChatCompletionForm,
        request: Request = None,
    ):
        generator = chat_completions(form=form, request=request)
        r = await generator
        if form.stream:
            return EventSourceResponse(r, headers=HEADERS)
        else:
            return r

if args.serving_type == 'whisper':
    @app.post('/audio/transcriptions')
    async def audio_transcriptions_main(
        file: bytes = File(),
        model: str = Form('base'),
        language: str = Form('None'),
        response_format: str = Form('text'),
        timestamp_granularities: str = Form('segment'),
        stream: bool = Form(False),
        repetition_penalty: float = Form(1.0),
        temperature: float = Form(0.0),
        top_p: float = Form(0.95),
        top_k: int = Form(50),
        request: Request = None,
    ):
        if isinstance(language, str):
            language = language.lower().strip()
            if language in {'null', 'none'}:
                language = None

        generator = audio_completions(
            file=file,
            language=language,
            response_format=response_format,
            timestamp_granularities=timestamp_granularities,
            stream=stream,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            request=request,
        )
        r = await generator
        if stream:
            return EventSourceResponse(r, headers=HEADERS)
        else:
            return r

@app.on_event("startup")
async def startup_event():
    app.state.background_prefill = asyncio.create_task(prefill())
    app.state.background_step = asyncio.create_task(step())
    load_model()

@app.on_event("shutdown")
async def shutdown_event():
    app.state.background_prefill.cancel()
    app.state.background_step.cancel()
    try:
        await app.state.background_prefill
    except asyncio.CancelledError:
        pass
    try:
        await app.state.background_step
    except asyncio.CancelledError:
        pass

if args.serving_type == 'whisper':
    async def warm(index=0, repeat=2):
        logging.info(f'{index} warming up whisper torch compile static cache')
        
        """
        file = BytesIO()
        sample_rate = 16000
        samples = np.zeros((30 * sample_rate,)).astype(np.int16)
        with wave.open(file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  
            wf.setframerate(sample_rate)
            wf.writeframes(samples.tobytes())
        file.seek(0)
        """

        file = os.path.join(os.path.dirname(__file__), 'warmup.wav')
        for k in range(repeat):
            generator = audio_completions(
                file=file,
                language=None,
                stream=True,
                request={'uuid': f'{index}-{k}'}
            )
            r = await generator
            async for t in r:
                logging.info(f'{index} {k} {t}')

    @app.on_event('startup')
    async def warmup():
        if args.torch_compile and args.static_cache:
            repeat = 2
            batch_size = args.continuous_batching_batch_size + 1
        else:
            repeat = 1
            batch_size = 2
        for i in range(1, batch_size, 1):
            tasks = []
            for index in range(i):
                task = asyncio.create_task(warm(index=index, repeat=repeat))
                tasks.append(task)
            await asyncio.gather(*tasks)
        
        args.ready = True

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.loglevel.lower(),
        access_log=True,
    )
