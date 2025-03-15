import asyncio
import logging
import uuid
import time
from collections import deque

HEADERS = {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive'
}

class InsertMiddleware:
    def __init__(self, app, max_concurrent=50):
        self.app = app
        self.max_concurrent = max_concurrent
        self.queue = deque()

    async def process_request(self, scope, receive, send):

        queue = asyncio.Queue()

        async def message_poller(sentinel, handler_task):
            nonlocal queue
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    handler_task.cancel()
                    return sentinel
                await queue.put(message)

        sentinel = object()
        handler_task = asyncio.create_task(self.app(scope, queue.get, send))
        asyncio.create_task(message_poller(sentinel, handler_task))

        try:
            await handler_task
            if 'time_max_tokens' in scope['request']:
                time_taken_first_token = scope['request']['time_first_token'] - scope['request']['time_in']
                time_taken_max_tokens = scope['request']['time_max_tokens'] - scope['request']['time_in']
                tps = scope['request']['total_tokens'] / time_taken_max_tokens

                if 'total_seconds' in scope['request']:
                    sps = scope['request']['total_seconds'] / time_taken_max_tokens
                    extra_whisper = f', Seconds Per Second {sps}'
                else:
                    extra_whisper = ''

                s = f"Complete {scope['request']['uuid']}, time first token {time_taken_first_token} seconds, time taken {time_taken_max_tokens} seconds, TPS {tps}{extra_whisper}"
                logging.info(s)
        except asyncio.CancelledError:
            logging.warning(f"Cancelling {scope['request']['uuid']} due to disconnect")
                

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        scope['request'] = {
            'uuid': str(uuid.uuid4()),
            'time_in': time.time()
        }

        await self.process_request(scope, receive, send)

        if self.queue:
            next_request = self.queue.popleft()
            next_request.set_result(None)