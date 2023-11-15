import aiohttp
import asyncio
import requests
import json
from typing import Iterable, List
import time
import numpy as np

res = []
async def post_http_request(delay, session,
                            prompt: str,
                            max_tokens: int,
                      api_url: str,
                      n: int = 1,
                      stream: bool = False) -> requests.Response:

    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt": prompt,
        "n": n,
        "use_beam_search": False,
        "temperature": 0.8,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    await asyncio.sleep(delay)
    rel_start = time.time() - start
    print(f'Start time: {rel_start:.2f}')

    async with session.post(api_url, headers=headers, json=pload) as response:
        #print(response.status)
        #print(await response.text())
        rel_finish = time.time() - start
        print(f'Finish Time: {rel_finish:.2f}')
        res.append(time.time() - start)



async def post_all(n_req, prompt, max_tokens, api_url, n, stream, concurrent=True, delay=0.1):
    loop = asyncio.get_event_loop()
    async with aiohttp.ClientSession(loop=loop) as session:

        if concurrent:
            tasks = []
            for i in range(n_req):
                tasks.append(loop.create_task(post_http_request(delay*i, session, prompt, max_tokens, api_url, n, stream)))
            await asyncio.gather(*tasks)
        else:
            for i in range(n_req):
                await post_http_request(0.0*i, session, prompt, max_tokens, api_url, n, stream)


if __name__ == '__main__':

    prompt = "San Francisco is a"
    api_url = f"http://localhost:8000/generate"
    n = 1
    stream = False
    concurrent = True
    delay = 0.01
    n_req = 50
    max_tokens = 1


    start = time.time()
    asyncio.run(post_all(n_req ,prompt, max_tokens, api_url, n, stream, concurrent, delay))

    print(np.median(np.array(res) - np.arange(n_req)*delay))

