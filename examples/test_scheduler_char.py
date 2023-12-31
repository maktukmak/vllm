import aiohttp
import asyncio
import requests
import json
from typing import Iterable, List
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


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
    #print(f'Start time: {rel_start:.2f}')

    async with session.post(api_url, headers=headers, json=pload) as response:
        if (response.status != 200):
            raise
        rel_finish = time.time() - start
        #print(f'Finish Time: {rel_finish:.2f}')
        res.append(rel_finish)



async def post_all(prompts, max_tokens, api_url, n, stream, concurrent, delays=0.1):
    n_req = len(prompts)
    loop = asyncio.get_event_loop()
    async with aiohttp.ClientSession(loop=loop) as session:

        if concurrent:
            tasks = []
            for i in range(n_req):
                tasks.append(loop.create_task(post_http_request(delays[i], session, prompts[i], max_tokens[i], api_url, n, stream)))
            await asyncio.gather(*tasks)
        else:
            for i in range(n_req):
                
                seq_start = time.time()
                await post_http_request(0.0, session, prompts[i], max_tokens[i], api_url, n, stream)
                if i < n_req-1:
                    while(time.time() - seq_start <  (delays[i+1] - delays[i])): True



if __name__ == '__main__':

    api_url = f"http://localhost:8000/generate"
    stream = False

    n_req = 100
    inp_lens = [16] * n_req
    prompts = ["San " * l for l in inp_lens]
    out_lens = [1] * n_req
    delays = np.cumsum([0.0000] * (n_req-1))
    delays = np.concatenate(([0.], delays))
    n = 1

    print('Concurrent')
    concurrent = True
    res = []
    start = time.time()
    asyncio.run(post_all(prompts, out_lens, api_url, n, stream, concurrent, delays))
    latencies_concurrent = np.array(res) - delays
    print(np.median(latencies_concurrent))


