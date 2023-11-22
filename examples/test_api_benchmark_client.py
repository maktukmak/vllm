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



    if True:


        rates = 2 ** np.arange(2, 19, 2)
        res_list_conc = []
        res_list_seq = []
        for rate in rates:

            n_req = rate * 10
            inp_lens = [16] * n_req
            prompts = ["San " * l for l in inp_lens]
            out_lens = [1] * n_req
            n = 1

            delays = np.cumsum([1 / rate] * (n_req-1))
            delays = np.concatenate(([0], delays))

            print('Concurrent')
            concurrent = True
            res = []
            start = time.time()
            asyncio.run(post_all(prompts, out_lens, api_url, n, stream, concurrent, delays))
            latencies_concurrent = np.array(res) - delays
            print(np.median(latencies_concurrent))
            res_list_conc.append(np.median(latencies_concurrent))

            print('Sequential')
            concurrent = False
            res = []
            start = time.time()
            asyncio.run(post_all(prompts, out_lens, api_url, n, stream, concurrent, delays))
            latencies_concurrent = np.array(res) - delays
            print(np.median(latencies_concurrent))
            res_list_seq.append(np.median(latencies_concurrent))


        plt.plot(rates, res_list_conc, marker='o', label='Continuous')
        plt.plot(rates, res_list_seq, marker='o', label='FCFS')
        plt.xlabel('Query per sec')
        plt.ylabel('Median turnaround time (s)')
        plt.legend()
        plt.xscale('log', base=2)
        plt.grid()
        plt.savefig('rate_vs_latency_cpu.jpg')


    if False:

        n_req = 100
        inp_lens = np.random.uniform(1, 512, n_req).astype(int)
        inp_lens = [256] * n_req
        prompts = ["San " * l for l in inp_lens]
        out_lens = [1] * n_req
        #delays = np.cumsum(np.random.poisson(10, n_req-1) * 0.001)
        delays = np.cumsum([0.0] * (n_req-1))
        delays = np.concatenate(([0], delays))
        n = 1
        
        
        print('Concurrent')
        concurrent = True
        res = []
        start = time.time()
        asyncio.run(post_all(prompts, out_lens, api_url, n, stream, concurrent, delays))
        latencies_concurrent = np.array(res) - delays
        print(np.median(latencies_concurrent))

        print('Sequential')
        concurrent = False
        res = []
        start = time.time()
        asyncio.run(post_all(prompts, out_lens, api_url, n, stream, concurrent, delays))
        latencies_sequential = np.array(res) - delays
        print(np.median(latencies_sequential))


        hist, bins = np.histogram(latencies_concurrent, bins=10)
        bin_centers = (bins[1:]+bins[:-1])*0.5
        plt.plot(bin_centers, np.cumsum(hist) / np.sum(hist), label='Continuous')

        hist, bins = np.histogram(latencies_sequential, bins=10)
        bin_centers = (bins[1:]+bins[:-1])*0.5
        plt.plot(bin_centers, np.cumsum(hist) / np.sum(hist), label='FCFS')
        plt.xlabel('Latency (s)')
        plt.ylabel('CDF')
        plt.legend()
        plt.savefig('cdf.jpg')

