import torch
import time
import numpy as np

n_head = 12
n_seq = 2048
n_dim = 64

n_batch_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]

for n_batch in n_batch_list:
    res = []
    for _ in range(20):
        query = torch.rand((n_batch, n_head, n_seq // n_batch, n_dim))
        key = torch.rand((n_batch, n_head, n_dim, n_seq // n_batch))

        st = time.time()
        attn = torch.matmul(query, key)
        end = time.time()
        res.append(end - st)

    print(np.mean(res[3:]))