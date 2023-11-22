import torch
import time
import numpy as np
import matplotlib.pyplot as plt

n_head = 12
n_seq = 2048
n_dim = 64
device = "cpu"
exp = 100

n_batch_list = [1, 2, 4, 8, 16, 32, 64, 128]
res_list_batched_mm = []
res_list_single_mm = []

for n_batch in n_batch_list:
    res = []
    for _ in range(exp):
        query = torch.rand((n_batch, n_head, n_seq // n_batch, n_dim)).to(device)
        key = torch.rand((n_batch, n_head, n_dim, n_seq // n_batch)).to(device)

        st = time.time()
        attn = torch.matmul(query, key)
        end = time.time()
        res.append(end - st)

    print(np.mean(res[3:]))
    res_list_batched_mm.append(np.mean(res[3:]))

    res = []
    for _ in range(exp):
        query = torch.rand((1, n_head, n_seq // n_batch, n_dim)).to(device)
        key = torch.rand((1, n_head, n_dim, n_seq // n_batch)).to(device)

        st = time.time()
        for _ in range(n_batch):
            attn = torch.matmul(query, key)
        end = time.time()
        res.append(end - st)

    print(np.mean(res[3:]))
    res_list_single_mm.append(np.mean(res[3:]))


plt.plot(n_batch_list, res_list_batched_mm, marker='o', label=device + '-batched')
plt.plot(n_batch_list, res_list_single_mm, marker='o', label=device + '-nonbatched')
plt.xlabel('Batch size')
plt.ylabel('Turnaround (s)')
max_val = max(max(res_list_batched_mm), max(res_list_single_mm))
plt.ylim(0, max_val + 0.1*max_val)
plt.xscale('log', base=2)
plt.legend()
plt.grid()
plt.savefig('gemm_' + device + '.jpg')

