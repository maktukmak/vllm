import torch
import time
import numpy as np
import matplotlib.pyplot as plt

n_head = 12
n_seq = 2048
n_dim = 64
device = "cpu"

n_batch_list = [1, 2, 4, 8, 16, 32, 64, 128]
res_list = []

for n_batch in n_batch_list:
    res = []
    for _ in range(20):
        query = torch.rand((n_batch, n_head, n_seq // n_batch, n_dim)).to(device)
        key = torch.rand((n_batch, n_head, n_dim, n_seq // n_batch)).to(device)

        st = time.time()
        attn = torch.matmul(query, key)
        end = time.time()
        res.append(end - st)

    print(np.mean(res[3:]))
    res_list.append(np.mean(res[3:]))


plt.plot(n_batch_list, res_list, marker='o', label=device)
plt.xlabel('Batch size')
plt.ylabel('Turnaround (s)')
plt.ylim(0, max(res_list) + 0.1*max(res_list))
plt.xscale('log', base=2)
plt.legend()
plt.grid()
plt.savefig('gemm_' + device + '.jpg')

