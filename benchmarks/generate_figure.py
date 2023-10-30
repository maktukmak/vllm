import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys

with open('d_res.pickle', 'rb') as handle:
    d_res = pickle.load(handle)



prompt_lens = list(set([key[0] for key in d_res.keys()]))
prompt_lens.sort()
batch_size = int(sys.argv[1])



barWidth = 0.25

br1 = np.arange(len(prompt_lens)) 
br2 = [x + barWidth for x in br1] 


single = np.array([d_res[(len, 1)] for len in prompt_lens])
dynamic = np.array([d_res[(len, batch_size)] for len in prompt_lens])
speedup = single / dynamic


plt.bar(br1, single, color ='r', width = barWidth, 
        edgecolor ='grey', label ='Single') 
plt.bar(br2, dynamic, color ='g', width = barWidth, 
        edgecolor ='grey', label ='Dynamic') 

plt.xlabel('Prompt len', fontweight ='bold', fontsize = 15) 
plt.ylabel('Latency', fontweight ='bold', fontsize = 15) 
plt.xticks([r + barWidth/2 for r in range(len(prompt_lens))], 
        prompt_lens)
plt.title('Batch size: ' + str(batch_size) )


for i in br1:
    plt.text(i, max(single[i], dynamic[i]) + 0.2, "{:.2f}".format(speedup[i]) + "X")

plt.legend()
plt.savefig('res_' + str(batch_size) + '.jpg')
