import matplotlib.pyplot as plt
import numpy as np
import math
import csv

#plt.rcParams['font.serif'] = "Times New Roman"
#plt.rcParams['font.family'] = "serif"
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

c1 = (31.0/255, 119/255.0, 180/255.0)
c2 = (174/255.0, 199/255.0, 232/255.0)
c3 = (1, 127/255.0, 14/255.0)
c4 = (1, 187/255.0, 120/255.0)
c5 = (44/255.0, 160/255.0, 44/255.0)
c6 = (152/255.0, 223/255.0, 138/255.0)
c7 = (214/255.0, 39/255.0, 40/255.0)
c8 = (1, 152/255.0, 150/255.0)
c9 = (148/255.0, 103/255.0, 189/255.0)
c10 = (192/255.0, 176/255.0, 213/255.0)

models = ['Qwen3-30B-A3B']
models_display = ['Qwen3-30B-A3B']
systems = ['vllm', 'sglang', 'pytorch+mpk']

results = dict()
with open('MoE.csv', 'r') as csvfile:
    data = csv.DictReader(csvfile)
    for row in data:
        model = row['model']
        batch_size = int(row['BS'])
        system = row['Impl']
        results[(model, batch_size, system)] = float(row['time'])


for system in systems:
    system_data = []
    for idx, model in enumerate(models):
        model_data = []
        for batch_size in [1, 2, 4, 8, 16]:
            model_data.append(results[(model, batch_size, system)])
        system_data.append(model_data)      
        results[system] = system_data          

vllm = results['vllm']
sglang = results['sglang']
mpk = results['pytorch+mpk']
# triton = results['triton']
# taso = results['taso']
# miso = results['mirage']

def autolabel(ax, rects, num, color):
        # attach some text labels
        i = 0
        for rect in rects:
            height = ax.get_ylim()[1]
            ax.text(rect.get_x(), height, '%.2lfx' % num[i], color=color, fontsize=12, ha='center', va='bottom')
            i = i + 1
width = 0.15
x = [0, 1, 2, 3, 4]
title = models_display 
systems = ['vLLM', 'SGLang', 'PyTorch + TGX']

fig, axes = plt.subplots(ncols = 1, nrows = 1, figsize=(15, 6), constrained_layout=True, squeeze=False)
# fig.tight_layout()
print("axes:", axes)
idx = 0

print("vllm:", vllm)
for i in range(len(axes)):
    for j in range(len(axes[i])):
        ax = axes[i][j]
        # ax = axes[i]
        print("idx:", idx)
        baseline = np.minimum(np.array(vllm[idx]), np.array(sglang[idx]))
        # baseline = np.minimum(baseline, np.array(sglang[idx]))
        # baseline = np.minimum(baseline, np.array(flashattn[idx]))
        # baseline = np.minimum(baseline, np.array(triton[idx]))
        # print(baseline, np.array(miso[idx]))
        opt = np.minimum(baseline, np.array(mpk[idx]))
        print("vllm[idx]:", vllm[idx])
        print("sglang[idx]:", sglang[idx])
        print("mpk:", mpk[idx])
        # print("baseline:", baseline)
        b0 = ax.bar(np.array(x)-1.5*width, opt / np.array(vllm[idx]), width, color = c8, edgecolor="white")
        b1 = ax.bar(np.array(x)-0.5*width, opt / np.array(sglang[idx]), width, color = c6, edgecolor = "white")
        # b2 = ax.bar(np.array(x)-0.5*width, opt / np.array(tensorrt[idx]), width, color = c2, edgecolor = "white")
        # b3 = ax.bar(np.array(x)+0.5*width, opt / np.array(pytorch[idx]), width, color = c10, edgecolor="white")
        # b4 = ax.bar(np.array(x)+1.5*width, opt / np.array(triton[idx]), width, color = c4, edgecolor="white")
        b5 = ax.bar(np.array(x)+0.5*width, opt / np.array(mpk[idx]), width, color = c3, edgecolor="white")
        #axes[i].axhline(y=metaflow[i], color = 'r', xmin = 3.5/7.5, xmax = 1, lw=2)
        ax.set_xlabel(title[idx], fontsize = 14)
        ax.set_xlim(-3*width, max(x) + 3*width)
        ax.tick_params(axis='both', which='major', labelsize=12)
        #axes[i].set_xticklabels(['A','B','C','D','E','F','G'], fontsize=12)
        autolabel(ax, b5, baseline / np.array(mpk[idx]), c3)
        idx += 1

#width2 = 0.23
#b3 = axes[2].bar(np.array(x) - width2, np.array(ft[2]), width2, color = c10, edgecolor="white")
#b4 = axes[2].bar(np.array(x), np.array(inc_decoding[2]), width2, color = c4, edgecolor="white")
#b5 = axes[2].bar(np.array(x) + width2, np.array(spec_infer[2]), width2, color = c3, edgecolor="white")
#axes[2].set_xlabel("LLaMA-65B\n(4 GPUs/node, 2 nodes)", fontsize=14)
#axes[2].set_xlim(-2*width2, 4+2*width2)
#axes[2].tick_params(axis='both', which='major', labelsize=12)
#autolabel(axes[2], b5, np.array(ft[2]) / np.array(spec_infer[2]), c3)

print(b0)
print(b1)
print(b5)

#plt.xticks(np.array(x) + 1.5 * width, ('GCN', 'GIN', 'GAT'))
plt.setp(axes, xticks=[0,1,2,3,4], xticklabels=['BS=1', 'BS=2', 'BS=4', 'BS=8','BS=16'])
fig.text(-0.032, 0.5, 'Relative Performance', fontweight='bold', ha='left', va='center', rotation='vertical', fontsize=14)
# fig.text(-0.01, 0.85, 'A100', ha='left', va='center', rotation='vertical', fontsize=14)
# fig.text(-0.01, 0.51, 'H100', ha='left', va='center', rotation='vertical', fontsize=14)
fig.text(-0.01, 0.18, 'B200', ha='left', va='center', rotation='vertical', fontsize=14)

#fig.text(0.5, 0.02, 'Number of GPU devices', fontweight='bold', ha='center', va='bottom',  fontsize=18)
# fig.legend([b0, b1, b2, b3, b4, b5], systems, loc = 'upper center', fontsize = 14, ncol = 6, bbox_to_anchor=(0.5,1.15))
fig.legend([b0, b1, b5], systems, loc = 'upper center', fontsize = 20, ncol = 6, bbox_to_anchor=(0.5,1.15), prop={'size': 24})

#save to png
plt.savefig('benchmark_moe.pdf', bbox_inches='tight', dpi=100)
# plt.show()
