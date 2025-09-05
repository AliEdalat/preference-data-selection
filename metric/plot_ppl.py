import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.style.use('classic')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

name = 'hh'

differences = []

idx = 1
with open(f'./Less-is-More/metric/sft_llama_uf_ppl.jsonl', 'r') as file:
    for line in file:
        data = json.loads(line)
        diff = data['ppl_chosen'][1]/data['ppl_chosen'][0] - data['ppl_rejected'][1]/data['ppl_rejected'][0]
        if diff < -2:
            diff = -2
        elif diff > 2:
            diff = 2
        differences.append(diff)

differences = np.array(differences)

differences = np.nan_to_num(differences, nan=0.0)

# indexs = np.random.choice(len(rewards), 3000)
# rewards = rewards[indexs]

fig, ax = plt.subplots(figsize=(5, 4))

n, bins, patches = ax.hist(differences, bins=100, density=True, alpha=0.7, 
                          color='#2E86C1', edgecolor='black', linewidth=0.3)

kde = stats.gaussian_kde(differences)
x_range = np.linspace(min(differences), max(differences), 200)
ax.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')

# mean_val = np.mean(rewards)
# median_val = np.median(rewards)
# ax.axvline(mean_val, color='green', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.3f}')
# ax.axvline(median_val, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median_val:.3f}')

ax.set_title('Distribution of Conditional PPL Margin', pad=15, fontsize=10, fontweight='bold')
ax.set_xlabel('Conditional PPL Margin', labelpad=9)
ax.set_ylabel('Density', labelpad=9)

ax.grid(True, alpha=0.3, linestyle='--')

ax.legend(frameon=True, fancybox=True, shadow=True)

plt.tight_layout()

plt.savefig(f'./Less-is-More/metric/figure/hh_sft_ppl_distribution.pdf', 
            bbox_inches='tight')
plt.close()



# import json
# import matplotlib.pyplot as plt
# import numpy as np


# name = 'uf'
# differences = []

# idx = 1

# with open(f'./adpo/metric/sft_{name}_ppl.jsonl', 'r') as file:
#     for line in file:
#         data = json.loads(line)
#         diff = data['ppl_chosen'][1]/data['ppl_chosen'][0] - data['ppl_rejected'][1]/data['ppl_rejected'][0]
#         if diff < -2:
#             diff = -2
#         elif diff > 2:
#             diff = 2
#         differences.append(diff)

# differences = np.array(differences)


# plt.figure(figsize=(10, 6))
# plt.hist(differences, bins=100)
# plt.title('Distribution of conditional PPL margin')
# plt.xlabel('PPL margin')
# plt.ylabel('Frequency')


# plt.grid(True, alpha=0.3)


# plt.savefig(f'./adpo/metric/{name}_sft_ppl_distribution.png')
# plt.close()

# print(f"Number of samples: {len(differences)}")
