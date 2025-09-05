import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.style.use('classic')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

name = 'hh'
rewards = []

f_dpo = open(f'./Less-is-More/metric/dpo_2k_hh_ppl.jsonl', 'r')
for dpo_line in f_dpo:
    dpo_data = json.loads(dpo_line)
    reward = dpo_data['reward'][0]
    rewards.append(reward)
f_dpo.close()

print(min(rewards), max(rewards))

def normalize(arr):
    arr_min = min(arr)
    arr_max = max(arr)
    print(arr_min, arr_max)
    arra = np.array(arr)
    # Normalize to [-2, 2]
    return 4 * ((arra - arr_min) / (arr_max - arr_min)) - 2

rewards = normalize(rewards)
rewards = np.nan_to_num(rewards, nan=0.0)
print(min(rewards), max(rewards))

fig, ax = plt.subplots(figsize=(5, 4))
n, bins, patches = ax.hist(rewards, bins=100, density=True, alpha=0.7, 
                          color='#2E86C1', edgecolor='black', linewidth=0.3)

kde = stats.gaussian_kde(rewards)
x_range = np.linspace(min(rewards), max(rewards), 200)
ax.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')

ax.set_title('Distribution of Conditional PPL Margin', pad=15, fontsize=10, fontweight='bold')
ax.set_xlabel('Conditional PPL Margin', labelpad=9)
ax.set_ylabel('Density', labelpad=9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig(f'./Less-is-More/metric/figure/hh_ifd_distribution.pdf', bbox_inches='tight')
plt.close()