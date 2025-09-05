import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.style.use('classic')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

name = 'uf'

rewards = []

# dpo_rewards = load_rewards(f'./Less-is-More/metric/dpo_2k_llama_uf_ppl.jsonl')
# sft_rewards = load_rewards(f'./Less-is-More/metric/sft_llama_uf_ppl.jsonl')

f_dpo = open(f'./Less-is-More/metric/dpo_2k_llama_uf_ppl.jsonl', 'r')
f_sft = open(f'./Less-is-More/metric/sft_llama_uf_ppl.jsonl', 'r')

for dpo_line, sft_line in zip(f_dpo, f_sft):
    dpo_data = json.loads(dpo_line)
    sft_data = json.loads(sft_line)
    
    reward = dpo_data['reward'][0] - sft_data['reward'][0]
    rewards.append(reward)

f_dpo.close()
f_sft.close()

# def load_rewards(filename):
#     rewards = []
#     with open(filename, 'r') as f:
#         for line in f:
#             # print(line)
#             try:
#                 data = json.loads(line)
#                 rewards.append(data['reward_diff'] if 'reward_diff' in data 
#                              else data['reward'][0])
#             except:
#               print("An exception occurred") 
#     return np.array(rewards)

# rewards = load_rewards(f'./Less-is-More/metric/uf_imp_rm_scores.jsonl')

rewards = np.array(rewards)

rewards = np.nan_to_num(rewards, nan=0.0)

# indexs = np.random.choice(len(rewards), 3000)
# rewards = rewards[indexs]

fig, ax = plt.subplots(figsize=(5, 4))

n, bins, patches = ax.hist(rewards, bins=100, density=True, alpha=0.7, 
                          color='#2E86C1', edgecolor='black', linewidth=0.3)

kde = stats.gaussian_kde(rewards)
x_range = np.linspace(min(rewards), max(rewards), 200)
ax.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')

# mean_val = np.mean(rewards)
# median_val = np.median(rewards)
# ax.axvline(mean_val, color='green', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.3f}')
# ax.axvline(median_val, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median_val:.3f}')

ax.set_title('Distribution of Implicit Reward Margin', pad=15, fontsize=10, fontweight='bold')
ax.set_xlabel('Implicit Reward Margin', labelpad=9)
ax.set_ylabel('Density', labelpad=9)

ax.grid(True, alpha=0.3, linestyle='--')

ax.legend(frameon=True, fancybox=True, shadow=True)

plt.tight_layout()

plt.savefig(f'./Less-is-More/metric/figure/uf_imp_reward_distribution.pdf', 
            bbox_inches='tight')
plt.close()