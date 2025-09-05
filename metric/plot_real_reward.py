import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.style.use('classic')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

name = 'uf'

rewards = []

# f_reward = open(f'./Less-is-More/metric/llama_uf_rm_scores.jsonl', 'r')
# f_reward = open(f'./Less-is-More/metric/hh_rm_scores.jsonl', 'r')
# f_reward = open(f'./Less-is-More/metric/hh_imp_rm_scores_2.jsonl', 'r')
# f_reward = open(f'./Less-is-More/metric/hh_bees_mul_rewards.jsonl', 'r')
f_reward = open(f'./Less-is-More/metric/uf_bees_mul_rewards.jsonl', 'r')






for line in f_reward:
    try:
        reward_data = json.loads(line)
        reward_diff = reward_data['reward_diff']
        rewards.append(reward_diff)
    except:
        print('!')

f_reward.close()

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

# ax.set_title('Distribution of External Reward Margin', pad=15, fontsize=10, fontweight='bold')
# ax.set_xlabel('External Reward Margin', labelpad=9)
ax.set_title('Distribution of BeeS (mul) Reward Margin', pad=15, fontsize=10, fontweight='bold')
ax.set_xlabel('BeeS (mul) Reward Margin', labelpad=9)
ax.set_ylabel('Density', labelpad=9)

ax.grid(True, alpha=0.3, linestyle='--')

ax.legend(frameon=True, fancybox=True, shadow=True)

plt.tight_layout()

plt.savefig(f'./Less-is-More/metric/figure/uf_bees_mul_reward_distribution.pdf', 
            bbox_inches='tight')
plt.close()



# import json
# import matplotlib.pyplot as plt
# import numpy as np

# name = 'llama_uf'

# rewards = []

# f_reward = open(f'./adpo/metric/{name}_rm_scores.jsonl', 'r')


# for line in f_reward:
#     reward_data = json.loads(line)
#     reward_diff = reward_data['reward_diff']
#     rewards.append(reward_diff)

# f_reward.close()

# rewards = np.array(rewards)

# plt.figure(figsize=(10, 6))
# plt.hist(rewards, bins=100)
# plt.title('Distribution of reward margin')
# plt.xlabel('reward margin')
# plt.ylabel('Frequency')

# plt.grid(True, alpha=0.3)

# plt.savefig(f'./adpo/metric/{name}_skywork_reward_distribution.png')
# plt.close()

# print(f"Number of samples: {len(rewards)}")