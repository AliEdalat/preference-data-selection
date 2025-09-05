import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

name = 'hh'


def load_rewards(filename):
    rewards = []
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            rewards.append(data['reward_diff'] if 'reward_diff' in data 
                         else data['reward'][0])
    return np.array(rewards)


s_dpo_rewards = load_rewards(f'./adpo/metric/1b-result/dpo_5k_{name}_ppl.jsonl')
s_sft_rewards = load_rewards(f'./adpo/metric/1b-result/sft_{name}_ppl.jsonl')
s_rewards = s_dpo_rewards - s_sft_rewards


dpo_rewards = load_rewards(f'./adpo/metric/dpo_2k_{name}_ppl.jsonl')
sft_rewards = load_rewards(f'./adpo/metric/sft_{name}_ppl.jsonl')
rewards = dpo_rewards - sft_rewards


s_rewards = np.nan_to_num(s_rewards, nan=0)
rewards = np.nan_to_num(rewards, nan=0)


s_rewards = s_rewards[::20]
rewards = rewards[::20]


plt.style.use('classic')
plt.figure(figsize=(6, 4))


scatter = plt.scatter(s_rewards, rewards, 
                     alpha=0.5,  
                     s=20,      
                     c=np.abs(s_rewards - rewards),  
                     cmap='viridis')


plt.colorbar(scatter, label='Absolute Difference')


z = np.polyfit(s_rewards, rewards, 1)
p = np.poly1d(z)
plt.plot(s_rewards, p(s_rewards), "r--", alpha=0.8, label=f'Trend line')


plt.xlabel('1B Implicit Reward Margin', fontsize=17)
plt.ylabel('3B Implicit Reward Margin', fontsize=17)
# plt.title('1B vs 3B DPO Implicit Reward Margin Distribution', fontsize=16)


plt.grid(True, alpha=0.3, linestyle='--')


plt.legend()

plt.xlim(-40, 60)
plt.ylim(-40, 60)


plt.axhline(y=0, color='k', linestyle=':', alpha=0.3)
plt.axvline(x=0, color='k', linestyle=':', alpha=0.3)


plt.tight_layout()  
plt.savefig(f'./adpo/metric/figure/{name}_DIM_distribution.png', bbox_inches='tight')
plt.close()
