import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datasets import load_from_disk
name = 'llama_uf_armo'

def load_rewards(filename):
    rewards = []
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            rewards.append(data['reward_diff'] if 'reward_diff' in data 
                         else data['reward'][0])
    return np.array(rewards)


rewards_skywork = load_rewards(f'./adpo/metric/{name}_rm_scores.jsonl')


ds = load_from_disk('dataset/llama3-ultrafeedback-armo')['train']
rewards = np.zeros(len(ds))
for i in range(len(ds)):
    scores = ds[i]['all_rm_scores']
    rewards[i] = max(scores) - min(scores)


rewards_skywork = np.nan_to_num(rewards_skywork, nan=0.5)
rewards = np.nan_to_num(rewards, nan=0)


rewards_skywork = rewards_skywork[::10]
rewards = rewards[::10]


plt.style.use('classic')
plt.figure(figsize=(6, 4))


scatter = plt.scatter(rewards_skywork, rewards, 
                     alpha=0.5,  
                     s=20,      
                     c=np.abs(rewards_skywork - rewards),  
                     cmap='viridis')


plt.colorbar(scatter, label='Absolute Difference')


z = np.polyfit(rewards_skywork, rewards, 1)
p = np.poly1d(z)
plt.plot(rewards_skywork, p(rewards_skywork), "r--", alpha=0.8, label=f'Trend line')


plt.xlabel('Skywork Reward Margin', fontsize=17)
plt.ylabel('Armorm Reward Margin', fontsize=17)
# plt.title('External vs DPO Implicit Reward Margin Distribution', fontsize=16)


plt.grid(True, alpha=0.3, linestyle='--')


plt.legend()

plt.xlim(-40, 60)
plt.ylim(0, 0.2)


plt.axhline(y=0, color='k', linestyle=':', alpha=0.3)
plt.axvline(x=0, color='k', linestyle=':', alpha=0.3)


plt.tight_layout() 
plt.savefig(f'./adpo/metric/figure/{name}_DM_distribution.pdf', bbox_inches='tight')
plt.close()
