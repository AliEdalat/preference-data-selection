import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

name = 'mistral_uf'


def load_rewards(filename):
    rewards = []
    with open(filename, 'r') as f:
        for line in f:
            # print(line)
            try:
                data = json.loads(line)
                rewards.append(data['reward_diff'] if 'reward_diff' in data 
                             else data['reward'][0])
            except:
              print("An exception occurred") 
    return np.array(rewards)


# rewards1, rewards2 = fusion.load_rewards(
    #     '/home/jovyan/llm_project/Less-is-More/metric/hh_imp_rm_scores_2.jsonl', 
    #     '/home/jovyan/llm_project/Less-is-More/metric/dpo_2k_hh_ppl.jsonl'
    # )



# rewards_real = load_rewards(f'./Less-is-More/metric/hh_rm_scores.jsonl')
# rewards_real = load_rewards(f'./Less-is-More/metric/llama_uf_rm_scores.jsonl')


# dpo_rewards = load_rewards(f'./Less-is-More/metric/dpo_2k_llama_uf_ppl.jsonl')
# sft_rewards = load_rewards(f'./Less-is-More/metric/sft_llama_uf_ppl.jsonl')
# rewards = dpo_rewards - sft_rewards

# rewards = load_rewards(f'./Less-is-More/metric/dpo_2k_hh_ppl.jsonl')

rewards_real = load_rewards(f'./Less-is-More/metric/uf_imp_rm_scores.jsonl')
rewards = load_rewards(f'./Less-is-More/metric/dpo_2k_llama_uf_ppl.jsonl')


rewards_real = np.nan_to_num(rewards_real, nan=0.5)
rewards = np.nan_to_num(rewards, nan=0)


# rewards_real = rewards_real[::20]
# rewards = rewards[::20]

xlen = min(rewards_real.shape[0],rewards.shape[0])
rewards_real = rewards_real[:xlen:20]
rewards = rewards[:xlen:20]

def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

rewards_real = normalize(rewards_real)
rewards = normalize(rewards)


plt.style.use('classic')
plt.figure(figsize=(6, 4))


scatter = plt.scatter(rewards_real, rewards, 
                     alpha=0.5,  
                     s=20,      
                     c=np.abs(rewards_real - rewards),  
                     cmap='viridis')


plt.colorbar(scatter, label='Absolute Difference')


z = np.polyfit(rewards_real, rewards, 1)
p = np.poly1d(z)
plt.plot(rewards_real, p(rewards_real), "r--", alpha=0.8, label=f'Trend line')


plt.xlabel('External Reward Margin', fontsize=17)
plt.ylabel('Implicit Reward Margin', fontsize=17)
# plt.title('External vs DPO Implicit Reward Margin Distribution', fontsize=16)


plt.grid(True, alpha=0.3, linestyle='--')


plt.legend()


plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)


plt.axhline(y=0, color='k', linestyle=':', alpha=0.3)
plt.axvline(x=0, color='k', linestyle=':', alpha=0.3)


plt.tight_layout()  
plt.savefig(f'./Less-is-More/metric/figure/uf_ifd_imp_distribution.pdf', bbox_inches='tight')
plt.close()
