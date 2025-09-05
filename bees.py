import json
# import matplotlib.pyplot as plt
import numpy as np
import os


def Sampling_ensemble(name, strategy):
    
    # Store the list of skywork reward differences
    rewards_real = []
    f_reward = open(f'./{name}_rm_scores.jsonl', 'r')

    for line in f_reward:
        # print(line)
        reward_data = json.loads(line)
        reward_diff = reward_data['reward_diff']
        rewards_real.append(reward_diff)

    f_reward.close()
    rewards_real = np.array(rewards_real)
    # Handle nan
    if np.isnan(rewards_real).any():
        rewards_real = np.nan_to_num(rewards_real, nan=0)
    else:
        print("There are no NaN values in the skywork reward difference matrix")


    # Store the implicit reward difference list
    rewards = []

    f_dpo = open(f'./Less-is-More/metric/dpo_2k_{name}_ppl.jsonl', 'r')
    # f_sft = open(f'./adpo/metric/sft_{name}_ppl.jsonl', 'r')

    # for dpo_line, sft_line in zip(f_dpo, f_sft):
    #     dpo_data = json.loads(dpo_line)
    #     sft_data = json.loads(sft_line)
        
    #     reward = dpo_data['reward'][0] - sft_data['reward'][0]
    #     rewards.append(reward)

    for dpo_line in f_dpo:
        dpo_data = json.loads(dpo_line)
        
        reward = dpo_data['reward'][0]
        rewards.append(reward)

    f_dpo.close()
    # f_sft.close()
    rewards = np.array(rewards)
    # Handle nan
    if np.isnan(rewards).any():
        rewards = np.nan_to_num(rewards, nan=0)
    else:
        print("There are no NaN values in the reward difference matrix")

    # clip and normalize
    margin1 = rewards
    margin2 = rewards_real
    # Find the upper limit
    max_margin1 = np.max(margin1)
    x = int(max_margin1/2)
    while True:
        samples_above = np.sum((margin1 > x) & (margin1 <= max_margin1))
        if samples_above < (max_margin1 - x) or samples_above < 30 or x > max_margin1:
            break
        x += 1
    max_margin2 = np.max(margin2)
    y = int(max_margin2/1.5)
    while True:
        samples_above = np.sum((margin2 > y) & (margin2 <= max_margin2))
        if samples_above < (max_margin2 - y) or samples_above < 30 or y > max_margin2:
            break
        y += 1
    

    if strategy == 'add':
        ml12 = min(len(margin1), len(margin2))
        margin1 = margin1[:ml12]
        margin2 = margin2[:ml12]
        
        margin = margin1 + margin2
    elif strategy == 'max':
        margin = np.maximum(margin1, margin2)
    elif strategy == 'min':
        margin = np.minimum(margin1, margin2)
    elif strategy == 'mul':
        # Clip margin to the range [-2, x]
        margin1 = np.clip(margin1, -2, x)
        margin2 = np.clip(margin2, -2, y)
        print(f"implicit reward margin clipped to range [-2, {x}]")
        print(f"skywork reward margin clipped to range [-2, {y}]")
        margin1 -= -2
        margin2 -= -2
        margin1 /= max(margin1)
        margin2 /= max(margin2)

        ml12 = min(len(margin1), len(margin2))
        margin1 = margin1[:ml12]
        margin2 = margin2[:ml12]
        
        margin = margin1 * margin2 / ((margin1 * margin2) + (1 - margin1) * (1 - margin2))


    print(margin[0], len(margin))

    # Write each number as a separate JSON object in the file
    filename = f'./Less-is-More/metric/{name}_bees_{strategy}_rewards.jsonl'
    with open(filename, "w") as file:
        for num in margin:
            json_line = json.dumps({"reward_diff": num})
            file.write(json_line + "\n")


if __name__ == '__main__':
    
    # Sampling_ensemble('hh','add')
    Sampling_ensemble('hh','mul')