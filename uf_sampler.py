import json
# import matplotlib.pyplot as plt
import numpy as np
import os


def Sampling(name, ds, strategy, part, Num):
    if strategy == 'random':
        if Num == -1:
            return ds.shuffle(seed=42)
        else:
            return ds.shuffle(seed=42).select(range(Num))
    if strategy == 'orth':

        rewards_real = []
        # f_reward = open(f'./Less-is-More/metric/{name}_rm_scores.jsonl', 'r')
        f_reward = open(f'./Less-is-More/metric/uf_orthogonal_fused_rewards.jsonl', 'r')

        for line in f_reward:
            reward_data = json.loads(line)
            reward_diff = reward_data['fused_reward']
            rewards_real.append(reward_diff)

        f_reward.close()
        rewards_real = np.array(rewards_real)
        # Handle nan
        if np.isnan(rewards_real).any():
            rewards_real = np.nan_to_num(rewards_real, nan=0)
        else:
            print("There are no NaN values in the skywork reward difference matrix")

        margin2 = rewards_real
        # margin = margin1 + margin2
        margin = margin2
    if strategy == 'att':

        rewards_real = []
        # f_reward = open(f'./Less-is-More/metric/{name}_rm_scores.jsonl', 'r')
        f_reward = open(f'./Less-is-More/metric/uf_attention_fused_rewards.jsonl', 'r')

        for line in f_reward:
            reward_data = json.loads(line)
            reward_diff = reward_data['fused_reward']
            rewards_real.append(reward_diff)

        f_reward.close()
        rewards_real = np.array(rewards_real)
        # Handle nan
        if np.isnan(rewards_real).any():
            rewards_real = np.nan_to_num(rewards_real, nan=0)
        else:
            print("There are no NaN values in the skywork reward difference matrix")

        margin2 = rewards_real
        # margin = margin1 + margin2
        margin = margin2
    if strategy == 'in_reward_margin':
        # Use the score given by the reward model of the dataset itself to calculate the margin
        # score_list = np.zeros(len(ds))
        # for i in range(len(ds)):
        #     scores = ds[i]['all_rm_scores']
        #     score_list[i] = max(scores) - min(scores)
        # margin1 = score_list * 150

        rewards_real = []
        # f_reward = open(f'./Less-is-More/metric/{name}_rm_scores.jsonl', 'r')
        # f_reward = open(f'./Less-is-More/metric/hh_imp_rm_scores_2.jsonl', 'r')
        f_reward = open(f'./Less-is-More/metric/uf_imp_rm_scores.jsonl', 'r')
        # sft_llama_uf_ppl

        for line in f_reward:
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

        margin2 = rewards_real
        # margin = margin1 + margin2
        margin = margin2


    elif strategy == 'ex_reward_margin':
        # Store the list of skywork reward differences
        rewards_real = []
        # f_reward = open(f'./Less-is-More/metric/{name}_rm_scores.jsonl', 'r')
        # f_reward = open(f'./llama_uf_rm_scores.jsonl', 'r')
        # f_reward = open(f'./hh_rm_scores.jsonl', 'r')
        f_reward = open(f'./Less-is-More/metric/llama_uf_rm_scores.jsonl', 'r')

        for line in f_reward:
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

        margin = rewards_real
        # Set edge value
        mid_edge = 1
        
    elif strategy == 'im_reward_margin':
        # Store the implicit reward difference list
        rewards = []

        f_dpo = open(f'./Less-is-More/metric/dpo_2k_{name}_ppl.jsonl', 'r')
        f_sft = open(f'./Less-is-More/metric/sft_{name}_ppl.jsonl', 'r')

        for dpo_line, sft_line in zip(f_dpo, f_sft):
            dpo_data = json.loads(dpo_line)
            sft_data = json.loads(sft_line)
            
            reward = dpo_data['reward'][0] - sft_data['reward'][0]
            rewards.append(reward)

        f_dpo.close()
        f_sft.close()
        rewards = np.array(rewards)
        # Handle nan
        if np.isnan(rewards).any():
            rewards = np.nan_to_num(rewards, nan=0)
        else:
            print("There are no NaN values in the reward difference matrix")

        margin = rewards
        # Set edge value
        mid_edge = 1
    elif strategy == 'ppl_margin':
        # List to store the differences
        differences = []

        idx = 1
        # Read jsonl file dpo_2k_hh_ppl sft_{name}_ppl.jsonl
        # ./Less-is-More/metric/dpo_2k_hh_ppl.jsonl
        # dpo_2k_llama_uf_ppl
        with open(f'./Less-is-More/metric/dpo_2k_llama_uf_ppl.jsonl', 'r') as file:
            for line in file:
                data = json.loads(line)
                # print(data)
                # print(data['ppl_chosen'][1],data['ppl_chosen'][0],data['ppl_rejected'][1],data['ppl_rejected'][0])
                # e = 0.00001 [0]
                e = 0.0000000001
                diff = data['ppl_chosen'][1]/(data['ppl_chosen'][0]+e) - data['ppl_rejected'][1]/(data['ppl_rejected'][0]+e)
                # Handle these abnormal samples so that they are not sampled
                if diff < -2:
                    diff = -0.25
                elif diff > 2:
                    diff = 0.25
                differences.append(diff)

        differences = np.array(differences)
        # Handle nan
        if np.isnan(differences).any():
            differences = np.nan_to_num(differences, nan=0.3)
        else:
            print("There are no NaN values in the ppl difference matrix")

        margin = differences
        # Set edge value
    mid_edge = 0.1

    # Select samples according to margin and partition
    if part == 'top':
        indices = np.argsort(margin)[-Num:]
    elif part == 'mid':
        middle_mask = (margin >= -mid_edge) & (margin <= mid_edge)
        indices = np.where(middle_mask)[0]

        # If the number of samples in the interval [-mid_edge, mid_edge] exceeds Num, randomly sample Num
        if len(indices) > Num:
            indices = np.random.choice(indices, size=Num, replace=False)
        elif len(indices) < Num:
            print(f"Warning: Only {len(indices)} samples found in middle range")
    elif part == 'bot':
        indices = np.argsort(margin)[:Num]
    else:
        print("Invalid part")
        return ds

    return ds.select(indices)

if __name__ == '__main__':
    # from datasets import load_dataset, DatasetDict, concatenate_datasets
    # data_dir = '/home/jovyan/llm_project/datasets/hh-rlhf/helpful-base'
    # ds1 = load_dataset('json', data_files={'train':os.path.join(data_dir,'train.jsonl.gz'),'test': os.path.join(data_dir, 'test.jsonl.gz')})['train']
    # data_dir = '/home/jovyan/llm_project/datasets/hh-rlhf/harmless-base'
    # ds2 = load_dataset('json', data_files={'train':os.path.join(data_dir,'train.jsonl.gz'),'test': os.path.join(data_dir, 'test.jsonl.gz')})['train']
    # ds = concatenate_datasets([ds1,ds2])
    # # ds_tr = Sampling('hh', ds, 'ex_reward_margin', 'top', 100)
    # # ds_tr = Sampling('hh', ds, 'random', 'top', 2000)
    # ds_tr = Sampling('hh', ds, 'random', 'mid', -1)
    # print(ds_tr)

    # # Save sample to JSONL
    # output_path = '/home/jovyan/llm_project/datasets/hh_full.jsonl'
    # ds_tr.to_json(output_path, orient='records', lines=True)

    from datasets import load_dataset, DatasetDict, concatenate_datasets
    data_dir = '/home/jovyan/llm_project/datasets/UF_DPO/data'
    ds = load_dataset('json', data_files={'train':os.path.join(data_dir,'train.jsonl'),'test': os.path.join(data_dir, 'test.jsonl')})['train']
    # ds_tr = Sampling('hh', ds, 'ex_reward_margin', 'top', 100)
    # ds_tr = Sampling('hh', ds, 'random', 'top', 2000)
    # ds_tr = Sampling('uf', ds, 'random', 'top', -1)
    # hh ex_reward_margin for uf
    # in_reward_margin
    # ds_tr = Sampling('hh', ds, 'ppl_margin', 'bot', 2000)
    # ds_tr = Sampling('hh', ds, 'in_reward_margin', 'top', 2000)
    # ds_tr = Sampling('hh', ds, 'orth', 'top', 2000)
    ds_tr = Sampling('hh', ds, 'random', 'top', 20000)
    print(ds_tr)

    # Save sample to JSONL
    output_path = '/home/jovyan/llm_project/datasets/uf_orth_P.jsonl'
    ds_tr.to_json(output_path, orient='records', lines=True)

    # data_dir = '/home/jovyan/llm_project/datasets/hh-rlhf/helpful-base'
    # ds1 = load_dataset('json', data_files={'train':os.path.join(data_dir,'train.jsonl.gz'),'test': os.path.join(data_dir, 'test.jsonl.gz')})['train']
    # data_dir = '/home/jovyan/llm_project/datasets/hh-rlhf/harmless-base'
    # ds2 = load_dataset('json', data_files={'train':os.path.join(data_dir,'train.jsonl.gz'),'test': os.path.join(data_dir, 'test.jsonl.gz')})['train']
    # ds = concatenate_datasets([ds1,ds2])

    # folder_path = "/home/jovyan/llm_project/datasets/ultrafeedback_binarized/data"

    # # Load the dataset from the local folder
    # dataset = load_dataset("parquet", data_files=f"{folder_path}/*.parquet")['train']
    
    # # ds_tr = Sampling('uf', dataset, 'random', 'top', 2000)
    # # print(ds_tr)

    # ds_tr = Sampling('uf', dataset, 'ex_reward_margin', 'bot', 2000)
    # print(ds_tr)

    # # Save sample to JSONL
    # output_path = '/home/jovyan/llm_project/datasets/uf_ex_reward_margin_sample_Z.jsonl'
    # ds_tr.to_json(output_path, orient='records', lines=True)
    
    print(f"Saved {len(ds_tr)} samples to {output_path}")