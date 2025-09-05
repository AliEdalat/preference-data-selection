import os
from typing import Any, List, Literal, Optional
from dataclasses import dataclass, field
# import wandb
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, TrainingArguments
import torch
from trl import DPOTrainer, DPOConfig
# from less_is_more.sampler import Sampling
# from less_is_more.m_sampler import Sampling_ensemble


# os.environ['WANDB_PROJECT'] = "adpo-final"

import json
# import matplotlib.pyplot as plt
import numpy as np


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
    if strategy == 'in_reward_margin':
        # Use the score given by the reward model of the dataset itself to calculate the margin
        score_list = np.zeros(len(ds))
        for i in range(len(ds)):
            scores = ds[i]['all_rm_scores']
            score_list[i] = max(scores) - min(scores)
        margin1 = score_list * 150

        rewards_real = []
        f_reward = open(f'./Less-is-More/metric/{name}_rm_scores.jsonl', 'r')

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
        margin = margin1 + margin2


    elif strategy == 'ex_reward_margin':
        # Store the list of skywork reward differences
        rewards_real = []
        f_reward = open(f'./Less-is-More/metric/{name}_rm_scores.jsonl', 'r')

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
        # Read jsonl file
        with open(f'./Less-is-More/metric/sft_{name}_ppl.jsonl', 'r') as file:
            for line in file:
                data = json.loads(line)
                diff = data['ppl_chosen'][1]/data['ppl_chosen'][0] - data['ppl_rejected'][1]/data['ppl_rejected'][0]
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


def Sampling_ensemble(name, ds, strategy, part, Num):
    
    # Store the list of skywork reward differences
    rewards_real = []
    f_reward = open(f'./Less-is-More/metric/{name}_rm_scores.jsonl', 'r')

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
        margin = margin1 * margin2 / ((margin1 * margin2) + (1 - margin1) * (1 - margin2))


    # Select samples according to margin and partition
    if part == 'top':
        indices = np.argsort(margin)[-Num:]
    if np.isnan(margin).any():
        nan_indices = np.where(np.isnan(margin))[0]
        print(f"NaN indices: {nan_indices}")
        

    return ds.select(indices)


def is_openai_format(messages: Any) -> bool:
    """
    Check if the input messages are in OpenAI format.
    Args:
        messages (`Any`):
            Messages to check.
    Returns:
        `bool`: Whether the messages are in OpenAI format.
    """
    if isinstance(messages, list) and all(isinstance(message, dict) for message in messages):
        return all("role" in message and "content" in message for message in messages)
    return False

def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]

def apply_chat_template(example, tokenizer, name):
    if 'uf' in name:
        return apply_chat_template_uf(example, tokenizer)
    elif 'tldr' in name:
        return apply_chat_template_tldr(example, tokenizer)
    elif 'hh' in name:
        return apply_chat_template_hh(example, tokenizer)
    else:
        raise ValueError(f"Unknown dataset name: {name}")


def apply_chat_template_uf(
    example,
    tokenizer
):

    if all(k in example.keys() for k in ("chosen", "rejected")):
        if not is_openai_format(example["chosen"]) or not is_openai_format(example["rejected"]):
            raise ValueError(
                f"Could not format example as dialogue for the task! Require OpenAI format for all messages"
            )

        # For DPO/ORPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
        # We therefore need to extract the N-1 turns to form the prompt (for multi-turn dialogues)
        prompt_messages = example["chosen"][:-1]
        # Now we extract the final turn to define chosen/rejected responses
        chosen_messages = example["chosen"][-1]['content']
        rejected_messages = example["rejected"][-1]['content']
        
        # /NAS/dengx/temp/condaenv/miniconda3/envs/tr_llm/lib/python3.10/site-packages/trl/trainer/utils.py line-460 to_pad need to modify for qwen to solve the first None problem in input_ids
        example["prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True).replace(tokenizer.bos_token, "") # for llama and mistral
        # example["prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False) # for qwen
        example["chosen"] = chosen_messages #+ tokenizer.eos_token
        example["rejected"] = rejected_messages #+ tokenizer.eos_token
    else:
        raise ValueError(
            f"Could not format example as dialogue for the task! Require either the "
            f"`[chosen, rejected]` or `[prompt, chosen, rejected]` keys but found {list(example.keys())}"
        )
   
    return example


def apply_chat_template_tldr(
    example,
    tokenizer,
):

    prompt_messages = [{'role':'user','content':f"Summarize the following text with a TL;DR:\n\n{example['info']['post']}"}]
    # Now we extract the final turn to define chosen/rejected responses
    chosen_messages = example['summaries'][example['choice']]['text']
    rejected_messages = example['summaries'][1-example['choice']]['text']
    
    example["prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True).replace(tokenizer.bos_token, "")
    example["chosen"] = chosen_messages # + tokenizer.eos_token
    example["rejected"] = rejected_messages # + tokenizer.eos_token
   
    return example

def apply_chat_template_hh(
    example,
    tokenizer,
):
    prompt = extract_anthropic_prompt(example['chosen'])
    prompt_messages = [{'role':'user','content':f"Complete the following dialogue: {prompt}"}]
    # Now we extract the final turn to define chosen/rejected responses
    chosen_messages = example['chosen'][len(prompt):]
    rejected_messages = example['rejected'][len(prompt):]
    
    example["prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True).replace(tokenizer.bos_token, "")
    example["chosen"] = chosen_messages  + tokenizer.eos_token
    example["rejected"] = rejected_messages  + tokenizer.eos_token
   
    return example

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, \
    what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. \
            You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=5e-7) # 5e-7, 1e-6, 5e-6
    weight_decay: Optional[float] = field(default=0.0)
    model_name: Optional[str] = field(
        default="",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    dataset_name: Optional[str] = field(
        default="tldr",
        metadata={
            "help": "the path of the preference datasets",
        },
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=2,
        metadata={"help": "The number of training epochs for the model."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        #default="paged_adamw_32bit",
        default="adamw_torch",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        # default="constant_with_warmup",
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )

    max_training_samples: Optional[int] = field(default=-1, metadata={"help": "the maximum sample size"})

    max_length: Optional[int] = field(default=2048)
    output_dir: Optional[str] = field(default="")
    run_name: Optional[str] = field(default="")
    beta: Optional[float] = field(default=0.1)
    strategy: Literal["random", "in_reward_margin", "ex_reward_margin", "im_reward_margin", "ppl_margin", 'add', 'mul', 'max', 'min'] = field(default="random")
    part: Literal["top", "mid","bot", ""] = field(default="")
    num_samples: Optional[int] = field(default=2000)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]



tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# load dataset and mapping

if script_args.dataset_name == 'tldr':
    ds = load_from_disk('dataset/TLDR')['train']   
elif script_args.dataset_name == 'hh':
    # ds1 = load_from_disk('dataset/HH/hh-helpful')['train']
    # ds2 = load_from_disk('dataset/HH/hh-harmless')['train']
    data_dir = '/home/jovyan/llm_project/datasets/hh-rlhf/helpful-base'
    ds1 = load_dataset('json', data_files={'train':os.path.join(data_dir,'train.jsonl.gz'),'test': os.path.join(data_dir, 'test.jsonl.gz')})['train']
    data_dir = '/home/jovyan/llm_project/datasets/hh-rlhf/harmless-base'
    ds2 = load_dataset('json', data_files={'train':os.path.join(data_dir,'train.jsonl.gz'),'test': os.path.join(data_dir, 'test.jsonl.gz')})['train']
    ds = concatenate_datasets([ds1,ds2])
elif script_args.dataset_name == 'uf':
    ds = load_from_disk('dataset/ultrafeedback_binarized')
elif script_args.dataset_name == 'llama_uf':
    ds = load_from_disk('dataset/llama3-ultrafeedback')['train']
elif script_args.dataset_name == 'mistral_uf':
    ds = load_from_disk('dataset/mistral-ultrafeedback')['train']
elif script_args.dataset_name == 'llama_uf_armo':
    ds = load_from_disk('dataset/llama3-ultrafeedback-armo')['train']

if script_args.strategy in ['add', 'mul', 'max', 'min']:
    ds_tr = Sampling_ensemble(script_args.dataset_name, ds, script_args.strategy, script_args.part, script_args.num_samples)
else:
    ds_tr = Sampling(script_args.dataset_name, ds, script_args.strategy, script_args.part, script_args.num_samples)
print(ds_tr)
ds_tr = ds_tr.map(apply_chat_template,fn_kwargs={"tokenizer":tokenizer, "name":script_args.dataset_name}, num_proc=8)

print(ds_tr[0])

model = AutoModelForCausalLM.from_pretrained(script_args.model_name,
    torch_dtype = torch.float16,
    # load_in_4bit = True,
    # attn_implementation='flash_attention_2',  
    trust_remote_code = True,
    use_cache = False,
    device_map='auto'
)

model_ref = AutoModelForCausalLM.from_pretrained(script_args.model_name,
    torch_dtype = torch.float16,
    # load_in_4bit = True,
    # attn_implementation='flash_attention_2',  
    trust_remote_code = True,
    use_cache = False,
    device_map='auto'
)


dpo_trainer = DPOTrainer(
    model = model,
    # ref_model = model_ref,
    args = DPOConfig(
        per_device_train_batch_size = script_args.per_device_train_batch_size,
        gradient_accumulation_steps = script_args.gradient_accumulation_steps,
        warmup_ratio = 0.1,
        # label_smoothing = 0.1,
        num_train_epochs = script_args.num_train_epochs,
        learning_rate = script_args.learning_rate,
        bf16 = script_args.bf16,
        optim = script_args.optim,
        weight_decay = script_args.weight_decay,
        lr_scheduler_type = script_args.lr_scheduler_type,
        seed = 42,
        output_dir = script_args.output_dir,
        gradient_checkpointing = script_args.gradient_checkpointing, # True or "unsloth" for very long context
        gradient_checkpointing_kwargs= {
            "use_reentrant": False
        },
        # report_to="wandb",
        report_to="none",
        run_name=script_args.run_name,
        save_strategy="steps",
        save_steps=100,
        loss_type="sigmoid",
        remove_unused_columns=True,
        logging_strategy="steps",
        logging_steps=100,
        beta = script_args.beta,
        max_length = script_args.max_length,
        max_prompt_length =1500,
    ),
    train_dataset = ds_tr,
    processing_class = tokenizer,
    # peft_config=peft_config,
)

dpo_trainer.train()

# Save the model state
# model = dpo_trainer.model
# model.save_pretrained(script_args.output_dir, safe_serialization=True)
# tokenizer.save_pretrained(script_args.output_dir)