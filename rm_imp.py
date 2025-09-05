import os
import json
import torch
import argparse
from tqdm import tqdm
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='hh')
    parser.add_argument("--save_path", type=str, default='./Less-is-More/metric/hh_imp_rm_scores_2.jsonl')
    parser.add_argument("--model_name", type=str, default='./models/dpo_rand_hh')
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    return args

def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]

def get_reward_score(rm_model, rm_tokenizer, prompt_messages, response, device, model_type):
    conv = prompt_messages + [{"role": "assistant", "content": response}]
    
    if model_type == "seq":
        conv_tokenized = rm_tokenizer.apply_chat_template(conv, tokenize=True, return_tensors="pt").to(device)
        with torch.no_grad():
            score = rm_model(conv_tokenized).logits[0][0].item()
        return score
    else:  # causal
        input_ids = rm_tokenizer.apply_chat_template(conv, return_tensors="pt", add_special_tokens=True).to(device)
        prompt_only_input = rm_tokenizer.apply_chat_template(prompt_messages, return_tensors="pt", add_special_tokens=True).to(device)
        prompt_len = prompt_only_input.shape[1]
        
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100  # Mask prompt tokens
        
        with torch.no_grad():
            outputs = rm_model(input_ids, labels=labels)
            loss = outputs.loss
            if loss is not None:
                score = -loss.item()  # Average log prob per token
            else:
                # Manual calculation if loss not returned
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                per_token_loss = per_token_loss.view(shift_labels.shape)
                mask = (shift_labels != -100)
                response_token_loss = per_token_loss[mask]
                if response_token_loss.numel() == 0:
                    score = 0.0
                else:
                    score = -response_token_loss.mean().item()
        return score

def main():
    args = parse_args()
    print(args)
    
    device = args.device
    
    # Load model and tokenizer with fallback
    try:
        rm = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        model_type = "causal"
    except:
        try:
            rm = AutoModelForSequenceClassification.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16,
                device_map=device,
                num_labels=1,
            )
            model_type = "seq"
        except Exception as e:
            raise ValueError(f"Model loading failed: {str(e)}")
    
    rm_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if rm_tokenizer.pad_token is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
    rm.eval()

    # Load dataset
    if args.task == 'tldr':
        ds = load_from_disk('dataset/TLDR')['train']
    elif args.task == 'hh':
        data_dir = '/home/jovyan/llm_project/datasets/hh-rlhf/helpful-base'
        ds1 = load_dataset('json', data_files={'train':os.path.join(data_dir,'train.jsonl.gz'),'test': os.path.join(data_dir, 'test.jsonl.gz')})['train']
        data_dir = '/home/jovyan/llm_project/datasets/hh-rlhf/harmless-base'
        ds2 = load_dataset('json', data_files={'train':os.path.join(data_dir,'train.jsonl.gz'),'test': os.path.join(data_dir, 'test.jsonl.gz')})['train']
        ds = concatenate_datasets([ds1, ds2])
    elif args.task == 'uf':
        ds = load_from_disk('dataset/ultrafeedback_binarized')
    elif args.task == 'llama_uf':
        ds = load_from_disk('dataset/llama3-ultrafeedback')['train']
    elif args.task == 'mistral_uf':
        ds = load_from_disk('dataset/mistral-ultrafeedback')['train']
    elif args.task == 'llama_uf_armo':
        ds = load_from_disk('dataset/llama3-ultrafeedback-armo')['train']

    # Create save file if it doesn't exist
    if not os.path.exists(args.save_path):
        with open(args.save_path, "w") as file:
            pass

    # Resume from last saved position
    with open(args.save_path, "r") as file:
        existing_num = sum(1 for _ in file)
        print(f'existing row:{existing_num}')
    ds = ds.select(range(existing_num, len(ds)))

    for i in tqdm(range(len(ds))):
        data_i = ds[i]
        
        if args.task == 'tldr':
            prompt_messages = [{'role':'user','content':f"Summarize the following text with a TL;DR:\n\n{data_i['info']['post']}"}]
            chosen_messages = data_i['summaries'][data_i['choice']]['text']
            rejected_messages = data_i['summaries'][1-data_i['choice']]['text']
        elif args.task == 'hh':
            prompt = extract_anthropic_prompt(data_i['chosen'])
            prompt_messages = [{'role':'user','content':f"Complete the following dialogue:{prompt}"}]
            chosen_messages = data_i['chosen'][len(prompt):]
            rejected_messages = data_i['rejected'][len(prompt):]
        elif args.task in ['uf', 'llama_uf', 'mistral_uf', 'llama_uf_armo']:
            prompt_messages = data_i["chosen"][:-1]
            chosen_messages = data_i["chosen"][-1]['content']
            rejected_messages = data_i["rejected"][-1]['content']

        # Calculate reward scores
        chosen_score = get_reward_score(rm, rm_tokenizer, prompt_messages, chosen_messages, device, model_type)
        rejected_score = get_reward_score(rm, rm_tokenizer, prompt_messages, rejected_messages, device, model_type)
        
        # Save results
        temp_data_i = {
            'chosen_score': chosen_score,
            'rejected_score': rejected_score,
            'reward_diff': chosen_score - rejected_score
        }

        with open(args.save_path, "a") as file:
            file.write(json.dumps(temp_data_i) + '\n')

    rm.cpu()
    print('Done: Data Analysis:', args.task)

if __name__ == "__main__":
    main()