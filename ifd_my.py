import os
import json
import torch
import argparse
from tqdm import tqdm
from datasets import load_from_disk, concatenate_datasets, load_dataset
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='hh')
    parser.add_argument("--save_path", type=str, default='./Less-is-More/metric/dpo_2k_hh_ppl.jsonl')
    parser.add_argument("--model_name", type=str, default='./models/dpo_rand_hh')
    parser.add_argument("--tokenizer_name", type=str, default="./models/LLaMA3.2-3B-SFT")
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()
    return args

def get_perplexity_and_embedding_whole_text(tokenizer, model, text, max_length):
    """Calculate perplexity for standalone text"""
    try:
        if not text.strip():
            logging.warning("Empty text encountered in whole_text")
            return 0
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True
        ).to(model.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = model(**inputs, labels=inputs["input_ids"])
        
        loss = outputs.loss
        perplexity = torch.exp(loss)
        return perplexity.item()

    except Exception as e:
        logging.error(f"Error in whole_text perplexity: {str(e)}")
        logging.error(f"Problematic text: {text[:200]}...")
        return 0

def get_perplexity_and_embedding_part_text(tokenizer, model, context, response, max_length):
    """Calculate conditional perplexity for response given context"""
    try:
        # Construct full conversation
        full_text = context + response
        
        # Tokenize with special tokens
        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True
        ).to(model.device)
        
        # Tokenize context to determine response start position
        context_inputs = tokenizer(
            context,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True
        )
        
        # Create labels (ignore context tokens)
        labels = inputs["input_ids"].clone()
        context_length = len(context_inputs["input_ids"])
        labels[:, :context_length] = -100
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = model(**inputs, labels=labels)
        
        # Calculate response length loss
        response_length = labels.shape[1] - context_length
        loss = outputs.loss
        perplexity = torch.exp(loss)
        token_loss = loss.item() * response_length
        
        return perplexity.item(), token_loss

    except Exception as e:
        logging.error(f"Error in conditional perplexity: {str(e)}")
        logging.error(f"Context: {context[:200]}...")
        logging.error(f"Response: {response[:200]}...")
        return 0, 0

def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    if search_term_idx == -1:
        search_term = '\n\nHuman:'
        search_term_idx = prompt_and_response.rfind(search_term)
        if search_term_idx == -1:
            return prompt_and_response
    return prompt_and_response[:search_term_idx + len(search_term)]

def main():
    args = parse_args()
    print(args)

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model loading
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name,
        trust_remote_code=True
    )
    
    # Set padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()

    # Dataset loading
    if args.dataset_name == 'tldr':
        ds = load_from_disk('dataset/TLDR')['train']
    elif args.dataset_name == 'hh':
        helpful_dir = '/home/jovyan/llm_project/datasets/hh-rlhf/helpful-base'
        harmless_dir = '/home/jovyan/llm_project/datasets/hh-rlhf/harmless-base'
        
        ds1 = load_dataset('json', data_files={
            'train': os.path.join(helpful_dir, 'train.jsonl.gz'),
            'test': os.path.join(helpful_dir, 'test.jsonl.gz')
        })['train']
        
        ds2 = load_dataset('json', data_files={
            'train': os.path.join(harmless_dir, 'train.jsonl.gz'),
            'test': os.path.join(harmless_dir, 'test.jsonl.gz')
        })['train']
        
        ds = concatenate_datasets([ds1, ds2])
    elif args.dataset_name == 'uf':
        ds = load_from_disk('dataset/ultrafeedback_binarized')
    elif args.dataset_name in ['llama_uf', 'mistral_uf']:
        ds = load_from_disk(f'dataset/{args.dataset_name}')['train']

    # Resume from existing results
    start_idx = 0
    if os.path.exists(args.save_path):
        with open(args.save_path, "r") as file:
            start_idx = sum(1 for _ in file)
        print(f"Resuming from index: {start_idx}")
    
    ds = ds.select(range(start_idx, len(ds)-1))

    for i in tqdm(range(len(ds)), desc="Processing samples"):
        data_i = ds[i]
        
        # Prepare context and responses
        if args.dataset_name == 'tldr':
            context = f"Summarize the following text with a TL;DR:\n\n{data_i['info']['post']}"
            chosen_response = data_i['summaries'][data_i['choice']]['text']
            rejected_response = data_i['summaries'][1-data_i['choice']]['text']
        elif args.dataset_name == 'hh':
            prompt = extract_anthropic_prompt(data_i['chosen'])
            context = f"Complete the following dialogue:\n{prompt}"
            chosen_response = data_i['chosen'][len(prompt):].strip()
            rejected_response = data_i['rejected'][len(prompt):].strip()
        elif 'uf' in args.dataset_name:
            context = tokenizer.apply_chat_template(
                data_i["chosen"][:-1],
                tokenize=False,
                add_generation_prompt=True
            )
            chosen_response = data_i["chosen"][-1]['content']
            rejected_response = data_i["rejected"][-1]['content']

        # Calculate perplexities
        ppl_chosen_alone = get_perplexity_and_embedding_whole_text(tokenizer, model, chosen_response, args.max_length)
        ppl_rejected_alone = get_perplexity_and_embedding_whole_text(tokenizer, model, rejected_response, args.max_length)
        ppl_chosen_condition, loss_chosen_condition = get_perplexity_and_embedding_part_text(tokenizer, model, context, chosen_response, args.max_length)
        ppl_rejected_condition, loss_rejected_condition = get_perplexity_and_embedding_part_text(tokenizer, model, context, rejected_response, args.max_length)

        # Prepare and save results
        result = {
            'ppl_chosen': [ppl_chosen_alone, ppl_chosen_condition],
            'ppl_rejected': [ppl_rejected_alone, ppl_rejected_condition],
            'reward': [loss_chosen_condition - loss_rejected_condition]
        }
        
        with open(args.save_path, "a") as file:
            file.write(json.dumps(result) + '\n')

    print(f'Completed processing {len(ds)} samples from {args.dataset_name}')

if __name__ == "__main__":
    main()