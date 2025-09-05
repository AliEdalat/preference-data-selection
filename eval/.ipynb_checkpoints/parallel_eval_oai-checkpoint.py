from openai import OpenAI
from datasets import load_from_disk, load_dataset
import time
from tqdm import tqdm
import argparse
import backoff
from openai import RateLimitError
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import openai

# setup deepseek API config
# client = OpenAI(
#     base_url="https://api.avalai.ir/v1",
#     api_key='aa-ScXriy8wT5Jrh4R5tVxH9IlY74aHdi56eAlG8YihRZ71sOGG',
#     timeout=120
# )

client = OpenAI(
    base_url="https://api.avalai.ir/v1",
    api_key='aa-mOIKIdeBbeV4eBPKIWadg1kHF2EzUGTWOGWh1I0BNgMwmwDn',
    timeout=360
)


@backoff.on_exception(
    backoff.expo,
    (RateLimitError, openai.BadRequestError),
    max_tries=5,
    max_time=600
)
def make_openai_request(client, prompt):
    try:
        # response = client.chat.completions.create(
        #     # model="gpt-4o-mini",
        #     # model="gpt-4-1106-preview",
        #     model="deepseek-v3-0324",
        #     messages=[
        #         {"role": "system", "content": "You are a helpful assistant"},
        #         {"role": "user", "content": prompt},
        #     ],
        #     max_tokens=2048,
        #     temperature=1.0,
        #     stream=False
        # )
        response = client.chat.completions.create(
            # model="gpt-4o-mini",
            # model="gpt-4-1106-preview",
            # model="deepseek-v3-0324",
            model="openai.gpt-oss-120b-1:0",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
            temperature=1.0,
            stream=False
        )
        return response
    except openai.BadRequestError as e:
        print(f"Error with prompt: {prompt[:100]}...")
        raise

async def process_sample(i, ds, prompt_template, task):
    try:
        if task == 'tldr':
            post, model_summary, chosen_summary = ds[i]['prompt'], ds[i]['model_response'][0], ds[i]['chosen_response'][0]
            prompt = prompt_template.format(Post=post, Summary_A=model_summary, Summary_B=chosen_summary)
        elif task == 'hh':
            post, model_response, chosen_response = ds[i]['prompt'], ds[i]['model_response'][0], ds[i]['chosen_response'][0]
            prompt = prompt_template.format(query=post, res_a=model_response, res_b=chosen_response)
        elif task == 'uf':
            post, model_response, chosen_response = ds[i]['prompt'], ds[i]['model_response'][0], ds[i]['chosen_response'][0]
            prompt = prompt_template.format(instruction=post, output_1=model_response, output_2=chosen_response)
        
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            response = await loop.run_in_executor(pool, partial(make_openai_request, client, prompt))
        
        answer = response.choices[0].message.content
        answer = answer.replace('**', '')
        # answer = answer.replace('**', '')
        # print(answer)
        # if task == "TLDR":
        #     return 1 if "Preferred: A" in answer else 0
        # elif task == 'hh':
        #     return 1 if "More helpful or harmless: A" in answer else 0
        
        if task == "tldr":
            if "Preferred: A" in answer:
                return 1
            elif "Preferred: A" not in answer and "Preferred: B" not in answer:
                print(f'The {i}-th sample is not properly judged.')
        elif task == 'hh':
            if "More helpful or harmless: A" in answer:
                return 1
            elif "More helpful or harmless: A" not in answer and "More helpful or harmless: B" not in answer:
                print(f'The {i}-th sample is not properly judged.')
        elif task == 'uf':
            if "The better response: m" in answer:
                return 1
            elif "\nm" in answer:
                return 1
            elif "The better response: M" not in answer and "The better response: m" not in answer and "\nM" not in answer and "\nm" not in answer:
                print(f'The {i}-th sample is not properly judged.')
                print(answer)
        return 0
    except Exception as e:
        print(f"Error processing sample {i}: {str(e)}")
        return 0

async def main():
    parser = argparse.ArgumentParser(description='file names')
    parser.add_argument('--task', type=str, default='hh')
    parser.add_argument('--file_name', type=str, default='uf_full') # exp_hh_P  uf_ex_z
    args = parser.parse_args()


    if args.task == 'tldr':
        with open('Less-is-More/template/summarization.txt', 'r') as file:
            prompt_template = file.read()
    elif args.task == 'hh':
        with open('Less-is-More/template/dialogue.txt', 'r') as file:
            prompt_template = file.read()
    elif args.task == 'uf':
        with open('Less-is-More/template/alpaca.txt', 'r') as file:
            prompt_template = file.read()

    data_dir = f'./Less-is-More/eval/result/gen/{args.file_name}.jsonl'
    ds = load_dataset('json', data_files={'train':data_dir})['train']
    N = len(ds)

    tasks = []
    for i in range(N):
        task = process_sample(i, ds, prompt_template, args.task)
        tasks.append(task)

    sem = asyncio.Semaphore(1)
    async def bounded_process_sample(task):
        async with sem:
            return await task

    results = []
    with tqdm(total=N, desc='Calling gpt4 model to judge the result:') as pbar:
        for task in asyncio.as_completed([bounded_process_sample(task) for task in tasks]):
            result = await task
            results.append(result)
            pbar.update(1)

    win_num = sum(results)
    print(f'Number of win: {win_num}, the win rate is {win_num/N:.4f}')

if __name__ == "__main__":
    asyncio.run(main())
