import ast
import math
import re
from multiprocessing import Pool, cpu_count
import argparse
import json
import os
from string import Template
import threading
import tqdm
import openai
from openai import OpenAI
import time
import random
import concurrent.futures 
import requests
from prompts import inference_prompts
from tools import write_file,extract_mask

parser = argparse.ArgumentParser(description='Example usage of argparse module')
# --- MODIFIED: Renamed model_name to output_prefix for clarity ---
parser.add_argument('--output_prefix', type=str, default='deepseek_r1_friday_test', help='Prefix for output files and checkpoints.') 
# --- MODIFIED: Changed argument name to inference_model for clarity ---
parser.add_argument('--inference_model', type=str, default='deepseek-v3-friday', help='The model name to be used for inference.')
# --- MODIFIED: Added base_url as a configurable argument ---
parser.add_argument('--base_url', type=str, default='https://aihubmix.com/v1', help='The base URL for the API endpoint.')
parser.add_argument('--api_key', type=str, default='YOUR_API_KEY_HERE', help='Your API key for the inference service.') # MODIFIED: Changed default to a placeholder
parser.add_argument('--dataset_list', type=str, default="")
parser.add_argument('--multi_round', type=bool, default=False)
parser.add_argument('--top_p', type=float, default=0.95) 
parser.add_argument('--parallel_size', type=str, default='[10]')
parser.add_argument('--n_responses', type=str, default='[1]') 
parser.add_argument('--max_tokens', type=int, default=8192) 
parser.add_argument('--temperature', type=float, default=0.0) 
parser.add_argument('--input_key', type=str, default="input")
parser.add_argument('--data_dir', type=str, default="") 
parser.add_argument('--output_dir',type=str,default="")
# 新增断点恢复相关参数
parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
parser.add_argument('--checkpoint_interval', type=int, default=100, help='Save checkpoint every N items')

args = parser.parse_args() 

# This seems unused, keeping it as is from the original script
args.app_ids = ["xx"]

# --- MODIFIED: Updated RemoteServer class to accept parameters properly ---
class RemoteServer:
    def __init__(self, model_name, api_key, base_url):
        """
        Initializes the RemoteServer client.
        Args:
            model_name (str): The name of the model to use for inference.
            api_key (str): The API key for authentication.
            base_url (str): The base URL of the API service.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        
        # This was the main issue: api_key and base_url were hardcoded before.
        # Now, it uses the values passed during initialization.
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
        print(f"Loaded model: {self.model_name}")
        print(f"Using base URL: {self.base_url}")

    def chat_sync(self, message, system=None, multi_round=args.multi_round, max_tokens=args.max_tokens, temperature=args.temperature, maxtry=50, response_format="text"):
        if not multi_round:
            assert isinstance(message, str), 'The input prompt should be a string.'
            if system:
                messages = [{"role": "system", "content": system},
                            {"role": "user", "content": message}]
            else:
                messages = [{"role": "user", "content": message}]
        else:
            assert isinstance(message, list), 'The input of multi round dialouge should be a list.'
            messages = message
        i = 0
        while i < maxtry:
            try:
                response = self.client.chat.completions.create(
                    model = self.model_name,
                    messages=messages,
                    stream=False,
                    extra_headers={"Accept":"text/event-stream"},
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=1234,
                    response_format={"type": response_format}
                )
                response_content = response.choices[0].message
                return {"response": response_content.content, "reasoning": getattr(response_content, 'reasoning_content', None)}
            except openai.RateLimitError as e:
                time.sleep(60)
                print(f"Try {i}/{maxtry}\t message:{message} \tError:{e}", flush=True)
                i += 1
                continue
            except openai.APIStatusError as e:
                print(e)
                print({'inputs':message})
                return {"response": " ", "reasoning": " "}
            # --- FIXED: Added a general exception handler for robustness ---
            except Exception as e:
                print(f"An unexpected error occurred on try {i}/{maxtry}: {e}")
                time.sleep(10)
                i += 1
        
        # If all retries fail, return an empty response
        print(f"Failed to get a response for message after {maxtry} tries.")
        return {"response": " ", "reasoning": "Failed after multiple retries."}

# --- MODIFIED: Model instantiation now correctly uses command-line arguments ---
# NOTE: The 'CloudServer' class was not defined in the original script. 
# If you need to use a Claude model, you will need to define that class.
if "claude" in args.inference_model:
    # model = CloudServer(model_name="anthropic.claude-sonnet-4") # This will raise a NameError
    raise NotImplementedError("The 'CloudServer' class is not defined in this script.")
else:
    model = RemoteServer(
        model_name=args.inference_model,
        api_key=args.api_key,
        base_url=args.base_url
    )

question_prompt_map = {
    "file": inference_prompts
}

# --- MODIFIED: CheckpointManager now uses the clearer `output_prefix` argument ---
class CheckpointManager:
    def __init__(self, output_dir, output_prefix):
        self.output_dir = output_dir
        self.output_prefix = output_prefix
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
    
    def get_checkpoint_path(self, chunk_index=None):
        if chunk_index is not None:
            return os.path.join(self.checkpoint_dir, f"{self.output_prefix}_chunk_{chunk_index}_checkpoint.json")
        else:
            return os.path.join(self.checkpoint_dir, f"{self.output_prefix}_checkpoint.json")
    
    def save_checkpoint(self, state, chunk_index=None):
        checkpoint_path = self.get_checkpoint_path(chunk_index)
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, chunk_index=None):
        checkpoint_path = self.get_checkpoint_path(chunk_index)
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            print(f"Checkpoint loaded: {checkpoint_path}")
            return state
        return None
    
    def get_processed_results_path(self, chunk_index=None):
        if chunk_index is not None:
            return os.path.join(self.checkpoint_dir, f"{self.output_prefix}_chunk_{chunk_index}_processed.jsonl")
        else:
            return os.path.join(self.checkpoint_dir, f"{self.output_prefix}_processed.jsonl")
    
    def save_processed_results(self, results, chunk_index=None):
        results_path = self.get_processed_results_path(chunk_index)
        save_jsonl(results_path, results)
    
    def load_processed_results(self, chunk_index=None):
        results_path = self.get_processed_results_path(chunk_index)
        if os.path.exists(results_path):
            return load_jsonl(results_path)
        return []
    
    def cleanup_checkpoints(self, chunk_index=None):
        """清理检查点文件"""
        checkpoint_path = self.get_checkpoint_path(chunk_index)
        results_path = self.get_processed_results_path(chunk_index)
        
        for path in [checkpoint_path, results_path]:
            if os.path.exists(path):
                os.remove(path)
                print(f"Cleaned up: {path}")

# (The rest of the functions: load_jsonl, save_jsonl, has_test_type, parse_dialog_history, predict_ans remain largely the same)
def load_jsonl(data_path):
    data_list = []
    with open(data_path) as f:
        for line in f:
            json_line = json.loads(line)
            data_list.append(json_line)
    return data_list

def save_jsonl(savepath, data):
    with open(savepath, 'w') as fw:
        for d in data:
            fw.write(json.dumps(d, ensure_ascii=False)+'\n')


def has_test_type(tests, type):
    test_list = json.loads(tests)
    for test in test_list:
        if test.get("testtype") == type:
            return True
    return False


def parse_dialog_history(dialog_text):
    if not dialog_text or dialog_text ==None:
        return []
    lines = dialog_text.strip().split('\n')
    messages = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = re.match(r'^(user|assistant):\s*(.*)', line, re.IGNORECASE)
        if match:
            role = match.group(1).lower()
            content = match.group(2).strip()
            messages.append({"role": role, "content": content})
    return messages

def predict_ans(data, retry_num=5, num_responses=5,input_key = "input",add_prompt = False):
    responses = []
    extract_answers = []

    query = data[input_key] if add_prompt == False else inference_prompts.format(proof_text=data[input_key])
    for _ in range(num_responses):
        cnt = 0
        while cnt < retry_num:
            result = model.chat_sync(query, temperature=args.temperature, max_tokens=args.max_tokens)
            
            if "response" in result and result['response'].strip(): # Check for non-empty response
                responses.append(result)
                extract_answer = extract_mask(result['response'])
                extract_answers.append(extract_answer)
                break
            else:
                cnt += 1
        if cnt == retry_num:
            responses.append({"response": "", "reasoning": "Failed after retries."}) # Add empty response object
            extract_answers.append(None)

    data['model_responses'] = responses
    data['extract_answers'] = extract_answers  

    if any(r['response'] == "" for r in responses):
        print("===== Found a timed-out case, please check. =====")
    return data

# (predict_multi_thread_with_checkpoint and predict_multi_thread are OK)
def predict_multi_thread_with_checkpoint(data, pbar, lock, checkpoint_manager, chunk_index, 
                                         parallel=60, retry_num=5, num_responses=5, 
                                         input_key="input", add_prompt=False):
    checkpoint_state = checkpoint_manager.load_checkpoint(chunk_index)
    processed_results = checkpoint_manager.load_processed_results(chunk_index)
    
    if checkpoint_state:
        start_idx = checkpoint_state.get('processed_count', 0)
        print(f"Resuming from index {start_idx} for chunk {chunk_index}")
        with lock:
            pbar.update(start_idx)
    else:
        start_idx = 0
        processed_results = []
    
    remaining_data = data[start_idx:]
    
    def process_single_item(item_data, item_idx):
        try:
            result = predict_ans(item_data, retry_num, num_responses, input_key, add_prompt)
            if result:
                result['original_index'] = item_idx 
            return result, item_idx
        except Exception as e:
            print(f"Error processing item {item_idx}: {e}")
            return None, item_idx
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures_to_idx = {
            executor.submit(process_single_item, item_data, start_idx + idx): start_idx + idx
            for idx, item_data in enumerate(remaining_data)
        }
        
        completed_count = start_idx
        
        for future in concurrent.futures.as_completed(futures_to_idx):
            try:
                result, original_idx = future.result()
                if result is not None:
                    processed_results.append(result)
                    completed_count += 1
                    
                    if completed_count % args.checkpoint_interval == 0:
                        checkpoint_state = {
                            'processed_count': completed_count,
                            'total_count': len(data),
                            'chunk_index': chunk_index,
                            'timestamp': time.time()
                        }
                        checkpoint_manager.save_checkpoint(checkpoint_state, chunk_index)
                        checkpoint_manager.save_processed_results(processed_results, chunk_index)
            except Exception as e:
                print(f"Error processing future: {e}")
            finally:
                with lock:
                    pbar.update(1)

    sorted_results = sorted(processed_results, key=lambda item: item['original_index'])

    checkpoint_state = {
        'processed_count': len(data),
        'total_count': len(data),
        'chunk_index': chunk_index,
        'timestamp': time.time(),
        'completed': True
    }
    checkpoint_manager.save_checkpoint(checkpoint_state, chunk_index)
    checkpoint_manager.save_processed_results(sorted_results, chunk_index)
    
    return sorted_results

def predict_multi_thread(data, pbar, lock, parallel=60, retry_num=5, num_responses=5,input_key = "input",add_prompt = False):
    predict_result = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures_to_idx = {
            executor.submit(predict_ans, d, retry_num, num_responses,input_key,add_prompt): idx for idx, d in enumerate(data)
        }
        for future in concurrent.futures.as_completed(futures_to_idx):
            try:
                result = future.result()
                predict_result.append(result)
            except Exception as e:
                print(f"Error processing {futures_to_idx[future]}: {e}")
            finally:
                with lock:
                    pbar.update(1)
    return predict_result

def process_chunk_with_checkpoint(chunk_data, chunk_index, pbar, lock, num_responses, parallel, 
                                  input_key="input", add_prompt=False, use_checkpoint=True):
    
    checkpoint_manager = CheckpointManager(args.output_dir, args.output_prefix)
    
    if use_checkpoint:
        results = predict_multi_thread_with_checkpoint(
            chunk_data, pbar, lock, checkpoint_manager, chunk_index,
            num_responses=num_responses, parallel=parallel, 
            input_key=input_key, add_prompt=add_prompt
        )
    else:
        results = predict_multi_thread(
            chunk_data, pbar, lock, 
            num_responses=num_responses, parallel=parallel,
            input_key=input_key, add_prompt=add_prompt
        )
    
    output_file_dir = args.output_dir
    if not os.path.isdir(output_file_dir):
        os.makedirs(output_file_dir)
    
    output_path = os.path.join(output_file_dir, f"{args.output_prefix}_chunk_{chunk_index}.jsonl")
    save_jsonl(output_path, results)
    
    if use_checkpoint:
        checkpoint_manager.cleanup_checkpoints(chunk_index)

def process_chunk(chunk_data, chunk_index, pbar, lock, num_responses,parallel,input_key = "input",add_prompt = False):
    results = predict_multi_thread(
        chunk_data, pbar, lock, 
        num_responses=num_responses,
        parallel=parallel,
        input_key=input_key,
        add_prompt = add_prompt
    )
    output_file_dir = args.output_dir
    if not os.path.isdir(output_file_dir):
        os.makedirs(output_file_dir)
    
    output_path = os.path.join(output_file_dir, f"{args.output_prefix}_chunk_{chunk_index}.jsonl")
    save_jsonl(output_path, results)

def merge_chunks(num_chunks):
    all_data_list = []
    chunk_output_dir = args.output_dir
    
    for chunk_index in range(num_chunks):
        chunk_output_path = os.path.join(chunk_output_dir, f"{args.output_prefix}_chunk_{chunk_index}.jsonl")
        if os.path.exists(chunk_output_path):
            all_data_list += load_jsonl(chunk_output_path)
            os.remove(chunk_output_path) # Delete chunk after merging
        else:
            print(f"Warning: Chunk file {chunk_output_path} not found")

    target_large_file = os.path.join(chunk_output_dir, f"{args.output_prefix}_final_result.jsonl")
    save_jsonl(target_large_file, all_data_list)
    print(f"Merged chunks into {target_large_file}")

# (resume_ckpt, inferece_file_with_checkpoint, inferece_file are OK, but updated to use output_prefix)
def resume_ckpt(args):
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        return False
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('_checkpoint.json')]
    if not checkpoint_files:
        return False
    
    print(f"Found {len(checkpoint_files)} checkpoint files. Resuming is possible.")
    return True

def inferece_file_with_checkpoint(ds_file, num_responses, parallel, input_key="input", add_prompt=False):
    data_list = load_jsonl(ds_file)
    chunks = [data_list] # This script doesn't actually chunk the file, but keeps the structure
    assert len(parallel) == 1
    assert len(num_responses) == 1
    parallel = parallel[0]
    num_responses = num_responses[0]
    
    use_checkpoint = args.resume or resume_ckpt(args)
    if use_checkpoint:
        print("Using checkpoint recovery mode.")
    
    total_items = len(data_list)
    processed_count = 0
    if use_checkpoint:
        checkpoint_manager = CheckpointManager(args.output_dir, args.output_prefix)
        for chunk_idx in range(len(chunks)):
            checkpoint_state = checkpoint_manager.load_checkpoint(chunk_idx)
            if checkpoint_state:
                processed_count += checkpoint_state.get('processed_count', 0)
    
    pbar = tqdm.tqdm(total=total_items, initial=processed_count)
    lock = threading.Lock()
    
    threads = []
    for index, c in enumerate(chunks):
        target_func = process_chunk_with_checkpoint if use_checkpoint else process_chunk
        t = threading.Thread(target=target_func, 
                             args=(c, index, pbar, lock, num_responses, parallel, 
                                   input_key, add_prompt))
        # For process_chunk_with_checkpoint, add the use_checkpoint arg if needed
        if use_checkpoint:
            t = threading.Thread(target=process_chunk_with_checkpoint, 
                                 args=(c, index, pbar, lock, num_responses, parallel, 
                                       input_key, add_prompt, True))

        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    merge_chunks(len(chunks))
    pbar.close()
    print("Completed file inference task.")


def main():
    if args.dataset_list:
        dataset_list = ast.literal_eval(args.dataset_list)
    else:
        dataset_list = None

    n_responses = ast.literal_eval(args.n_responses)
    parallel_size = ast.literal_eval(args.parallel_size)
    input_key = args.input_key

    if dataset_list is not None:
        for dataset_name, num_responses, parallel in zip(dataset_list, n_responses, parallel_size):
            print(f"|- Processing dataset {dataset_name}, generating {num_responses} responses per query")
            data_list = load_jsonl(args.data_dir) # Assumes data_dir points to the file for this dataset
            chunks = [data_list]

            pbar = tqdm.tqdm(total=len(data_list), desc=dataset_name)
            lock = threading.Lock()

            threads = []
            for index, c in enumerate(chunks):
                # --- FIXED: Bug with incorrect arguments passed to process_chunk ---
                # The original script passed `dataset_name`, which is not a valid argument for the function.
                t = threading.Thread(target=process_chunk, args=(c, index, pbar, lock, num_responses, parallel, input_key))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

            merge_chunks(len(chunks))
            pbar.close()
            print(f"Completed prediction task for {dataset_name}.")
    else:
        inferece_file_with_checkpoint(args.data_dir, num_responses=n_responses, parallel=parallel_size, 
                                      input_key=input_key, add_prompt=True)
    

if __name__ == '__main__':
    start_time = time.time()
    main()  
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
