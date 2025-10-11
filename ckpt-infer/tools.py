import pathlib
import json
from tqdm import tqdm
import pandas as pd
import os
import random
import re

def load_json(file_path):
    with open(file_path, 'r') as f:
        ds = json.load(f)
    return ds

def load_jsonl(file_path):
    with open(file_path,'r') as f:
        ds = [json.loads(line) for line in f.readlines()]
    return ds

def write_json(file_path,data):
    with open(file_path, 'a') as f:
        json.dump(data,f,ensure_ascii=False,indent=4)

def write_jsonl(file_path,data):
    data=tqdm(data)
    for d in data:
        with open(file_path, 'a') as f:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

def write_file(path,data):
    if path.endswith('jsonl'):
        write_jsonl(path,data)
    else:
        write_json(path,data)

def read_file(file_path):
    if file_path.endswith('json'):
        ds = load_json(file_path)
    else:
        ds = load_jsonl(file_path)
    return ds


def general_read_file(file_path,is_ares = False):
    test_ds = None
    if is_ares:
        assert os.path.isdir(file_path)
        train_ds = load_jsonl(file_path+"/train.json")
        test_ds = load_jsonl(file_path+"/test.json")
    else:

        if file_path.endswith("xlsx"):
            ##读取人工标注
            train_ds = pd.read_excel(file_path)
        else:
            train_ds = read_file(file_path)

    return train_ds,test_ds

def split_train_test(ds,train_test_ratio = "80:20"):
    ratio=train_test_ratio.split(":")
    random.shuffle(ds)
    train = int(int(ratio[0]) / 100 * len(ds))
    train_ds = ds[:train]
    test_ds = ds[train+1:]

    return train_ds,test_ds

def general_write_file(file_path,data,is_ares = False,ratio = "8:2"):
    """
    write 的时候基本上都是json或者jsonl
    """
    if is_ares:
        os.makedirs(file_path,exist_ok=True)
        assert os.path.isdir(file_path)
        train_ds,test_ds = split_train_test(data,ratio)

        write_jsonl(file_path+"/train.json",train_ds)
        write_jsonl(file_path+"/test.json",test_ds)
    else:
        write_file(file_path,data)

import os

def merge_dir(dir, write_path):
    # 检查目录是否存在
    if not os.path.exists(dir):
        print(f"Directory {dir} does not exist.")
        return

    for root, dirs, files in os.walk(dir):
        for file in files:
            file_path = os.path.join(root, file)
            ds = read_file(file_path)
            write_file(write_path,ds)

    print(f"Merged all files from {dir} into {write_path}")


def merge_files(files):
    full_ds = []
    for file in files:
        ds = read_file(file)
        full_ds.extend(ds)
    return full_ds

def extract_mask(text):
    # 提取MASK结果
    # mask_pattern = r'### \[MASK\]_(\d+) Restoration Result:\s*\n\$\$(.*?)\$\$'
    mask_pattern = r'\*\*\[MASK\]_(\d+) Restoration Result:\*\*\s*\$\$(.*?)\$\$'
    mask_results = re.findall(mask_pattern, text, re.DOTALL)
    return {
        'mask_id': mask_results[0][0] if mask_results else None,
        'formula': mask_results[0][1].strip() if mask_results else None,
    }

