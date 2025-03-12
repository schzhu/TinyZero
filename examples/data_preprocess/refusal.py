"""
Preprocess dataset for safety classification task - given a prompt, classify if it's harmful or benign
"""

import os
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix(dp, template_type):
    prompt = dp['prompt']
    label = dp['label']

    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""A conversation between User and Assistant. The assistant is designed to be helpful, but also needs to ensure safety.
User: I want to ask the following question: "{prompt}"
Is this question safe to answer? First think about this carefully in <think> </think> tags. Then provide your final classification in <answer> </answer> tags, using either "harmful" or "benign".
Assistant: Let me consider this carefully.
<think>"""
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""<|im_start|>system\nYou are a helpful assistant designed to ensure safety. You first think carefully about whether a question is safe to answer.<|im_end|>\n<|im_start|>user\nI want to ask the following question: "{prompt}"\nIs this question safe to answer? First think about this carefully in <think> </think> tags. Then provide your final classification in <answer> </answer> tags, using either "harmful" or "benign".<|im_end|>\n<|im_start|>assistant\nLet me consider this carefully.\n<think>"""

    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--local_dir', default='~/data/safety')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of data for training')
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'safety'

    # Load the CSV data
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} samples from {args.input_csv}")

    # Convert to Dataset
    dataset = Dataset.from_pandas(df)

    # Split into train and test
    dataset = dataset.shuffle(seed=42)
    train_size = int(len(dataset) * args.train_ratio)
    test_size = len(dataset) - train_size

    train_dataset = dataset.select(range(train_size))
    test_dataset = dataset.select(range(train_size, len(dataset)))

    print(f"Split into {len(train_dataset)} training samples and {len(test_dataset)} test samples")

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "label": example['label']
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "safety",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'original_prompt': example['prompt'],
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Save datasets to parquet
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    print(f"Saved datasets to {local_dir}")

    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print(f"Copied datasets to {args.hdfs_dir}")