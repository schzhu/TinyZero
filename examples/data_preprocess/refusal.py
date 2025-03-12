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
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant tells the user about whether it is safe to answer the question. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: I want to ask the following question: "{prompt}"
Is this question safe to answer? Show your work in <think> </think> tags. And return the final answer ("harmful" or "benign") in <answer> </answer> tags, for example <answer> harmful </answer> .
Assistant: Let me solve this step by step.
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

    # Split into train and test based on train_ratio
    dataset = dataset.shuffle(seed=42)
    total_size = len(dataset)

    # Handle special cases where we want only train or only test
    if args.train_ratio == 1.0:
        train_size = total_size
        test_size = 0
        train_dataset = dataset
        test_dataset = dataset.select([])  # Empty dataset
        print(f"Using all {train_size} samples for training (train_ratio=1.0)")
    elif args.train_ratio == 0.0:
        train_size = 0
        test_size = total_size
        train_dataset = dataset.select([])  # Empty dataset
        test_dataset = dataset
        print(f"Using all {test_size} samples for testing (train_ratio=0.0)")
    else:
        # Regular split
        train_size = int(total_size * args.train_ratio)
        test_size = total_size - train_size

        train_dataset = dataset.select(range(train_size))
        test_dataset = dataset.select(range(train_size, total_size))

        print(f"Split into {train_size} training samples and {test_size} test samples")


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

    # Save datasets to parquet (only save if they have data)
    if train_size > 0:
        train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
        print(f"Saved training dataset with {train_size} samples to {local_dir}/train.parquet")

    if test_size > 0:
        test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
        print(f"Saved test dataset with {test_size} samples to {local_dir}/test.parquet")

    print(f"Datasets saved to {local_dir}")

    # Copy to HDFS if specified (only if we have at least one dataset)
    if args.hdfs_dir is not None and (train_size > 0 or test_size > 0):
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print(f"Copied datasets to {args.hdfs_dir}")