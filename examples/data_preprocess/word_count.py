"""
Preprocess dataset for word count task - generate prompts asking for stories of exactly N words
"""

import os
import random
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse


def generate_prompts(num_samples, min_words=10, max_words=200, seed_value=42):
    """Generate random word count targets for stories"""
    random.seed(seed_value)

    data = []
    for _ in tqdm(range(num_samples)):
        word_count = random.randint(min_words, max_words)
        data.append({"target_words": word_count})

    return pd.DataFrame(data)


def make_prefix(dp, template_type):
    target_words = dp['target_words']

    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""A conversation between User and Assistant. The user asks the assistant to write a story with exactly {target_words} words. The assistant first drafts the story in the mind and double checks the word count, and revise it in the mind until the word count matches. Then it provides the user with the final story.
User: Please write me a creative story that is exactly {target_words} words long. Not one word more or less. Draft and iteratively revise your draft until the word count matches in <think> </think> tags. And return the final story in <answer> </answer> tags. For example, <think> [your draft] [word count check] [your draft] [word count check] ... </think> <answer> [your final answer] </answer>.
Assistant: Let me draft it and check the word count in mind.\n<think>.
"""
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""<|im_start|>system\nYou are a helpful assistant that can write creative stories with precise word counts.<|im_end|>\n<|im_start|>user\nPlease write me a creative story that is exactly {target_words} words long. Not one word more or less. Show your final story in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\nI'll write a creative story with exactly {target_words} words.\n"""

    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/word_count')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_samples', type=int, default=50000)
    parser.add_argument('--test_samples', type=int, default=500)
    parser.add_argument('--min_words', type=int, default=10)
    parser.add_argument('--max_words', type=int, default=200)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    data_source = 'word_count'

    # Generate training data
    print(f"Generating {args.train_samples} training samples...")
    train_df = generate_prompts(
        num_samples=args.train_samples,
        min_words=args.min_words,
        max_words=args.max_words,
        seed_value=args.seed
    )

    # Generate test data
    print(f"Generating {args.test_samples} test samples...")
    test_df = generate_prompts(
        num_samples=args.test_samples,
        min_words=args.min_words,
        max_words=args.max_words,
        seed_value=args.seed + 1  # Use different seed for test data
    )

    # Convert to Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "target_words": example['target_words']
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "creativity",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Save datasets to parquet
    print(f"Saving datasets to {local_dir}...")
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print(f"Copied datasets to {args.hdfs_dir}")

    print("Dataset generation complete!")