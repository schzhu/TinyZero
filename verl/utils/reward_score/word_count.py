"""
Reward scoring function for the word count story task
"""

import re
import random
import math


def extract_solution(solution_str):
    """Extract the story from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None

    # Extract the story from <answer> tags
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    if match:
        story = match.group(1).strip()
        return story
    else:
        return None


def count_words(text):
    """Count the number of words in a text."""
    if text is None:
        return 0

    # Split by whitespace and count non-empty words
    words = [word for word in text.split() if word.strip()]
    return len(words)


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, max_score=1.0):
    """The scoring function for word count story task.

    Args:
        solution_str: the solution text
        ground_truth: dictionary containing the target word count
        method: the method to extract the solution
        format_score: the score for correct format but wrong word count
        max_score: the maximum score for perfect word count
    """
    target_words = ground_truth['target_words']

    story = extract_solution(solution_str=solution_str)
    word_count = count_words(story)

    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Target words: {target_words}")
        print(f"Actual words: {word_count}")
        print(f"Solution string: {solution_str}")
        # print(f"Story: {story[:100]}..." if story else "No story found")

    if story is None:
        if do_print:
            print(f"No story found in <answer> tags")
        return 0

    # Calculate score based on how close the word count is to the target
    if word_count == target_words:
        # Exact match gets max score
        score = max_score
        if do_print:
            print(f"Perfect word count! Score: {score}")
    else:
        # Calculate score based on percentage difference
        diff = abs(word_count - target_words)

        # Different approaches for scoring:

        # 1. Linear penalty (simpler)
        # max_diff = max(target_words, 50)  # Cap the maximum difference considered
        # penalty = min(diff / max_diff, 1.0)
        # score = max(format_score, max_score * (1 - penalty))

        # 2. Exponential decay (rewards closer attempts better)
        # For target_words of 100, error of 1 word → 99% of max_score
        # For target_words of 100, error of 10 words → 90% of max_score
        # For target_words of 100, error of 50 words → 61% of max_score
        decay_factor = 0.1 * target_words  # Adjust scaling based on target length
        score = max(format_score, max_score * math.exp(-diff / decay_factor))

        if do_print:
            print(f"Word count off by {diff}. Score: {score}")

    return score