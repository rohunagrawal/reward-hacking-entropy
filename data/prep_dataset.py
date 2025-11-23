#!/usr/bin/env python3

from datasets import load_dataset, Dataset, DatasetDict
import argparse

def process_fn(entry):
    all_test_inputs = []
    all_test_outputs = []
    for io_pair in entry["input_output"]:
        all_test_inputs.append(io_pair["input"])
        all_test_outputs.append(io_pair["output"])

    data = {
        "query": entry["query"],
        "difficulty": entry["difficulty"],
        "import_prefix": entry["prompt"],
        "test_inputs_outputs": {        # reformatted for SandboxFusion
            "inputs": all_test_inputs,
            "outputs": all_test_outputs,
            "fn_name": entry["entry_point"]
        },
    }

    return data


def main():
    parser = argparse.ArgumentParser(description='Load LeetCode dataset')
    parser.add_argument('--output', type=str, default='leetcode',
                       help='Output dataset name/path')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split to use')
    
    args = parser.parse_args()
    
    print("Loading LeetCode dataset...")
    dataset = load_dataset("newfacade/LeetCodeDataset", split=args.split)
    
    print(f"Dataset loaded: {len(dataset)} problems")
    
    # Process the dataset
    processed_dataset = dataset.map(process_fn)
    
    # Save the dataset
    print(f"Saving dataset to '{args.output}'...")
    processed_dataset.save_to_disk(args.output)
    
    print("Done!")
    print(f"\nSample problems:")
    train_split = processed_dataset
    for i in range(5):
        print(f"Query: {train_split[i]['query']}")
        print(f"Import prefix: {train_split[i]['import_prefix']}")
        print(f"Entry point: {train_split[i]['entry_point']}")
        print(f"Test: {train_split[i]['test']}")
        print()


if __name__ == '__main__':
    main()

