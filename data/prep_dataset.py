
import chz
from datasets import load_dataset, DatasetDict, ClassLabel

@chz.chz
class Config:
    output: str = "data/leetcode"
    split: str = "train"
    filter_by: str = "difficulty"  # difficulty or init_rollout_entropy
    train_ratio: float = 0.8  # Ratio of data to use for training (rest for held-out)
    seed: int = 42
    model_name_path: str = "meta-llama/Llama-3.2-3B"  # Only matters if filter_by is initial rollout entropy
    bin_num: int = 3  # Only matters if filter_by is init_rollout_entropy

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

def main(config: Config):
    print("Loading LeetCode dataset...")
    dataset = load_dataset("newfacade/LeetCodeDataset", split=config.split)
    
    print(f"Dataset loaded: {len(dataset)} problems")
    
    # Process the dataset
    processed_dataset = dataset.map(process_fn)
    
    # Balance categories to have the same number of samples
    print(f"Balancing dataset by '{config.filter_by}'...")
    
    # Group by category
    category_groups = {}
    for i, example in enumerate(processed_dataset):
        category = example[config.filter_by]
        if category not in category_groups:
            category_groups[category] = []
        category_groups[category].append(i)
    
    # Find minimum count
    min_count = min(len(indices) for indices in category_groups.values())
    print(f"Minimum category count: {min_count}")
    print(f"Categories: {list(category_groups.keys())}")
    
    # Sample min_count from each category
    import random
    random.seed(config.seed)
    balanced_indices = []
    for category, indices in category_groups.items():
        sampled = random.sample(indices, min_count)
        balanced_indices.extend(sampled)
        print(f"  {category}: {len(indices)} -> {min_count} samples")
    
    # Create balanced dataset
    balanced_dataset = processed_dataset.select(balanced_indices)
    print(f"Balanced dataset: {len(balanced_dataset)} problems")
    
    # Convert filter_by column to ClassLabel for stratification
    categories = sorted(category_groups.keys())
    class_label = ClassLabel(names=categories)
    
    def add_class_label(example):
        example[f"{config.filter_by}_label"] = class_label.str2int(str(example[config.filter_by]))
        return example
    
    balanced_dataset = balanced_dataset.map(add_class_label)
    
    # Update feature to be ClassLabel type
    features = balanced_dataset.features.copy()
    features[f"{config.filter_by}_label"] = class_label
    balanced_dataset = balanced_dataset.cast(features)
    
    # Split into train and held-out sets with stratification by filter_by
    print(f"Splitting dataset with train_ratio={config.train_ratio}, stratified by '{config.filter_by}'...")
    split_dataset = balanced_dataset.train_test_split(
        test_size=1.0 - config.train_ratio,
        seed=config.seed,
        stratify_by_column=f"{config.filter_by}_label"
    )
    
    train_dataset = split_dataset['train']
    heldout_dataset = split_dataset['test']
    
    print(f"Train set: {len(train_dataset)} problems")
    print(f"Held-out set: {len(heldout_dataset)} problems")
    


    # Create DatasetDict with both splits
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'dev': heldout_dataset
    })

    # Save the dataset
    print(f"Saving dataset to '{config.output}'...")
    dataset_dict.save_to_disk(config.output)

    # Output distribution to json
    # Print distribution by filter_by for both splits
    print(f"\nDistribution by '{config.filter_by}':")
    train_counts = {}
    heldout_counts = {}

    for example in train_dataset:
        category = example[config.filter_by]
        train_counts[category] = train_counts.get(category, 0) + 1

    for example in heldout_dataset:
        category = example[config.filter_by]
        heldout_counts[category] = heldout_counts.get(category, 0) + 1

    for category in sorted(set(list(train_counts.keys()) + list(heldout_counts.keys()))):
        train_count = train_counts.get(category, 0)
        heldout_count = heldout_counts.get(category, 0)
        total = train_count + heldout_count
        print(f"  {category}: Train={train_count} ({train_count/total*100:.1f}%), Held-out={heldout_count} ({heldout_count/total*100:.1f}%)")

    import json
    distribution = {
        "train": train_counts,
        "dev": heldout_counts
    }
    with open(f"{config.output}/filter_by_distribution.json", "w") as f:
        json.dump(distribution, f, indent=4)

    # print("Done!")
    # print(f"\nSample problems from train split:")
    # for i in range(min(3, len(train_dataset))):
    #     print(f"Query: {train_dataset[i]['query']}")
    #     print(f"Import prefix: {train_dataset[i]['import_prefix']}")
    #     print(f"Entry point: {train_dataset[i]['entry_point']}")
    #     print(f"Difficulty: {train_dataset[i]['difficulty']}")
    #     print()


if __name__ == '__main__':
    chz.nested_entrypoint(main)

