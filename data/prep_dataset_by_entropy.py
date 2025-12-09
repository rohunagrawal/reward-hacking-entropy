import json
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

import chz
import datasets
import numpy as np
from tqdm import tqdm
import tinker
from datasets import DatasetDict, ClassLabel
from tinker import types
from transformers import AutoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    output: str = "data/leetcode_entropy"
    split: str = "train"
    train_ratio: float = 0.8
    seed: int = 42
    base_url: str | None = None
    model_name: str = "meta-llama/Llama-3.2-3B"
    lora_rank: int = 16  # Needed to init training client before saving sampler weights
    max_tokens: int = 1024
    temperature: float = 1.0
    samples_per_problem: int = 1
    bin_num: int = 3
    n_threads: int = 10


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


class SimpleRenderer:
    """Matches the renderer used in train.py for prompt construction."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.stop_sequences = ["<end_of_turn>", "<eos>", "<|eot_id|>"]

        if not self.tokenizer.chat_template:
            # Default Llama-3 style template
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n"
                "{{ message['content'] }}<|eot_id|>"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
                "{% endif %}"
            )

    def build_generation_prompt(self, messages: List[Dict[str, str]]) -> types.ModelInput:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        return types.ModelInput.from_ints(tokens=tokens)

    def parse_response(self, tokens: List[int]) -> tuple[Dict[str, str], None]:
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return {"role": "assistant", "content": text}, None

    def get_stop_sequences(self) -> List[str]:
        return self.stop_sequences


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
        "test_inputs_outputs": {
            "inputs": all_test_inputs,
            "outputs": all_test_outputs,
            "fn_name": entry["entry_point"],
        },
    }

    return data


def compute_entropy_for_problem(
    renderer: SimpleRenderer,
    sampling_client: tinker.SamplingClient,
    sampling_params: tinker.types.SamplingParams,
    query: str,
) -> float:
    convo = [{"role": "user", "content": query}]
    model_input = renderer.build_generation_prompt(convo)
    prompt_tokens = model_input.to_ints()

    future = sampling_client.sample(
        prompt=model_input,
        num_samples=1,
        sampling_params=sampling_params,
    )
    sample_result = future.result()
    sampled_tokens = sample_result.sequences[0].tokens
    sampled_logprobs = sample_result.sequences[0].logprobs

    if sampled_logprobs is None or len(sampled_logprobs) == 0:
        return 0.0

    # Approximate entropy with average negative log-likelihood of the sampled tokens
    avg_nll = -sum(sampled_logprobs) / len(sampled_logprobs)
    logger.debug(
        "Computed entropy %.4f for prompt length %d and completion length %d",
        avg_nll,
        len(prompt_tokens),
        len(sampled_tokens),
    )
    return avg_nll


def main(config: Config):
    set_seed(config.seed)
    logger.info("Loading LeetCode dataset split '%s'...", config.split)
    dataset = datasets.load_dataset("newfacade/LeetCodeDataset", split=config.split)
    logger.info("Dataset loaded: %d problems", len(dataset))

    processed_dataset = dataset.map(process_fn)

    logger.info("Setting up tokenizer and sampling client for entropy computation...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    renderer = SimpleRenderer(tokenizer)

    service_client = tinker.ServiceClient(base_url=config.base_url)
    # Must save weights first before creating a sampling client (see train.py)
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name, rank=config.lora_rank
    )
    sampling_path = training_client.save_weights_for_sampler(name="prep_entropy").result().path
    sampling_client = service_client.create_sampling_client(model_path=sampling_path)

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature,
    )

    def entropy_worker(query: str) -> float:
        try:
            entropy_vals = [
                compute_entropy_for_problem(renderer, sampling_client, sampling_params, query)
                for _ in range(config.samples_per_problem)
            ]
            return float(np.mean(entropy_vals))
        except Exception as e:
            logger.error("Failed to compute entropy for query: %s", e)
            return 0.0

    entropies: list[float] = []
    queries = [example["query"] for example in processed_dataset]
    with ThreadPoolExecutor(max_workers=config.n_threads) as executor:
        for entropy in tqdm(
            executor.map(entropy_worker, queries),
            desc="Computing entropy",
            total=len(queries),
        ):
            entropies.append(entropy)

    assert len(entropies) == len(processed_dataset), "Entropy list length mismatch"

    # Bin entropies into equal sized bins
    sorted_indices = np.argsort(entropies)
    bin_indices = np.array_split(sorted_indices, config.bin_num)
    entropy_bins = [0] * len(entropies)
    bin_ranges = []

    for bin_id, indices in enumerate(bin_indices, start=1):
        if len(indices) == 0:
            bin_ranges.append({"bin": bin_id, "min_entropy": None, "max_entropy": None, "count": 0})
            continue

        bin_entropy_values = [entropies[int(i)] for i in indices]
        min_entropy = float(min(bin_entropy_values))
        max_entropy = float(max(bin_entropy_values))
        bin_ranges.append(
            {
                "bin": bin_id,
                "min_entropy": min_entropy,
                "max_entropy": max_entropy,
                "count": len(indices),
            }
        )
        for idx in indices:
            entropy_bins[int(idx)] = bin_id

    processed_dataset = processed_dataset.add_column("init_rollout_entropy", entropies)
    processed_dataset = processed_dataset.add_column("entropy_bin", entropy_bins)

    # Balance categories to have the same number of samples
    logger.info("Balancing dataset across entropy bins...")
    category_groups: dict[int, list[int]] = {}
    for i, example in enumerate(processed_dataset):
        category = example["entropy_bin"]
        category_groups.setdefault(category, []).append(i)

    min_count = min(len(indices) for indices in category_groups.values())
    logger.info("Minimum bin count: %d | Bins: %s", min_count, list(category_groups.keys()))

    balanced_indices = []
    for category, indices in category_groups.items():
        sampled = random.sample(indices, min_count)
        balanced_indices.extend(sampled)
        logger.info("  Bin %s: %d -> %d samples", category, len(indices), min_count)

    balanced_dataset = processed_dataset.select(balanced_indices)
    logger.info("Balanced dataset size: %d", len(balanced_dataset))

    # Convert bin to ClassLabel for stratification
    categories = sorted(category_groups.keys())
    class_label = ClassLabel(names=[str(c) for c in categories])

    def add_class_label(example):
        example["entropy_bin_label"] = class_label.str2int(str(example["entropy_bin"]))
        return example

    balanced_dataset = balanced_dataset.map(add_class_label)
    features = balanced_dataset.features.copy()
    features["entropy_bin_label"] = class_label
    balanced_dataset = balanced_dataset.cast(features)

    logger.info(
        "Splitting dataset with train_ratio=%.2f, stratified by entropy_bin...",
        config.train_ratio,
    )
    split_dataset = balanced_dataset.train_test_split(
        test_size=1.0 - config.train_ratio,
        seed=config.seed,
        stratify_by_column="entropy_bin_label",
    )

    train_dataset = split_dataset["train"]
    heldout_dataset = split_dataset["test"]

    dataset_dict = DatasetDict({"train": train_dataset, "dev": heldout_dataset})

    logger.info("Saving dataset to '%s'...", config.output)
    os.makedirs(config.output, exist_ok=True)
    dataset_dict.save_to_disk(config.output)

    bin_ranges_path = os.path.join(config.output, "entropy_bin_ranges.json")
    with open(bin_ranges_path, "w") as f:
        json.dump(bin_ranges, f, indent=4)
    logger.info("Saved entropy bin ranges to %s", bin_ranges_path)

    # Print distribution
    train_counts = {}
    heldout_counts = {}
    for example in train_dataset:
        cat = example["entropy_bin"]
        train_counts[cat] = train_counts.get(cat, 0) + 1
    for example in heldout_dataset:
        cat = example["entropy_bin"]
        heldout_counts[cat] = heldout_counts.get(cat, 0) + 1

    logger.info("Distribution by entropy_bin:")
    for category in sorted(set(list(train_counts.keys()) + list(heldout_counts.keys()))):
        train_count = train_counts.get(category, 0)
        heldout_count = heldout_counts.get(category, 0)
        total = train_count + heldout_count
        logger.info(
            "  Bin %s: Train=%d (%.1f%%), Held-out=%d (%.1f%%)",
            category,
            train_count,
            (train_count / total * 100) if total else 0.0,
            heldout_count,
            (heldout_count / total * 100) if total else 0.0,
        )


if __name__ == "__main__":
    chz.nested_entrypoint(main)
