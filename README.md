# Setup

Set up your favorite virtual environment and then:
```
git clone git@github.com:rohunagrawal/reward-hacking-entropy.git
cd reward-hacking-entropy
conda create -n hack python==3.11
conda activate hack
pip install -r requirements.txt

wandb login
hf auth login
```

# Set your tinker API key
```
export TINKER_API_KEY=...
```

# Prepare dataset
- model_name_path and bin_num only matter if filter_by==init_rollout_entropy
```bash
python data/prep_dataset.py \
    filter_by=difficulty \
    train_ratio=0.8 \   
    model_name_path="meta-llama/Llama-3.2-3B" \       
    bin_num=3
```

# Reward Function for Coding
- Host [SandBox Fusion](https://bytedance.github.io/SandboxFusion/docs/docs/get-started#local-deployment) (create a sandbox to test code safely):
  - ```docker run -it -p YOUR_PORT_NUMBER:8080 volcengine/sandbox-fusion:server-20250609``` (Uvicorn always starts on port 8080 in docker. Map it to some port number on your machine)
  - Test connection:
    ```
    curl 'http://localhost:YOUR_PORT_NUMBER/run_code' -H 'Content-Type: application/json'   --data-raw '{"code": "print(\"Hello, world!\")", "language": "python"}'
    ```
  - Pass in the url when creating the LeetCode() object below.
- Create LeetCode() object: class defined in [leetcode.py](reward_function/leetcode.py)
  - Entry function: ```process_code_result()```. Returns a dictionary with ```g_score``` (compilability) and ```f_score``` (correctness).
  - ```g_type``` options:
    - ```is_compilable```: syntax-only check.
    - ```llm_judge```: LLM-as-a-judge on compilability. Provide either a Tinker model path (```g_judge_model_path=tinker://...```) or a base model name (```g_judge_model_name=...```, e.g., ```meta-llama/Llama-3.2-3B```).
  - ```check_correctness()```: use SandBox Fusion. Fall back to prime_code if SandBox Fusion is not available.
  - Training reward defaults to ```g_score``` only.

# Run an RL training run
Modify port number and task difficulty in [train.sh](train.sh)
```
bash train.sh
```
- Checkpoints: 
  - Name of ckpt is f"{config.log_path.replace('/', '_')}_checkpoint_{step:06d}". (I think) they are saved remotely on Tinker's server
    - You'll get something like name=tinker://953697c9-8560-5244-a341-caf888b3beeb:train:0/weights/outputs_test_checkpoint_000001
  - The names are saved in local txt file:
    ```
    # Output to txt file
    with open(os.path.join(config.log_path, f"checkpoint_names.txt"), "a") as f:
        f.write(saved_path + "\n")
    with open(os.path.join(config.log_path, "latest_checkpoint.txt"), "w") as f:
        f.write(saved_path + "\n")
    ```
  - If you use the same log_path, the code will try to read in latest_checkpoint.txt and resume from there.

# Get f & g of ckpts on held-out dataset
- Run: The script will read in model_name, filter_by and filter_difficulty from the config.json used during training to load the corresponding model and held out dataset.
  ```bash
  export PYTHONPATH=:${PYTHONPATH}
  
  YOUR_PORT_NUMBER=8000
  eval_batch_size=16
  
  # docker run -it -p $YOUR_PORT_NUMBER:8080 volcengine/sandbox-fusion:server-20250609
  # docker run -it -p 8001:8080 volcengine/sandbox-fusion:server-20250609
  
  training_output_dir="outputs/rl-leetcode/meta-llama/Llama-3.2-3B/g/Easy"
  do_baseline=True    # Can just run baseline once FOR EACH BIN. The result will be saved under training_output_dir
  baseline_result_fn="outputs/rl-leetcode/meta-llama/Llama-3.2-3B/g/Easy/baseline/outputs.jsonl"   # provide the path here FOR EACH BIN when do_baseline=False to avoid recomputing baseline. (Lorena: I"m being lazy :\)
  
  python analysis/measure_f_g_on_heldout.py \
      training_output_dir=${training_output_dir} \
      sandbox_url="http://localhost:$YOUR_PORT_NUMBER/run_code" \
      eval_batch_size=$eval_batch_size \
      do_baseline=$do_baseline \
      baseline_result_fn=${baseline_result_fn}
  ```
- Expected outputs:
  - If do_baseline=True:
    - Sampling outputs: ```${training_output_dir}/heldout_outputs/baseline_outputs.jsonl```
    - f/g scores: ```${training_output_dir}/baseline_avg_heldout_f_g_scores.json```
  - Sampling outputs: ```${training_output_dir}/heldout_outputs/${ckpt_path}/outputs.jsonl```
  - f/g scores: ```${training_output_dir}/avg_heldout_f_g_scores.json```: 
    ```json
    {
        "baseline": {"avg_f": 0, "avg_g": 0, "avg_g_minus_f": 0},     // if do_baseline=False, will try to read from baseline_result_fn
        "checkpoint_name_1": {"avg_f": 0, "avg_g": 0, "avg_g_minus_f": 0},
        "checkpoint_name_2": {...}
    }
    ```
  - Plot of g-f scores across baseline and ckpts: ```${training_output_dir}/heldout_g_minus_f_scores.png```
- Can replot baseline & ckpts scores with:
  ```
  plot_scores(os.path.join("outputs/rl-leetcode/meta-llama/Llama-3.2-3B/g/Easy", "avg_heldout_f_g_scores.json"))
  ```
