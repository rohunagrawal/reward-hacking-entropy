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
```
python data/prep_dataset.py
```
- Split size:
  - Easy: 638
  - Medium: 1397
  - Hard: 606

# Reward Function for Coding
- Host [SandBox Fusion](https://bytedance.github.io/SandboxFusion/docs/docs/get-started#local-deployment) (create a sandbox to test code safely):
  - ```docker run -it -p YOUR_PORT_NUMBER:8080 volcengine/sandbox-fusion:server-20250609``` (Uvicorn always starts on port 8080 in docker. Map it to some port number on your machine)
  - Test connection:
    ```
    curl 'http://localhost:YOUR_PORT_NUMBER/run_code' -H 'Content-Type: application/json'   --data-raw '{"code": "print(\"Hello, world!\")", "language": "python"}'
    ```
  - Pass in the url when creating the LeetCode() object below.
- Create LeetCode() object: class defined in [leetcode.py](reward_function/leetcode.py)
  - Entry function: ```process_code_result()```. Returns a dictionary with ```is_compilable_reward``` and ```correctness_reward``` 
  - ```is_compilable(completion: str)```: only checking whether the code itself is compilable, not whether the whole code with imports and test cases is compilable. (may happen in sandbox fusion)
  - ```check_correctness()```: use SandBox Fusion. Fall back to prime_code if SandBox Fusion is not available.

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
