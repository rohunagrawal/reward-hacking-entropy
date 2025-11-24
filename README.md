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
```
python train.py
```