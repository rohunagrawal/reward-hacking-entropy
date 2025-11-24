
export TINKER_API_KEY=YOUR_KEY

filter_difficulty="Easy"
YOUR_PORT_NUMBER=8000

# docker run -it -p $YOUR_PORT_NUMBER:8080 volcengine/sandbox-fusion:server-20250609
# docker run -it -p 8001:8080 volcengine/sandbox-fusion:server-20250609

python train.py \
    filter_difficulty=$filter_difficulty \
    sandbox_url="http://localhost:$YOUR_PORT_NUMBER/run_code" \
    log_path="outputs/rl-leetcode/llama-3.2-1b/$filter_difficulty" \
    epochs=5