
export PYTHONPATH=:${PYTHONPATH}

YOUR_PORT_NUMBER=8000
eval_batch_size=16

# docker run -it -p $YOUR_PORT_NUMBER:8080 volcengine/sandbox-fusion:server-20250609
# docker run -it -p 8001:8080 volcengine/sandbox-fusion:server-20250609

python analysis/measure_f_g_on_heldout.py \
    training_output_dir="outputs/rl-leetcode/meta-llama/Llama-3.2-3B/100f_plus_g/Easy" \
    sandbox_url="http://localhost:$YOUR_PORT_NUMBER" \
    eval_batch_size=$eval_batch_size

#bash measure_f_g_on_heldout.sh