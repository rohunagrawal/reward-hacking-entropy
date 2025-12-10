
export PYTHONPATH=:${PYTHONPATH}
eval_batch_size=16


YOUR_PORT_NUMBER=8000
# docker run -it -p $YOUR_PORT_NUMBER:8080 volcengine/sandbox-fusion:server-20250609
# docker run -it -p 8000:8080 volcengine/sandbox-fusion:server-20250609
filter_difficulty="Easy"
training_output_dir="outputs/rl-leetcode/meta-llama/Llama-3.2-3B/g/${filter_difficulty}"
do_baseline=True    # Can just run baseline once FOR EACH BIN. The result will be saved under training_output_dir
baseline_result_fn="outputs/rl-leetcode/meta-llama/Llama-3.2-3B/g/${filter_difficulty}/heldout_outputs/baseline_outputs.jsonl"   # provide the path here FOR EACH BIN when do_baseline=False to avoid recomputing baseline. (Lorena: sorry for being lazy)

python analysis/measure_f_g_on_heldout.py \
    training_output_dir=${training_output_dir} \
    sandbox_url="http://localhost:$YOUR_PORT_NUMBER/run_code" \
    eval_batch_size=$eval_batch_size \
    do_baseline=$do_baseline \
    baseline_result_fn=${baseline_result_fn}

#bash measure_f_g_on_heldout.sh