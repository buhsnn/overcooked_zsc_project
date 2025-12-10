CUDA_VISIBLE_DEVICES=4 python main.py \
    --train_steps_per_iter $1 \
    --w_regret 0.0 \
    --w_novelty 0.0 \
    --w_progress 0.0 \
