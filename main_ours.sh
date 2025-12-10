tspi=(
    5000
    10000
    20000
    40000
)

for tspi_val in "${tspi[@]}"; do
    CUDA_VISIBLE_DEVICES=4 python main.py \
        --train_steps_per_iter $tspi_val \
        --w_novelty 0.01 &
    CUDA_VISIBLE_DEVICES=5 python main.py \
        --train_steps_per_iter $tspi_val \
        --w_novelty 0.03 &
    CUDA_VISIBLE_DEVICES=6 python main.py \
        --train_steps_per_iter $tspi_val \
        --w_novelty 0.1 &
done
