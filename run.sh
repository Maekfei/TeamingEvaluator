python train.py \
    --train_years 2006 2014 \
    --test_years 2015 2018 \
    --lr 1e-2 \
    --weight_decay 5e-5 \
    --hidden_dim 32 \
    --num_layers 3 \
    --epochs 800 \
    --cold_start_prob 0.5 \
    --beta 0 \
    --grad_clip 1.0 \
    --lr_patience 4 \
    --lr_factor 0.5 \
    --early_stop_patience 8 \
    --min_lr 1e-5 \
    --eval_mode team > base_3_2.txt 2>&1

python train.py \
    --train_years 2006 2014 \
    --test_years 2015 2018 \
    --lr 1e-2 \
    --weight_decay 5e-5 \
    --hidden_dim 32 \
    --num_layers 3 \
    --epochs 800 \
    --cold_start_prob 0.5 \
    --beta 0 \
    --grad_clip 1.0 \
    --lr_patience 4 \
    --lr_factor 0.5 \
    --early_stop_patience 8 \
    --min_lr 1e-5 \
    --eval_mode team > base_3_3.txt 2>&1

python train.py \
    --train_years 2006 2014 \
    --test_years 2015 2018 \
    --lr 1e-2 \
    --weight_decay 5e-5 \
    --hidden_dim 32 \
    --num_layers 3 \
    --epochs 800 \
    --cold_start_prob 0.5 \
    --beta 0 \
    --grad_clip 1.0 \
    --lr_patience 4 \
    --lr_factor 0.5 \
    --early_stop_patience 8 \
    --min_lr 1e-5 \
    --eval_mode team > base_3_4.txt 2>&1

python train.py \
    --train_years 2006 2014 \
    --test_years 2015 2018 \
    --lr 1e-2 \
    --weight_decay 5e-5 \
    --hidden_dim 32 \
    --num_layers 3 \
    --epochs 800 \
    --cold_start_prob 0.5 \
    --beta 0 \
    --grad_clip 1.0 \
    --lr_patience 4 \
    --lr_factor 0.5 \
    --early_stop_patience 8 \
    --min_lr 1e-5 \
    --eval_mode team > base_3_5.txt 2>&1
