python -u train.py \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETT-small/ETTh2.csv \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --epoch_count 10 \

python -u train.py \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path  ETT-small/ETTh2.csv \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --epoch_count 10 \

  python -u train.py \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path  ETT-small/ETTh2.csv \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --epoch_count 10 \

  python -u train.py \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path  ETT-small/ETTh2.csv \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --epoch_count 10 \
