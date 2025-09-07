python -u train.py \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path exchange_rate/exchange_rate.csv \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \


python -u train.py \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path exchange_rate/exchange_rate.csv \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \


  python -u train.py \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path exchange_rate/exchange_rate.csv \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \


  python -u train.py \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path exchange_rate/exchange_rate.csv \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \

