#! /bin/csh -f
#run empirical question
cp -rf ./rnn.py ./rnn_bk.py
cp -rf ./rnn_empirical_questions.py ./rnn.py
#Q5.1
#python3 ./rnn.py --train_data ./data/HW7_large_stories/train_stories.json --val_data ./data/HW7_large_stories/valid_stories.json --train_losses_out large_train_losses.txt --val_losses_out large_val_losses.txt --metrics_out large_metrics.txt --embed_dim 128 --hidden_dim 128 --dk 128 --dv 128 --num_sequence 50000 --batch_size 128
python3 ./rnn.py --train_data ./data/HW7_large_stories/train_stories.json --val_data ./data/HW7_large_stories/valid_stories.json --train_losses_out large_train_losses.txt --val_losses_out large_val_losses.txt --metrics_out large_metrics.txt --embed_dim 128 --hidden_dim 128 --dk 128 --dv 128 --num_sequence 50000 --batch_size 128

#Q5.4
#python3 ./rnn.py --train_data ./data/HW7_large_stories/train_stories.json --val_data ./data/HW7_large_stories/valid_stories.json --train_losses_out large_train_losses.txt --val_losses_out large_val_losses.txt --metrics_out large_metrics.txt --embed_dim 512 --hidden_dim 512 --dk 256 --dv 256 --num_sequence 250000 --batch_size 128
mv ./rnn_bk.py ./rnn.py
rm -rf ./rnn.bk.py