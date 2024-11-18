#! /bin/csh -f
#run empirical question
python3 ./rnn_empirical_questions.py --train_data ./data/HW7_large_stories/train_stories.json --val_data ./data/HW7_large_stories/valid_stories.json --train_losses_out large_train_losses.txt --val_losses_out large_val_losses.txt --metrics_out large_metrics.txt --embed_dim 512 --hidden_dim 512 --dk 256 --dv 256 --num_sequence 250000 --batch_size 128