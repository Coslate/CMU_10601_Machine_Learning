#! /usr/bin/csh -f

python ./majority_vote.py ./handout/education_train.tsv ./handout/education_test.tsv  ./output/education_train_labels.txt ./output/education_test_labels.txt ./output/education_metrics.txt 0
python ./majority_vote.py ./handout/heart_train.tsv ./handout/heart_test.tsv  ./output/heart_train_labels.txt ./output/heart_test_labels.txt ./output/heart_metrics.txt 0
