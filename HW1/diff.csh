#! /usr/bin/csh -f

diff ./output/heart_metrics.txt ./handout/example_output/heart_metrics.txt 
diff ./output/heart_train_labels.txt ./handout/example_output/heart_train_labels.txt 
diff ./output/heart_test_labels.txt ./handout/example_output/heart_test_labels.txt 
diff ./output/education_metrics.txt ./handout/example_output/education_metrics.txt 
diff ./output/education_train_labels.txt ./handout/example_output/education_train_labels.txt 
diff ./output/education_test_labels.txt ./handout/example_output/education_test_labels.txt 
