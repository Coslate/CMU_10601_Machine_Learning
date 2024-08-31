
'''
    Author      : Patrick Chen
    Date        : 2024/08/27
'''

import numpy as np
import sys


#########################
#     Main-Routine      #
#########################
def main():
    #Process the argument
    print(f"> ArgumentParser...")
    (train_in_file, test_in_file, train_out_file, test_out_file, metrix_out_file, is_debug) = argumentParser()

    # Read input data.
    print(f"> readData()...")
    train_x, train_y = readData(train_in_file)
    test_x, test_y = readData(test_in_file)

    # Training.
    print(f"> MajorityVoteModel.train()...")
    model = MajorityVoteModel()
    model.train(train_x, train_y)

    # Predicting.
    print(f"> MajorityVoteModel.predict()...")
    predict_train_x = model.predict(train_x)
    predict_test_x = model.predict(test_x)

    # Evaluating.
    error_train = evaluateMetrixAccuracy(train_y, predict_train_x)
    error_test = evaluateMetrixAccuracy(test_y, predict_test_x)

    # WriteOut.
    print(f"> writeOutFile()...")
    writeOutFile(train_out_file, predict_train_x)
    writeOutFile(test_out_file, predict_test_x)
    writeOutMetrix(metrix_out_file, error_train, error_test)


    if(is_debug):
        print(f"train_in_file = {train_in_file}")
        print(f"test_in_file = {test_in_file}")
        print(f"train_out_file = {train_out_file}")
        print(f"test_out_file = {test_out_file}")
        print(f"metrix_out_file = {metrix_out_file}")
        print(f"train_x = {train_x[0]}")
        print(f"train_x = {train_x[-1]}")
        print(f"train_y = {train_y}")
        print(f"predict_train_x.shape = {predict_train_x.shape}")
        print(f"predict_test_x.shape = {predict_test_x.shape}")

#########################
#     Sub-Routine       #
#########################
def argumentParser() -> (str, str, str, str, str, int):
    train_in_file   = None
    test_in_file    = None
    train_out_file  = None
    test_out_file   = None
    metrix_out_file = None
    is_debug        = 0
    argc            = len(sys.argv)

    #.tsv
    if argc >= 2: train_in_file   = sys.argv[1]
    #.tsv
    if argc >= 3: test_in_file    = sys.argv[2]
    #.txt
    if argc >= 4: train_out_file  = sys.argv[3]
    #.txt
    if argc >= 5: test_out_file   = sys.argv[4]
    #.txt
    if argc >= 6: metrix_out_file = sys.argv[5]  

    if argc >= 7: is_debug        = int(sys.argv[6])

    if train_in_file is None:
        print(f"Error: Input argument argv[1] for train input file is not set.")
    if test_in_file is None:
        print(f"Error: Input argument argv[2] for test input file is not set.")
    if train_out_file is None:
        print(f"Error: Input argument argv[3] for train output file is not set.")
    if test_out_file is None:
        print(f"Error: Input argument argv[4] for test output file is not set.")
    if metrix_out_file is None:
        print(f"Error: Input argument argv[5] for metrix output file is not set.")                                

    return (train_in_file, test_in_file, train_out_file, test_out_file, metrix_out_file, is_debug)

def readData(input_file: str) -> ([], []):
    data = np.genfromtxt(input_file, delimiter="\t", dtype=None, encoding=None)
    # remove title
    # data[1:, 0:-1]
    data_x = np.array([[int(x_element) for x_element in x[0:-1]] for x in data[1:, :]])
    # data[1:, -1]
    data_y = np.array([int(y_element) for y_element in data[1:, -1]])
    return data_x, data_y

def writeOutFile(output_file: str, output_content_arr: []) -> None :
    with open(output_file, 'w') as f:
        for data_line in output_content_arr:
            f.write(f"{data_line[0]}\n")

def writeOutMetrix(output_file: str, error_train: float, error_test: float) -> None:
    with open(output_file, 'w') as f:
        f.write(f"error (train): {error_train}\n")
        f.write(f"error (test): {error_test}\n")

def evaluateMetrixAccuracy(gt_y: [], pd_y: [[]]) -> float:
    len_gt_y = len(gt_y)
    len_pd_y = len(pd_y)
    if len_gt_y != len_pd_y:
        print(f"Error: length of gt_y, {len_gt_y}, deos not equal to length of pd_y, {len_pd_y}.")

    cnt_err = 0
    for index, true_y in enumerate(gt_y):
        if true_y != pd_y[index][0]:
            cnt_err += 1

    return cnt_err/len_gt_y


class MajorityVoteModel:
    def __init__(self) -> None:
        self.model_label = 0

    def train(self, train_x: [], train_y: []) -> None:
        count = {}
        max_label_cnt = -float('inf')
        max_label = None

        for label in train_y:
            if label not in count:
                count[label] = 0
            else:
                count[label] += 1

        for label, cnt in count.items():
            if cnt > max_label_cnt:

                max_label_cnt = cnt
                max_label = label
            elif cnt == max_label_cnt:
                if label > max_label:
                    max_label = label
        
        self.model_label = max_label

    def hx(self, x: []) -> [[]]:
        ans = np.array([self.model_label for xelement in x])
        ans = ans.reshape([x.shape[0], 1])
        return ans

    def predict(self, x: []) -> [[]]:
        return self.hx(x)

#---------------Execution---------------#
if __name__ == '__main__':
    main()