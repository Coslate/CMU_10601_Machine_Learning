
'''
    Author      : Patrick Chen
    Date        : 2024/09/09
'''

import numpy as np
import sys


#########################
#     Main-Routine      #
#########################
def main():
    #Process the argument
    print(f"> argumentParser...")
    (in_file, out_file, is_debug) = argumentParser()

    # Read input data.
    print(f"> readData()...")
    train_x, train_y = readData(in_file)

    # Training.
    print(f"> MajorityVoteModel.train()...")
    model = MajorityVoteModel()
    model.train(train_x, train_y)

    # Predicting.
    print(f"> MajorityVoteModel.predict()...")
    predict_train_x = model.predict(train_x)

    # Evaluating.
    print(f"> evaluateMetrixAccuracy()...")
    error = evaluateMetrixAccuracy(train_y, predict_train_x)
    entropy = evaluateEntropy(train_y)

    # WriteOut.
    print(f"> writeOutFile()...")
    writeOutMetrix(out_file, [entropy, error], ['entropy', 'error'])

    if(is_debug):
        print(f"in_file = {in_file}")
        print(f"out_file = {out_file}")

#########################
#     Sub-Routine       #
#########################
def argumentParser() -> (str, str, str, str, str, int):
    in_file     = None
    out_file    = None
    is_debug    = 0
    argc        = len(sys.argv)

    #.tsv
    if argc >= 2: in_file   = sys.argv[1]
    #.tsv
    if argc >= 3: out_file    = sys.argv[2]
    #.txt
    if argc >= 4: is_debug        = int(sys.argv[3])

    if in_file is None:
        print(f"Error: Input argument argv[1] for input file is not set.")
    if out_file is None:
        print(f"Error: Input argument argv[2] for output file is not set.")

    return (in_file, out_file, is_debug)

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
        f.write(f"error(train): {error_train}\n")
        f.write(f"error(test): {error_test}\n")

def writeOutMetrix(output_file: str, met: [], met_name: []) -> None:
    with open(output_file, 'w') as f:
        for name, value in zip(met_name, met):
            f.write(f"{name}: {value}\n")

def expCalc(p: float) -> float:
    return 0 if p == 0 else p*np.log2(p) 

def evaluateEntropy(y: []) -> float:
    total_y = len(y)
    p_y1 = np.sum(y == 1)/total_y
    p_y0 = np.sum(y == 0)/total_y

    return -(expCalc(p_y0) + expCalc(p_y1))


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