
'''
    Author      : Patrick Chen
    Date        : 2024/09/09
'''

import numpy as np
import sys
import subprocess
import re
import matplotlib.pyplot as plt


#########################
#     Main-Routine      #
#########################
def main():
    #Process the argument
    #print(f"> argumentParser...")
    (in_train_file, in_test_file, max_depth, out_plot_file) = argumentParser()
    print(f"in_train_file = {in_train_file}")
    print(f"in_test_file = {in_test_file}")
    print(f"max_depth = {max_depth}")
    print(f"out_plot_file = {out_plot_file}")
    result_train_err = []
    result_test_err = []
    x_max_depth = [x for x in range(max_depth+1)]

    for max_depth_i in range(max_depth+1):
        subprocess.run(["python", "decision_tree.py", f"{in_train_file}", f"{in_test_file}", f"{max_depth_i}", f"./plot_folder/heart_{max_depth_i}_train.txt", f"./plot_folder/heart_{max_depth_i}_test.txt", f"./plot_folder/heart_{max_depth_i}_metrics.txt", f"./plot_folder/heart_{max_depth_i}_print.txt"])

        with open(f'./plot_folder/heart_{max_depth_i}_metrics.txt', 'r') as file:
            content = file.read()
        pattern = r'error\(train\):\s*([0-9.]+)|error\(test\):\s*([0-9.]+)'
        matches = re.findall(pattern, content)

        result_train_err.append(round(float(matches[0][0]), 4))
        result_test_err.append(round(float(matches[1][1]), 4))

    #result_test_err.reverse()
    #result_train_err.reverse()

    # Create a figure and axis
    plt.figure()

    # Plot the first line
    plt.plot(x_max_depth, result_train_err, label='Train Error', marker='o')

    # Plot the second line
    plt.plot(x_max_depth, result_test_err, label='Test Error', marker='o')

    # Add labels and a title
    plt.xlabel('Max_Depth')
    plt.ylabel('Error Rate')
    plt.title('Train and Test Error VS Max_Depth Over heart Dataset')

    # Add a legend to distinguish the lines
    plt.legend()

    # Show the plot
    #plt.show()    

    # Save the image
    plt.savefig(f'{out_plot_file}')

#########################
#     Sub-Routine       #
#########################
def argumentParser() -> (str, str, int, str):
    in_train_file  = None
    in_test_file   = None
    max_depth      = 0
    out_plot_file  = None
    argc           = len(sys.argv)

    #.tsv
    if argc >= 2: in_train_file   = sys.argv[1]
    #.tsv
    if argc >= 3: in_test_file    = sys.argv[2]
    if argc >= 4: max_depth       = int(sys.argv[3])
    if argc >= 5: out_plot_file   = sys.argv[4]

    if in_train_file is None:
        print(f"Error: Input argument argv[1] for input train file is not set.")
    if in_test_file is None:
        print(f"Error: Input argument argv[2] for input test file is not set.")
    if max_depth is None:
        print(f"Error: Input argument argv[3] for max_depth is not set.")
    if out_plot_file is None:
        print(f"Error: Input argument argv[4] for out_plot_file is not set.")

    return (in_train_file, in_test_file, max_depth, out_plot_file)

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