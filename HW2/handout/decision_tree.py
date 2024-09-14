import argparse
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Any

class Node:
    '''
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    '''
    def __init__(self):
        self.left = None
        self.right = None
        self.attr = None #string, for print_tree
        self.vote = None #only leaf node has its value
        self.depth = None # for print_tree
        self.y_stat0 = None # at current node before splitting, number of data with y==0, for print_tree
        self.y_stat1 = None # at current node before splitting, number of data with y==1, for print_tree

class Dataset:
    def __init__(self, features: [], data_x: NDArray[Tuple[Any, Any]], data_y: NDArray[Tuple[Any]]):
        self.feature = [x for x in features]
        self.data_x = np.copy(data_x)
        self.data_y = np.copy(data_y)

class dTreeModel:
    def __init__(self) -> None:
        self.root_node = None
        self.feature_map = {} #map['feature_name'] = index

    def expCalc(self, p: float) -> float:
        return 0 if p == 0 else p*np.log2(p) 

    def evaluateEntropy(self, y: []) -> float:
        total_y = len(y)
        if total_y != 0:
            p_y1 = np.sum(y == 1)/total_y
            p_y0 = np.sum(y == 0)/total_y
        else:
            p_y1 = 0
            p_y0 = 0

        return -(self.expCalc(p_y0) + self.expCalc(p_y1))

    def findBestAttr(self, Dv:Dataset) -> int:
        data_x_feature_name = Dv.feature
        data_x = Dv.data_x
        data_y = Dv.data_y

        #H_Y = self.evaluateEntropy(data_y)
        largest_mutial_info = -float('inf')
        largest_index = 0
        total_data_num = len(data_y)
        H_Y = self.evaluateEntropy(data_y)
        for index_column, x in enumerate(data_x_feature_name):
            H_YX = 0
            for x_v in (0, 1): #traverse all value in x column (feature)
                index = np.where(data_x[:, index_column]==x_v)[0]
                H_condY = self.evaluateEntropy(data_y[index])
                H_YX += len(index)/total_data_num*H_condY

            mutual_info = H_Y - H_YX
            if mutual_info > 0 and mutual_info > largest_mutial_info:
                largest_mutial_info = mutual_info
                largest_index = index_column

        return (largest_index, data_x_feature_name[largest_index])

    def nodeSplit(self, Dv: Dataset, depth: int) -> Node:
        current_node = Node()
        current_node.depth = depth
        current_node.y_stat0 = np.sum(Dv.data_y==0)
        current_node.y_stat1 = np.sum(Dv.data_y==1)
        empty_data_set = (len(Dv.data_y) == 0)
        all_label_same = (len(Dv.data_y) == len(np.where(Dv.data_y==0)[0])) or (len(Dv.data_y) == len(np.where(Dv.data_y==1)[0]))
        tree_too_deep  = (depth >= max_depth)

        all_feature_same = True
        for row in range(Dv.data_x.shape[0]-1):
            this_row = Dv.data_x[row, :]
            next_row = Dv.data_x[(row+1), :]
            all_feature_same = all_feature_same and all((this_row==next_row))

        if(empty_data_set or all_label_same or tree_too_deep or all_feature_same):
            maj_vote_model = MajorityVoteModel()
            maj_vote_model.train(Dv.data_x, Dv.data_y)
            current_node.vote = maj_vote_model.model_label
        else:
            x_attr = self.findBestAttr(Dv)
            current_node.attr     = x_attr
            x_attr_column         = Dv.data_x[:, x_attr[0]]
            remain_data_x         = np.delete(Dv.data_x, x_attr[0], axis=1)
            remain_feature        = [x for x in Dv.feature if x != x_attr[1]]

            # left subtree
            x_attr_column_0_index = np.where(x_attr_column==0)[0]
            new_data_y_0          = Dv.data_y[x_attr_column_0_index]
            new_data_x_0          = remain_data_x[x_attr_column_0_index, :]
            Dv_subset0            = Dataset(remain_feature, new_data_x_0, new_data_y_0)

            # right subtree
            x_attr_column_1_index = np.where(x_attr_column==1)[0]
            new_data_y_1          = Dv.data_y[x_attr_column_1_index]
            new_data_x_1          = remain_data_x[x_attr_column_1_index, :]
            Dv_subset1            = Dataset(remain_feature, new_data_x_1, new_data_y_1)

            # release memory
            del Dv

            # recursive splitting
            current_node.left     = self.nodeSplit(Dv_subset0, depth+1)# x_attr == 0
            current_node.right    = self.nodeSplit(Dv_subset1, depth+1)# x_attr == 1
        return current_node

    def train(self, data_x_feature_name:[], data_x: NDArray[Tuple[Any, Any]], data_y:NDArray[Tuple[Any]]) -> None:
        Dv = Dataset(data_x_feature_name, data_x, data_y)
        self.feature_map = {x: index for index, x in enumerate(data_x_feature_name)}
        self.root_node = self.nodeSplit(Dv, 0)

    def hx(self, x: NDArray[Tuple[Any]]) -> [[]]:
        current_node = self.root_node
        while True:
            if current_node.attr is None: #leaf node
                return current_node.vote
            else:
                if x[self.feature_map[current_node.attr[1]]] == 0:
                    current_node = current_node.left
                else:
                    current_node = current_node.right

    def predict(self, x_test: NDArray[Tuple[Any, Any]]) -> NDArray[Tuple[Any, Any]]:
        y_head = np.zeros((x_test.shape[0], 1), dtype=int)
        for index, x_inst in enumerate(x_test):
            y_head[index] = self.hx(x_inst)

        return y_head


class MajorityVoteModel:
    def __init__(self) -> None:
        self.model_label = 1

    def train(self, train_x: [], train_y: []) -> None:
        count = {}
        max_label_cnt = -float('inf')
        max_label = 1

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

def print_tree(node: Node, f):
    if node is None:
        return

    print_sentence = f"[{node.y_stat0} 0/{node.y_stat1} 1]"
    #print(print_sentence)
    f.write(print_sentence+'\n')
    if node.attr is not None:
        print_sentence = f"{(node.depth+1)*'| '}"
        #print(print_sentence, end='')
        f.write(print_sentence)

        print_sentence = f"{node.attr[1]} == 0: "
        #print(print_sentence, end='')
        f.write(print_sentence)

        print_tree(node.left, f)
    if node.attr is not None:
        print_sentence = f"{(node.depth+1)*'| '}"
        #print(print_sentence, end='')
        f.write(print_sentence)

        print_sentence = f"{node.attr[1]} == 1: "
        #print(print_sentence, end='')
        f.write(print_sentence)

        print_tree(node.right, f)

def readData(input_file: str) -> ([], {}, [], []):
    data = np.genfromtxt(input_file, delimiter="\t", dtype=None, encoding=None)
    data_x_feature_name = []
    for i in range(data.shape[1]-1):
        data_x_feature_name.append(str(data[0, i]))

    data_x = np.array([[int(x_element) for x_element in x[0:-1]] for x in data[1:, :]])
    data_y = np.array([int(y_element) for y_element in data[1:, -1]])
    return data_x_feature_name, data_x, data_y

def writeOutFile(output_file: str, output_content_arr: []) -> None :
    with open(output_file, 'w') as f:
        for data_line in output_content_arr:
            f.write(f"{data_line[0]}\n")

def writeOutMetrix(output_file: str, error_train: float, error_test: float) -> None:
    with open(output_file, 'w') as f:
        f.write(f"error(train): {error_train}\n")
        f.write(f"error(test): {error_test}\n")

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

if __name__ == '__main__':
    # Initialize arguments
    train_input = ""
    test_input  = ""
    max_depth   = 0
    train_out   = ""
    test_out    = ""
    metrics_out = ""
    print_out   = ""
    is_debug    = 0

    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the test input .tsv file')
    parser.add_argument("max_depth", type=int, 
                        help='maximum depth to which the tree should be built')
    parser.add_argument("train_out", type=str, 
                        help='path to output .txt file to which the feature extractions on the training data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .txt file to which the feature extractions on the test data should be written')
    parser.add_argument("metrics_out", type=str, 
                        help='path of the output .txt file to which metrics such as train and test error should be written')
    parser.add_argument("print_out", type=str,
                        help='path of the output .txt file to which the printed tree should be written')
    parser.add_argument("--is_debug", type=int, default=0,
                        help='show debug msg.')
    args = parser.parse_args()

    # Command Input
    train_input = args.train_input
    test_input  = args.test_input
    max_depth   = args.max_depth
    train_out   = args.train_out
    test_out    = args.test_out
    metrics_out = args.metrics_out
    print_out   = args.print_out
    is_debug    = args.is_debug

    if(is_debug):
        print(f"train_input = {train_input}")
        print(f"test_input  = {test_input}")
        print(f"max_depth   = {max_depth}")
        print(f"train_out   = {train_out}")
        print(f"test_out    = {test_out}")
        print(f"metrics_out = {metrics_out}")
        print(f"print_out   = {print_out}")

    # Readinput Training Data
    if is_debug: print(f"> Read Data...")
    feature_name_train, data_x_train, data_y_train = readData(train_input)
    feature_name_test, data_x_test, data_y_test = readData(test_input)

    # Training.
    if is_debug: print(f"> dTreeModel.train()...")
    model = dTreeModel()
    model.train(feature_name_train, data_x_train, data_y_train)

    # Print the Tree.
    if is_debug: print(f"> print_tree()...")
    with open(print_out, mode='w') as f:
        print_tree(model.root_node, f)

    # Predicting.
    if is_debug: print(f"> dTreeModel.predict()...")
    predict_train_x = model.predict(data_x_train)
    predict_test_x = model.predict(data_x_test)

    # Evaluating.
    if is_debug: print(f"> evaluateMetrixAccuracy()...")
    error_train = evaluateMetrixAccuracy(data_y_train, predict_train_x)
    error_test = evaluateMetrixAccuracy(data_y_test, predict_test_x)

    # WriteOut.
    if is_debug: print(f"> writeOutFile()...")
    writeOutFile(train_out, predict_train_x)
    writeOutFile(test_out, predict_test_x)
    writeOutMetrix(metrics_out, error_train, error_test)


    if(is_debug):
        print(f"data_x_train.shape = {data_x_train.shape}")
        print(f"data_x_test.shape = {data_x_test.shape}")
        print(f"data_y_train.shape = {data_y_train.shape}")
        print(f"data_y_test.shape = {data_y_test.shape}")



    #Here is a recommended way to print the tree to a file
    # with open(print_out, "w") as file:
    #     print_tree(dTree, file)