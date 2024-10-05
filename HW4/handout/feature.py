import csv
import numpy as np
import argparse

VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################

# Set print options to avoid scientific notation
#np.set_printoptions(suppress=True, precision=6)
#np.set_printoptions(formatter={'float': '{:0.6f}'.format})

def wordfeatureEmbedding(input_file: str, feature_dictionary_in: str) -> [[]]:
    dataset     = load_tsv_dataset(input_file)
    feature_map = load_feature_dictionary(feature_dictionary_in)
    output_embedding = []

    for (label, review) in dataset:
        words = review.split(" ")
        valid_words = [x for x in words if x in feature_map]
        valid_features  = [np.float64(feature_map[x]) for x in valid_words]
        average_feature = np.mean(np.stack(valid_features), axis=0)
        round_feature   = np.round(average_feature, decimals=6)
        output_embedding.append((np.float64(label), round_feature))

    return output_embedding

def writeOutFile(output_file: str, out_content: [[]]) -> None:
    with open(output_file, "w") as f:
        for (label, feature) in out_content:
            feature_str = "\t".join([f"{val:.6f}" for val in feature])
            f.write(f"{label:.6f}\t{feature_str}\n")

def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    with open(file, encoding='utf-8') as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("validation_input", type=str, help='path to validation input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the input .tsv file')
    parser.add_argument("feature_dictionary_in", type=str, 
                        help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("train_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("validation_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the validation data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the test data should be written')
    args = parser.parse_args()

    train_input           = args.train_input
    validation_input      = args.validation_input
    test_input            = args.test_input
    feature_dictionary_in = args.feature_dictionary_in
    train_out             = args.train_out
    validation_out        = args.validation_out
    test_out              = args.test_out


    train_embedded = wordfeatureEmbedding(train_input, feature_dictionary_in)
    val_embedded   = wordfeatureEmbedding(validation_input, feature_dictionary_in)
    test_embedded  = wordfeatureEmbedding(test_input, feature_dictionary_in)


    writeOutFile(train_out, train_embedded)
    writeOutFile(validation_out, val_embedded)
    writeOutFile(test_out, test_embedded)
