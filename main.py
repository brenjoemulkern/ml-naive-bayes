import numpy as np
import pandas as pd
import scipy.sparse as scp
import sys
from log_regression import LogisticRegressionClassifier

from naive_bayes import NaiveBayesClassifier

np.random.seed(0)


def build_dataframe(csv_file):
    # make dataframe from csv
    d_frame = pd.read_csv(csv_file, header=None)
    print('Making dataframe from file: ', csv_file)
    print('Number of rows: ', d_frame.shape[0])
    print('Number of columns: ', d_frame.shape[1])
    return d_frame

def partition_data(data: pd.DataFrame, class_index: int):
    # returns a list of sparse matrices, each contains all documents of a specific class
    partitioned_data = []
    for i in range(1,21):
        class_df = data[data[class_index] == i]
        sparse_class_data = scp.csr_matrix(class_df.values)
        partitioned_data.append(sparse_class_data)

    return partitioned_data

def build_delta_matrix(class_array):
    delta_row_list = []
    for i in class_array:
        row = np.zeros(20, dtype=int)
        row[(i.data) - 1] = 1
        delta_row_list.append(row)

    delta_array = np.asarray(delta_row_list)
    return delta_array.transpose()

def build_X_array(data):
    data = np.asarray(data)[:, 1:61189]
    ones_column = np.array([np.ones(data.shape[0], dtype=int)])
    normal_data = data/data.sum(axis=1, keepdims=True)
    X_array = np.concatenate((ones_column.T, normal_data), axis=1)
    return X_array

def build_W_matrix(row_count):
    W = np.random.rand(row_count, 61189)
    W[:, -1] = 0
    return W

def write_csv(class_list, filename):
    print('Writing csv')
    filename = filename + '.csv'
    class_indices = np.arange(12001, 18775)
    class_array = np.asarray(class_list)
    full_array = np.concatenate(([class_indices], [class_array]), axis=0)
    full_array_transpose = np.transpose(full_array)
    full_dataframe = pd.DataFrame(full_array_transpose, columns = ['id','class'])
    full_dataframe.to_csv(filename, index=False)

def main():

    # make dataframe for training data
    if len(sys.argv) == 1:
        training_file = 'dummy_data.csv'
        testing_file = 'dummy_test.csv'
    elif len(sys.argv) == 3:
        training_file = sys.argv[1]
        testing_file = sys.argv[2]
    else:
        sys.exit('Please run program as specified in README with training and testing csv files as first and second command line arguments.')

    train_df = build_dataframe('data/' + training_file)
    test_df = build_dataframe('data/' + testing_file)

    # create sparse matrix for training data
    sparse_train_data = scp.csr_matrix(train_df.values)

    # parition data by class
    train_df_class_list = partition_data(train_df, 61189)

    """
    # Naive Bayes Classification Code
    nbc = NaiveBayesClassifier(sparse_train_data)
    class_list = []

    # classify each row in the test data
    for row in test_df.to_numpy():
        new_class = nbc.classify(row, train_df_class_list)
        print('Classifying document', row[0])
        class_list.append(new_class)

    write_csv(class_list)
    """
    # make Y vector
    Y = sparse_train_data.getcol(61189)

    # make delta matrix for lgr
    delta = build_delta_matrix(Y)

    # make X array
    X = build_X_array(train_df)

    # make W matrix
    W = build_W_matrix(len(train_df_class_list))

    # make prob_Y_WX
    X_tran = X.T
    

    
    lrc = LogisticRegressionClassifier(
                                       m=train_df.shape[0], 
                                       k=len(train_df_class_list), 
                                       n=61188, eta=0.01, 
                                       lamb=0.01, 
                                       delta = delta,
                                       X=X,
                                       Y=Y,
                                       W=W
                                       )

    weights_array = lrc.create_weights(1000)
    print(weights_array.shape)

    class_list = lrc.classify(test_df)
    write_csv(class_list, 'log_regression_classified')

if __name__ == "__main__":
    main()