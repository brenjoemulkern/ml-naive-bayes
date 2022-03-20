import numpy as np
import pandas as pd
import scipy.sparse as scp
import sys

from naive_bayes import NaiveBayesClassifier


def build_sparse_matrix(matrix_list):
    with open('data/dummy_data.csv') as training_data:
        for row in training_data:
            vector = np.fromstring(row, '\t')
            matrix = scp.csr_matrix(vector)
            matrix_list.append(matrix)
            print(len(matrix_list))
            
    full_matrix = scp.vstack(matrix_list)
    print(full_matrix.shape[0])
    return full_matrix

def build_dataframe(csv_file):
    # make dataframe from csv
    d_frame = pd.read_csv(csv_file, header=None)
    print('Making dataframe from file: ', csv_file)
    print('Number of rows: ', d_frame.shape[0])
    print('Number of columns: ', d_frame.shape[1])
    return d_frame

def partition_data(data: pd.DataFrame, class_index: int):
    partitioned_data = []
    for i in range(1,21):
        class_df = data[data[class_index] == i]
        sparse_class_data = scp.csr_matrix(class_df.values)
        partitioned_data.append(sparse_class_data)

    return partitioned_data

def write_csv(class_list):
    print('Writing csv')
    class_indices = np.arange(12001, 18775)
    class_array = np.asarray(class_list)
    full_array = np.concatenate(([class_indices], [class_array]), axis=0)
    full_array_transpose = np.transpose(full_array)
    full_dataframe = pd.DataFrame(full_array_transpose, columns = ['id','class'])
    full_dataframe.to_csv('naive_bayes_classified', index=False)

def main():

    # make dataframe for training data
    if len(sys.argv) == 1:
        training_file = 'training.csv'
        testing_file = 'testing.csv'
    elif len(sys.argv) == 3:
        training_file = sys.argv[1]
        testing_file = sys.argv[2]
    else:
        sys.exit('Please run program as specified in README with training and testing csv files as first and second command line arguments.')

    train_df = build_dataframe('data/' + training_file)
    test_df = build_dataframe('data/' + testing_file)
    # print(train_df)
    sparse_train_data = scp.csr_matrix(train_df.values)
    # print(sparse_train_data.getrow(0)[1])

    train_df_class_list = partition_data(train_df, 61189)

    nbc = NaiveBayesClassifier(sparse_train_data)
    class_list = []
    for row in test_df.to_numpy():
        new_class = nbc.classify(row, train_df_class_list)
        print('Classifying document', row[0])
        class_list.append(new_class)

    write_csv(class_list)

if __name__ == "__main__":
    main()