from naive_bayes import NaiveBayesClassifier
import pandas as pd
import scipy.sparse as scp
import numpy as np
import csv


def build_sparse_matrix(matrix_list):
    with open('data/dummy_data.csv') as training_data:
        for row in training_data:
            vector = np.fromstring(row, '\t')
            matrix = scp.csr_matrix(vector)
            matrix_list.append(matrix)
            # print(matrix)
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
        partitioned_data.append(class_df)

    return partitioned_data

def build_numpy_matrix():
    matrix = np.loadtxt(open('data/dummy_data.csv', "rb"), delimiter=",")

def main():

    # make dataframe for training data
    train_df = build_dataframe('data/dummy_data.csv')
    test_df = build_dataframe('data/dummy_test.csv')
    # print(train_df)
    sparse_train_data = scp.csr_matrix(train_df.values)
    # print(sparse_train_data.getrow(0)[1])

    train_df_class_list = partition_data(train_df, 61189)

    nbc = NaiveBayesClassifier()

    partitioned_data = []
    # for i in range(0, 20):
    #     partition = scp.csr_matrix(train_df_class_list[i].values)
    #     partitioned_data.append(partition)

    nbc = NaiveBayesClassifier()
    y_prob_list = nbc.compute_mle_y(sparse_train_data)
    print(len(y_prob_list))

    



if __name__ == "__main__":
    main()