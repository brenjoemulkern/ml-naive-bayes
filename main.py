import numpy as np
import pandas as pd
import scipy.sparse as scp
import sys
from log_regression import LogisticRegressionClassifier

from naive_bayes import NaiveBayesClassifier

np.random.seed(0)


def build_dataframe(csv_file):
    # make dataframe from csv
    print('Making dataframe from file:', csv_file)
    d_frame = pd.read_csv(csv_file, header=None)
    print('Dataframe complete.')
    print('Number of rows: ', d_frame.shape[0])
    print('Number of columns: ', d_frame.shape[1])
    return d_frame

def split_training_data(training_data: pd.DataFrame):
    split_training_df = training_data.iloc[:9601, :]
    split_testing_df = training_data.iloc[9601:, :61189]

    return split_training_df, split_testing_df

def get_truth_data(training_data: pd.DataFrame):
    truth = training_data.iloc[9601:, 61189:]
    truth_np = truth.to_numpy()
    write_csv(truth_np.flatten(), 'ground_truth_data', 12001, truth_np.shape[0] + 12001)

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
    ones_column = np.array([np.ones(data.shape[0], dtype=float)])
    x_col_sums = data.sum(axis = 0)
    x_col_sums[x_col_sums == 0] = 1
    normal_data = (data.T / x_col_sums[:, np.newaxis]).T
    X_array = np.concatenate((ones_column.T, normal_data), axis=1)
    return scp.csr_matrix(X_array), x_col_sums

def build_W_matrix(row_count):
    W = np.random.rand(row_count, 61189)
    W[:, -1] = 0
    return scp.csr_matrix(W)

def write_csv(class_list, filename, start_row, end_row):
    print('Writing csv')
    filename = filename + '.csv'
    class_indices = np.arange(start_row, end_row)
    class_array = np.asarray(class_list)
    full_array = np.concatenate(([class_indices], [class_array]), axis=0)
    full_array_transpose = np.transpose(full_array)
    full_dataframe = pd.DataFrame(full_array_transpose, columns = ['id','class'])
    full_dataframe.to_csv(filename, index=False)

def main():
    # use this to direct command line args
    process_data = 0
    run_nb = 0
    run_lr = 0


    if len(sys.argv) == 1:
        training_file = 'training.csv'
        testing_file = 'testing.csv'
        process_data = 0
        run_nb = 1
        run_lr = 1
    
    elif len(sys.argv) == 2 and sys.argv[1] == '-val':
        training_file = 'training.csv'
        process_data = 1
        run_nb = 1
        run_lr = 1

    elif len(sys.argv) == 2 and sys.argv[1] == '-nb':
        training_file = 'training.csv'
        testing_file = 'testing.csv'
        process_data = 0
        run_nb = 1

    elif len(sys.argv) == 2 and sys.argv[1] == '-lr':
        training_file = 'training.csv'
        testing_file = 'testing.csv'
        process_data = 0
        run_nb = 1
   
    elif len(sys.argv) == 3 and sys.argv[1] == '-val' and sys.argv[2] == '-nb':
        training_file = 'training.csv'
        testing_file = 'testing.csv'
        process_data = 1
        run_nb = 1

    elif len(sys.argv) == 3 and sys.argv[1] == '-val' and sys.argv[2] == '-lr':
        training_file = 'training.csv'
        testing_file = 'testing.csv'
        process_data = 1
        run_lr = 1

    elif len(sys.argv) == 3 and sys.argv[1] == '-val' and sys.argv[2].endswith('.csv'):
        process_data = 1
        run_lr = 1

    elif len(sys.argv) == 3 and sys.argv[1].endswith('.csv') and sys.argv[2].endswith('.csv'):
        training_file = sys.argv[1]
        testing_file = sys.argv[2]

    elif len(sys.argv) == 4 and sys.argv[1] == '-nb' and sys.argv[2].endswith('.csv') and sys.argv[3].endswith('.csv'):
        training_file = sys.argv[2]
        testing_file = sys.argv[3]
        run_nb = 1

    elif len(sys.argv) == 4 and sys.argv[1] == '-lr' and sys.argv[2].endswith('.csv') and sys.argv[3].endswith('.csv'):
        training_file = sys.argv[2]
        testing_file = sys.argv[3]
        run_lr = 1
    
    else:
        sys.exit('Please run program as specified in README with training and testing csv files as first and second command line arguments.')

    if process_data == 0:
        train_df = build_dataframe('data/' + training_file)
        test_df = build_dataframe('data/' + testing_file)

    elif process_data == 1:
        full_df = build_dataframe('data/' + training_file)
        train_df, test_df = split_training_data(full_df)
        # get_truth_data(full_df)

    # create sparse matrix for training data
    sparse_train_data = scp.csr_matrix(train_df.values)

    # parition data by class
    train_df_class_list = partition_data(train_df, 61189)


    """ run this block for naive bayes """
    if run_nb == 1:
        print('Running Naive Bayes Classification')
        # Naive Bayes Classification Code
        nbc = NaiveBayesClassifier(sparse_train_data)
        class_list = []

        # classify each row in the test data
        for row in test_df.to_numpy():
            new_class = nbc.classify(row, train_df_class_list)
            class_list.append(new_class)
            if row[0] % 100 == 0:
                print('Classifying document', row[0])

        write_csv(class_list, 'naive_bayes_classified', 12001, len(class_list) + 12001)
    
    """ run this block for log regression """
    if run_lr == 1:
        print('Running Logistic Regression Classification')
        # make Y vector
        Y = sparse_train_data.getcol(61189)

        # make delta matrix for lgr
        delta = build_delta_matrix(Y)

        # make X array
        X, x_col_sums = build_X_array(train_df)

        # make W matrix
        W = build_W_matrix(len(train_df_class_list))
        
        lrc = LogisticRegressionClassifier(
                                        m=train_df.shape[0], 
                                        k=len(train_df_class_list), 
                                        n=61188, eta=0.009, 
                                        lamb=0.01, 
                                        delta = delta,
                                        X=X,
                                        x_col_sums = x_col_sums,
                                        Y=Y,
                                        W=W
                                        )

        weights_array = lrc.create_weights(10000)
        print(weights_array.shape)

        class_list = lrc.classify(test_df)
        write_csv(class_list, 'log_regression_classified', 12001, class_list.shape[0] + 12001)

if __name__ == "__main__":
    main()