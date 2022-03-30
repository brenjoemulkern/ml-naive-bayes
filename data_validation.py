import pandas as pd
import numpy as np
import sys

class DataValidator:

    def build_dataframe(csv_file):
        # make dataframe from csv
        print('Making dataframe from file:', csv_file)
        d_frame = pd.read_csv(csv_file, header=0, names=['id', 'class'])
        print('Dataframe complete.')
        print('Number of rows: ', d_frame.shape[0])
        print('Number of columns: ', d_frame.shape[1])
        return d_frame

    def calculate_accuracy(test_data, truth_data):
        # make dataframe where row gets a 1 if classes match and a 0 if not
        acc_df = np.where(truth_data['class'] == test_data['class'], 1, 0)

        # divide sum by total number of rows for accuracy
        accuracy = acc_df.sum() / truth_data.shape[0]

        return accuracy

    def make_confusion_matrix(test_data, truth_data):
        # build confusion matrix by joining results of truth and sample data
        joined_df = test_data.join(truth_data.iloc[:, 1:], lsuffix='-s', rsuffix='-t')

        # plot results in matrix along both axes
        confusion_matrix = pd.crosstab(joined_df['class-s'], joined_df['class-t'], rownames=['Sample'], colnames=['Truth'])
        return confusion_matrix


def main():

    # take testfile and truthfile as args
    testfile = sys.argv[1]
    truthfile = sys.argv[2]

    # make dataframes
    classify_results = DataValidator.build_dataframe(testfile)
    truth_results = DataValidator.build_dataframe(truthfile)

    # calculate accuracy
    accuracy = DataValidator.calculate_accuracy(classify_results, truth_results)

    print('Accuracy:', accuracy)

    # save confusion matrix to csv
    confusion_matrix = DataValidator.make_confusion_matrix(classify_results, truth_results)
    confusion_matrix.to_csv('val_confusion_matrix.csv', index=True)
    

if __name__ == "__main__":
    main()
