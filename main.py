import pandas as pd

def main():
    pass

def Make_Pandas_Dataframe(csv_file):
    # make dataframe from csv
    d_frame = pd.read_csv(csv_file, header=None)
    print('Making dataframe from file: ', csv_file)
    print('Number of rows: ', d_frame.shape[0])
    print('Number of columns: ', d_frame.shape[1])
    return d_frame


if __name__ == "__main__":
    main()