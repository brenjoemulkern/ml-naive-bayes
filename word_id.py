import pandas as pd

# read words as df
top_words = pd.read_csv('important_words.csv')

# make numpy array
words = top_words['wordID'].to_numpy()

# read vocab as df, make array
all_words = pd.read_csv('data/vocabulary.txt', sep=' ')
all = all_words['word'].to_numpy()

# make list of words iteratively
important_list = []
for i in words:
    important_list.append(all[i-1])

print(important_list)