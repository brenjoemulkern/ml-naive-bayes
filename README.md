# Naive Bayes and Logistic Regression

## Project 2 for CS529 Spring 2022

### Description

The program implements a Naive Bayes classification algorithm and a Logistic Regression algorithm.

The Naive Bayes calculates the MLE of Y class data and the MAP of X|Y and classifies each document one row at a time.

The Logistic Regression uses matrix multiplication to represent the X data and weight matrix W to calculate probabilities, then undergoes 10000 update steps of gradient descent.

### Running the Code

The program will search for .csv files in the data directory of the repo.
If the program is run with no command line arguments, the default training and testing files it will search for are 'training.csv' and 'testing.csv' respectively, located in the data directory.

If run with no command line arguments, the program will first run the naive bayes classification, then will run the logistic regression classification, both on the same training and testing files.

<code>python main.py</code>

If training and testing files with different names are desired to be used, the filenames can be listed as command line arguments:

<code>python main.py training_file.csv testing_file.csv</code>

If only one classifier is desired to be run, the second command line argument must be '-nb' for naive bayes and '-lr' for logistic regression:

<code>python main.py -nb</code> will run the Naive Bayes classifier with default training and testing files

<code>python main.py -lr</code> will run the Logistic Regression classifier with default training and testing files

The above commands can also be run with specified training and testing files: 

<code>python main.py -nb training_file.csv testing_file.csv</code> will run Naive Bayes on specified files

<code>python main.py -lr training_file.csv testing_file.csv</code> will run Logistic Regression on specified files

For validation, the second command line argument must be '-val'.

<code>python main.py -val</code> will run both Naive Bayes and Logistic Regression and will split the training file.

The above command can be used with a single specified training file:

<code>python main.py -val training_file.csv</code>

Also, it can be combined with '-nb' or '-lr':

<code>python main.py -val -nb training_file.csv</code> will run Naive Bayes with the specified file

<code>python main.py -val -lr training_file.csv</code> will run Logistic Regression with the specified file

<code>python main.py -val -nb</code> will run Naive Bayes with default file

<code>python main.py -val -lr</code> will run Logistic Regression with default file

### Runtime

The program will first read the training and testing files, which will take around 7-10 minutes.

The naive bayes training will print updates at every 100 documents classified (out of ~6700) to indicate its progress.

The logistic regression training will print updates at every 100 update steps (out of 10000) to indicate its progress.

### Output

The program will classify the test data using Naive Bayes classification and/or Logistic Regression classification.  

The Naive Bayes classifications are written to a .csv in the root directory of the project, titled "naive_bayes_classified.csv".
The Logistic Regression classifications are written to a .csv in the root directory of the project, titled "log_regression_classified.csv".
