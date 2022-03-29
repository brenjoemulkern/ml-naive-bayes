import numpy as np
import pandas as pd
import scipy.sparse as scp

class NaiveBayesClassifier:

    def __init__(self, data):
        self.prob_y_list = (self.compute_mle_y(data))

    def compute_mle_y(self, data: scp.csr_matrix):
        # create empty list for probabilities of different classes
        # should be of length 20
        data_length = data.shape[0]
        prob_y_list = []

        # get column of classes and count occurrences of each
        y_column = data.getcol(61189)
        y_count_array = np.bincount(y_column.data)
        
        # calculate probability by dividing counts by total data
        for i in range(1,21):
            prob_y_list.append(y_count_array[i]/data_length)

        return prob_y_list

    def compute_map_xy(self, data: scp.csr_matrix, vocab_size: int, class_id: int):
        beta = 1/vocab_size
        alpha = beta + 1
        
        # to avoid looping, represent data as array and perform operations on all elements in array
        summed_xs = (np.sum(data, axis=0)).flatten()
        
        total_words = np.sum(summed_xs[0:, 1:61189])

        # numerator is total number of individual words plus alpha-1
        numerator = summed_xs[0:, 1:61189] + (alpha - 1)

        # denominator is total words plus alpha-1 times vocabulary size
        denominator = total_words + (beta * vocab_size)

        # map_xy is an array, where each element is x_i
        map_xy = numerator / denominator
        return map_xy

    def classify(self, new_document, partition_data_list):
        prob_list = []
        y_log_prob = np.log2(self.prob_y_list)
        for i in range(1,21):
            
            # for each class, compute x|y map estimate
            map_xy = self.compute_map_xy(partition_data_list[i-1], 61189, i)

            # multiply new document by log of map estimate, sum arrays, add log of mle y estimate
            xy_prob_list = np.log2(map_xy)
            full_prob = y_log_prob[i-1] + np.sum(np.multiply(new_document[1:], xy_prob_list))
            prob_list.append(full_prob)
        
        # find argmax then add 1 to class (indices are 0-19, classes are 1-20)
        new_class = prob_list.index(max(prob_list)) + 1

        return new_class
        