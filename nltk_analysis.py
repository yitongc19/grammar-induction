"""
Analyze the Dakota data using the NLTK library

Author: Yitong Chen 
Last Updated: 05/22/2018
"""
from nltk import *
from math import *
from sklearn.model_selection import KFold
import re
import collections
import numpy as np

def create_word_list(raw_data):
    """Create a list of words from the input data.
    
    Parameters: 
    raw_data - a csv file containing Dakota data
    
    Returns:
    words - the list of words with line breakers inserted
    """
    words = []
    for line in raw_data:
        line = line.strip()
        line_split = re.split(r'\s+|[",;?]\s*', line)
        line_split = list(filter(lambda a: a != "", line_split))
        # Manually remove the repetitive lyrics from the word list
        if len(line_split) > 0 and not "pum" in line_split:
            for word in line_split:
                words.append(word)
            # Insert a line breaker
            words.append("<line>")

    return words


def create_char_list(raw_data):
    """Create a list of chars from the input data.
    
    Parameters: 
    raw_data - a csv file containing Dakota data
    
    Returns:
    chars - the list of chars with word boundaries inserted
    """
    chars = []
    for line in raw_data:
        line = line.strip()
        line_split = re.split(r'\s+|[",;?]\s*', line)
        line_split = list(filter(lambda a: a != "", line_split))
        # Manually remove the repetitive lyrics from the word list
        if len(line_split) > 0 and not "pum" in line_split:
            for word in line_split:
                for ch in word:
                    if ch != '':
                        chars.append(ch)
                # Insert a word boundary
                chars.append("<word>")
    return chars


def unigram_analysis_word(words):
    """Run unigram analysis on the list of Dakota words.
    Output the result to the user.
    
    Parameters:
    words - the list of Dakota words
    """
    fwords = FreqDist(words)
    fdwords = SimpleGoodTuringProbDist(fwords)
    common = fwords.most_common(16)
    idx = 0
    # Remove the count of the line breaker
    while idx < 16:
        if common[idx][0] == "<line>":
            common.remove(common[idx])
            break
        idx += 1
        
    return common, fdwords
    

def multigram_analysis(words, n, k):
    """Run multigram analysis on the list of Dakota words.
    Output the result to the user.
    
    Parameters:
    words - the list of Dakota words
    n - gram size
    """
    gram = ngrams(words, n)
    fgram = FreqDist(gram)
    fdgram = SimpleGoodTuringProbDist(fgram)
    # Retrieve data points with highest frequency
    common = fgram.most_common(k)
#    print(common)
    # Retrieve filtered data points
    common_filtered = []
    counter = 0
    for entry in fgram.most_common(150):
        if counter == k:
            break
        if not "<line>" in entry[0] and not "<word>" in entry[0]:
            common_filtered.append(entry)
            counter += 1

    return common_filtered, fdgram


def count_inversion(lst):
    """Helper function to count inversions.
    
    Parameters:
    lst - the list to be analyzed
    
    Returns: sorted list and number of inversions
    """
    if len(lst) == 1:
        return lst, 0
    else:
        left, left_count = count_inversion(lst[:len(lst)//2])
        right, right_count = count_inversion(lst[len(lst)//2:])
        inter, inter_count = merge_and_count(left, right)
        return inter, left_count + right_count + inter_count


def merge_and_count(lst1, lst2):
    """Helper function to merge two lists and count the 
    number of inversions across the two lists.
    
    Parameters:
    lst1 - list of integers
    lst2 - another list of integers
    
    Returns: sorted list and number of inversions
    """
    a = 0
    b = 0
    count = 0
    sorted_list = []
    while a < len(lst1) and b < len(lst2):
        if lst1[a] <= lst2[b]:
            sorted_list.append(lst1[a])
            a += 1
        else:
            sorted_list.append(lst2[b])
            count += len(lst1) - a
            b += 1
    if a < len(lst1):
        for i in range(a, len(lst1)):
            sorted_list.append(lst1[i])
    else: 
        for i in range(b, len(lst2)):
            sorted_list.append(lst2[i])
            
    return sorted_list, count
    
    
def cross_validate_uni(word_train, word_test, uni_scores, uni_prob_scores):
    """Cross validate for unigrams.
    
    Parameters:
    word_train - a list of words as training data
    word_test - a list of words as testing data
    uni_score - a list of scores based on inversions
    uni_prob_score - a list of scores based on probability
    """
    # 10-fold test on unigram
    uni_train, fdtrain = unigram_analysis_word(word_train)
    uni_test, fdtest = unigram_analysis_word(word_test)        
    uni_test_list = [entry[0] for entry in uni_test]
    inversion = []
    for entry in uni_train:
        if entry[0] in uni_test_list:            
            inversion.append(uni_test_list.index(entry[0]))
        else:
            inversion.append(10)
    sorted, score = count_inversion(inversion)
    uni_scores.append(score)
    uni_most_common = uni_train[0][0]
    uni_prob_most_common_train = fdtrain.prob(uni_most_common)
    uni_prob_most_common_test = fdtest.prob(uni_most_common)     
    uni_prob_scores.append(abs(uni_prob_most_common_train - uni_prob_most_common_test))

    
def cross_validate_multi(word_train, word_test, multi_score, multi_prob_score, n):
    """Cross validate for multigrams.
    
    Parameters:
    word_train - a list of words as training data
    word_test - a list of words as testing data
    uni_score - a list of scores based on inversions
    uni_prob_score - a list of scores based on probability
    """
    # 10-fold test on unigram
    multi_train, fdtrain = multigram_analysis(word_train, n, 10)
    multi_test, fdtest = multigram_analysis(word_test, n, 10)        
    multi_test_list = [entry[0] for entry in multi_test]
    inversion = []
    for entry in multi_train:
        if entry[0] in multi_test_list:            
            inversion.append(multi_test_list.index(entry[0]))
        else:
            inversion.append(10)
    sorted, score = count_inversion(inversion)
    multi_score.append(score)
    multi_most_common = multi_train[0][0]
    multi_prob_most_common_train = fdtrain.prob(multi_most_common)
    multi_prob_most_common_test = fdtest.prob(multi_most_common)     
    multi_prob_score.append(abs(multi_prob_most_common_train - multi_prob_most_common_test))
    
    
def cross_validate(words):
    """Run 10-fold cross-validation on the data.
    Result of the test implies the reliability of the ngram analysis.
    
    Parameters:
    words - the list of dakota words
    """
    word_array = np.array(words)
    # Define 5 split fold
    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(word_array)
    uni_scores = []
    uni_prob_scores = []
    bi_scores = []
    bi_prob_scores = []
    tri_scores = []
    tri_prob_scores = []
    # Create train and test sets and run the cross validation
    for train_index, test_index in kf.split(word_array):
        word_train, word_test = word_array[train_index], word_array[test_index]
        cross_validate_uni(word_train, word_test, uni_scores, uni_prob_scores)
        cross_validate_multi(word_train, word_test, bi_scores, bi_prob_scores, 2)
        cross_validate_multi(word_train, word_test, tri_scores, tri_prob_scores, 3)
    
    print("------------------------*------------------------")
    print("The standard deviataion of the unigram score is", np.std(uni_scores))
    print("The mean of the unigram score is", np.mean(uni_scores))
    print("The standard deviataion of the unigram probability score is", np.std(uni_prob_scores))
    print("------------------------*------------------------")
    print("The standard deviataion of the bigram score is", np.std(bi_scores))
    print("The mean of the bigram score is", np.mean(bi_scores))
    print("The standard deviataion of the bigram probability score is", np.std(bi_prob_scores))
    print("------------------------*------------------------")
    print("The standard deviataion of the trigram score is", np.std(tri_scores))
    print("The mean of the trigram score is", np.mean(tri_scores))
    print("The standard deviataion of the trigram probability score is", np.std(tri_prob_scores))
    print("------------------------*------------------------")
     
 
def display_data(word_lst):
    """Helper function for displaying the data.
    
    Parameters:
    word_lst - a list of data points
    """
    res = []
    for entry in word_lst:
        res.append("".join(entry[0]))
    print(res)
        
def main():
    # Open the raw Dakota data
    raw_data = open("dakota.csv", "r")
    raw_data_copy = open("dakota.csv", "r")
    words = create_word_list(raw_data)
    chars = create_char_list(raw_data_copy)
    for i in range(2, 5):
        common, fdgram = multigram_analysis(chars, i, 20)
        display_data(common)
        print("-----------------------")
    # Run analysis on word-level with 10-fold probability test
#    unigram_analysis_word(words)
#    for i in range(2, 4):
#        multigram_analysis(words, i)
    # Run analysis on morpheme level
    
    # Cross validation test
#    cross_validate(words)

    raw_data.close()
    raw_data_copy.close()
    
    
if __name__ == "__main__":
    main()