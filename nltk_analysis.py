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

def is_vowel(ch):
    """Check whether the ch is a vowel.
    The list of vowels is proposed based on Shaw (1980) and 
    the orthography of the transcription data. 
    
    Parameters:
    ch - character to be analyzed
    
    Returns: whether the ch represents a vowel
    """
    return (ch == 'a' or ch == 'ȧ' or ch == 'ạ' or ch == 'á' or
            ch == 'u' or ch == 'u̇' or ch == 'ụ' or ch == 'ú' or
            ch == 'i' or ch == 'i̇' or ch == 'ị' or ch == 'í' or
            ch == 'o' or ch == 'e')


def is_final_cons(ch):
    """Check whether the ch is a syllable final consonant.
    The list of final consonants is proposed based on Mirzayan (2010)
    and the orthography of the transcription data.
    
    Parameters:
    ch - character to be analyzed
    
    Returns: whether the ch represents a final consonant.
    """
    return (ch == 'n' or ch == 'ṇ' or ch == 'ṅ' or ch == 'm' or
            ch == 'ṃ' or ch == 'ṁ' or ch == 'b' or ch == 'ḃ' or
            ch == 'ḅ' or ch == 'p' or ch == 'ṗ' or ch == 'p̣' or
            ch == 'g' or ch == 'ġ' or ch == 'g̣' or ch == 'k' or
            ch == 'k̇' or ch == 'ḳ' or ch == 'l' or ch == 'l̇' or
            ch == 'ḷ' or ch == 'x' or ch == 'ƞ' or ch == 'h' or
            ch == 'ḣ')


def is_cons_cluster(ch1, ch2):
    """Check whether the ch1 and ch2 forms a legitimate consonant cluster.
    The list of possible consonants is proposed based on Rood (2016)
    and the orthography of the transcription data.
    
    Parameters:
    ch1 - first consonant, in particular, one that is a potential
          final consonant
    ch2 - second consonant
    
    Returns: whether the ch1, ch2 forms a legimate consonant cluster.
    """
    return (ch1 + ch2 == "mn" or ch1 + ch2 == "bd" or ch1 + ch2 == "ks"
            or ch1 + ch2 == "pt")
 
    
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
        line_split = re.split(r'\s+|[",;?.]\s*', line)
        line_split = list(filter(lambda a: a != "", line_split))
        # Manually remove the repetitive lyrics from the word list
        if len(line_split) > 0 and not "pum" in line_split:
            for word in line_split:
                words.append(word)
            # Insert a line breaker
            words.append("<l>")

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
        line_split = re.split(r'\s+|[",;?.]\s*', line)
        line_split = list(filter(lambda a: a != "", line_split))
        # Manually remove the repetitive lyrics from the word list
        if len(line_split) > 0 and not "pum" in line_split:
            for word in line_split:
                for ch in word:
                    if ch != '':
                        chars.append(ch)
                # Insert a word boundary
                chars.append("<w>")
    return chars


def create_syllable_list(raw_data):
    """Create a list of syllables from the input data.
    When determining the syllables, the algorithm makes several assumptions:
    - No vowel cluster is possible
    - Any consonant clusters at the boundaries are grouped together.
      For instance, "wamni" will be processed as "wa<s>mni" rather than 
      "wam<s>ni" even though either is technically possible.
    - From left to right, the first syllable that starts with a vowel will
      be treated as a single-vowel syllable. For instance, "oti" will be 
      processed as "o<s>ti" rather than "ot<s>i" even though both conform
      to the syllable structure of Dakota unless the vowel is followed by 
      "h" or "ƞ".
    - The syllable structure reported in Mirzayan (2010) for Lakota is adopted.
    
    Parameters: 
    raw_data - a csv file containing Dakota data
    
    Returns:
    chars - the list of syllables with syllable boundaries inserted
    """
    syllables = []
    for line in raw_data:
        line = line.strip()
        line_split = re.split(r'\s+|[",;?.]\s*', line)
        line_split = list(filter(lambda a: a != "", line_split))
        # Manually remove the repetitive lyrics from the word list
        if len(line_split) > 0 and not "pum" in line_split:
            for word in line_split:
#                syllables.append("<s>")
                idx = 0
                while idx < len(word):
                    cur_syllable = ""
                    cur_char = word[idx]
                    if is_vowel(cur_char):
                        cur_syllable += cur_char
                        if (idx + 1) < len(word):
                            lookahead = word[idx + 1]
                            if lookahead == 'ƞ' or lookahead == 'h':
                                cur_syllable += lookahead
                                idx += 1
                    else:
                        while idx < len(word) and not is_vowel(word[idx]):
                            cur_char = word[idx]
                            cur_syllable += cur_char
                            idx += 1
                        if idx < len(word):
                            cur_syllable += word[idx]
                            if (idx + 1) < len(word):
                                lookahead = word[idx + 1]
                                if lookahead == 'ƞ':
                                    cur_syllable += lookahead
                                    idx += 1
                                else:
                                    if is_final_cons(lookahead):
                                        if idx + 2 < len(word):
                                            lookfurther = word[idx + 2]
                                            if (not is_cons_cluster(lookahead, lookfurther)
                                                and not is_vowel(lookfurther)):
                                                cur_syllable += lookahead
                                                idx += 1
                                        else:
                                            cur_syllable += lookahead
                                            idx += 1
                    syllables.append(cur_syllable)
                    syllables.append("<s>")
                    idx += 1
    return syllables


def unigram_analysis(words, c):
    """Run unigram analysis on the list of Dakota words.
    Output the result to the user.
    
    Parameters:
    words - the list of Dakota words
    """
    fwords = FreqDist(words)
    fdwords = SimpleGoodTuringProbDist(fwords)
    common = fwords.most_common(c)
    idx = 0
    # Remove the count of the line breaker
    while idx < c:
        if common[idx][0] == "<l>" or common[idx][0] == "<s>":
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
        if not "<l>" in entry[0] and not "<w>" in entry[0]:
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
    uni_train, fdtrain = unigram_analysis(word_train, 16)
    uni_test, fdtest = unigram_analysis(word_test, 16)        
    uni_test_list = [entry[0] for entry in uni_test]
    inversion = []
    for entry in uni_train:
        if entry[0] in uni_test_list:            
            inversion.append(uni_test_list.index(entry[0]))
        else:
            inversion.append(15)
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
    raw_data_syllable = open("test_data.csv", "r")
    words = create_word_list(raw_data)
    chars = create_char_list(raw_data_copy)
    for i in range(2, 5):
        common, fdgram = multigram_analysis(chars, i, 20)
        display_data(common)
        print("-----------------------")
    syllables = create_syllable_list(raw_data_syllable)
    common, fgram = unigram_analysis(syllables, 31)
    print(common)
    
    # Run analysis on word-level with 10-fold probability test
#    unigram_analysis(words)
#    for i in range(2, 4):
#        multigram_analysis(words, i)
    # Run analysis on morpheme level
    
    # Cross validation test
#    cross_validate(words)

    raw_data.close()
    raw_data_copy.close()
    
    
if __name__ == "__main__":
    main()