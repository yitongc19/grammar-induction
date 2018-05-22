"""
Analyze the Dakota data using the NLTK library

Author: Yitong Chen 
Last Updated: 05/01/2018
"""
from nltk import *
import matplotlib.pyplot as plt
import re
import collections

def create_word_list(raw_data):
    """Create a list of words from the input data.
    
    Parameters: 
    raw_data - a csv file containing Dakota data
    
    Returns:
    words - the list of words with line breakers inserted
    """
    multiword_entry = 0
    words = []
    for line in raw_data:
        line = line.strip()
        line_split = re.split(r'\s+|[",;?]\s*', line)
        line_split = list(filter(lambda a: a != "", line_split))
        # Manually remove the repetitive lyrics from the word list
        if len(line_split) > 0 and not "pum" in line_split:
            for word in line_split:
                words.append(word)
            # Manually insert a line breaker
            words.append("<line>")
            # Count multiword entries
            if (len(line_split) > 1):
                multiword_entry += 1
    return words


def unigram_analysis_word(words):
    """Run unigram analysis on the list of Dakota words.
    Output the result to the user.
    
    Parameters:
    words - the list of Dakota words
    """
    fwords = FreqDist(words)
    fdwords = SimpleGoodTuringProbDist(fwords)
    common = fwords.most_common(21)
    idx = 0
    # Remove the count of the line breaker
    while idx < 21:
        if common[idx][0] == "<line>":
            common.remove(common[idx])
            break
        idx += 1
        
    print(common)
    #    fwords.plot(20, cumulative=False)
    print("----------------")
    print("Probability of '" + common[0][0] + "': ", fdwords.prob(common[0]))
    print("\n")
    

def multigram_analysis_word(words, n):
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
    common = fgram.most_common(10)
    # Retrieve filtered data points
    common_filtered = []
    counter = 0
    for entry in fgram.most_common(150):
        if counter == 10:
            break
        if not "<line>" in entry[0]:
            common_filtered.append(entry)
            counter += 1
    print(len(common_filtered))
            
    print(common)
    print(common_filtered)
    print("----------------")
    most_common = " ".join(common_filtered[0][0])
    print("Probability of '" + most_common + "': ", fdgram.prob(most_common))
    print("\n")


def count_inversion(lst):
    """Helper function to count inversions.
    
    Parameters:
    lst - the list to be analyzed
    
    Returns: number of inversions
    """
    pass
    
    
def main():
    # Open the raw Dakota data
    raw_data = open("test_data.csv", "r")
    words = create_word_list(raw_data)
    # Run analysis on word-level with 10-fold probability test
    unigram_analysis_word(words)
    for i in range(2, 4):
        multigram_analysis_word(words, i)
    # Run analysis on morpheme level
    
    raw_data.close()
    
    
if __name__ == "__main__":
    main()