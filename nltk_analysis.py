"""
Analyze the Dakota data using the NLTK library

Author: Yitong Chen 
Last Updated: 05/01/2018
"""
from nltk import *
import re


def main():
    raw_data = open("dakota_2.csv", "r")
    multiword_entry = 0
    words = []
    for line in raw_data:
        line_split = re.split(r'\s+|[",;?]\s*', line)
        for word in line_split:
            words.append(word)
        # Count multiword entries
        if (len(line_split) > 1):
            multiword_entry += 1
    
    words = list(filter(None, words))
    
    fwords = FreqDist(words)
#    print(fwords.most_common(20))
    
    fwords.plot(20, cumulative=False)
    
    
    # 2-5 grams
    for i in range(2, 6):
        gram = ngrams(words, 2)
        fgram = FreqDist(gram)
#        print(fgram.most_common(20))
        fgram.plot(20, cumulative=False)
    
    raw_data.close()
    
if __name__ == "__main__":
    main()