"""
This program does a basic n-gram analysis of the Dakota data.

Author: Yitong Chen
Last-Updated: 04/30/2018
Reference: https://github.com/Elucidation/Ngram-Tutorial
"""
import random
import re

def generateNgram(words, n=1):
    """Function for generating an n-gram"""
    gram = dict()
    
    # 1-5 grams would make sense given our database
    assert n >= 1 and n <= 5
    
    for i in range(len(words) - (n - 1)):
        key = tuple(words[i:i+n])
        if key in gram:
            gram[key] += 1
        else:
            gram[key] = 1
    gram = sorted(gram.items(), key=lambda x:x[1], reverse=True)
    return gram

def weighted_choice(choices, n = 1):
    total = sum(w for c, w in choices)
    r = random.randint(0, total//n)
    upto = 0
    for c, w in choices:
        if upto + w > r:
            return c
        upto += w

def getNGramSentenceRandom(gram, word, n = 50):
    word_list = []
    for i in range(n):
        word_list.append(word)
        choices = [element for element in gram if element[0][0] == word]
        if not choices:
            break
        word = weighted_choice(choices, n)[1]
    print(" ".join(word_list))
    
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
    
    start_word = words[int(len(words) / random.randint(0, 8000))]
    print("Start word: ", start_word)
    
    for i in range(2, 6):
        gram = generateNgram(words, i)
        print(i, "-gram sentence:")
        getNGramSentenceRandom(gram, start_word, 10)
    
    
    # close file when finished
    raw_data.close()

if __name__ == "__main__":
    main()