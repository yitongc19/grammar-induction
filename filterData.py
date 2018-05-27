"""
This program investigates the distribution of the k̇ symbol in
the Dakota database.

Author: Yitong Chen
Last Updated: 05/26/2018
"""
import re

class Entry:
    def __init__(self, imgId, lineNum, word):
        self.imgId = imgId
        self.lineNum = lineNum
        self.word = word
        
    def __str__(self):
        return(self.imgId + "," + str(self.lineNum) + "," + self.word)

    def __repr__(self):
        return self.__str__()

def processFile(data):
    """Scan through file for datapoints containing k̇.
    
    Parameter:
    data - the Dakota data file
    
    Return: dictionary containing the frequency of k̇
    """
    dist = {}
    lineNum = 0
    for line in data:
        split_data = line.strip().split(",")
        for item in split_data:
            if "k̇" in item:
                imgId = split_data[0]
                author = split_data[len(split_data) - 1]
                words = re.split(r'\s+|[",;?.]\s*', item)
                for word in words:
                    if "k̇" in word:
                        targetWord = word
                if author in dist:
                    dist[author]["total"] += 1
                    dist[author]["entries"].append(Entry(imgId, lineNum, targetWord))
                else:
                    author_dict = {}
                    author_dict["total"] = 1
                    author_dict["entries"] = [Entry(imgId, lineNum, targetWord)]
                    dist[author] = author_dict
                break
        lineNum += 1
    return dist


def outputData(dist, out):
    """Write the findings to a file.
    
    Paramter: 
    dist - dictionary with distribution data
    out - file that the result should be written to
                 -> Author, 
                 Line number, Image Number, utterance detail
    """
    for key in dist:
        for entry in dist[key]["entries"]:
            out.write(str(entry))
            out.write("," + key + "\n")
            

def main():
    try:
        data = open("dataToFilter.csv", "r")
        result = open("processResult.csv", "w")
    except:
        print("Error encountered when opening file.")
        quit()
    distributionData = processFile(data)
    outputData(distributionData, result)
    
    data.close()
    result.close()
    

if __name__ == "__main__":
    main()