from scipy.spatial.distance import cosine
import os
import pickle
from bs4 import BeautifulSoup
import numpy as np

import pymorphy2
import re
from stop_words import get_stop_words

AllowedWords = '[а-я]+'

wordNormal = pymorphy2.MorphAnalyzer()
stop_words = {wordNormal.normal_forms(i)[0] for i in get_stop_words('ru')}

MODEL_NAME = 'finaleModel.bin'

#Here we create a good search text from user's question

def MAKE_FILE_SEARCH(path):
    with open(path, 'r') as file:
        #untested
        return MAKE_SEARCH(file.read().strip())

def MAKE_SEARCH(search):
    goodSearch = ''
    search = ' '.join(search.split()).lower()
    for word in re.findall(AllowedWords, search):
        goodSearch += str(word) + ' '
        normal = wordNormal.normal_forms(word)[0]
        if normal not in stop_words:
            goodSearch += str(normal) + ' '
    with open('./Technical/input.txt', 'w') as inp:
        inp.write(goodSearch)
    
#And then let's get a good vector from our model
    
    command = """./sent2vec/fasttext print-sentence-vectors ./Model/{} < ./Technical/input.txt""".format(MODEL_NAME)
    asearch = os.popen(command).read()
    searchVec = [float(i) for i in asearch.strip().split()]

#Lets load out clearVec and file mapping pickles
    clearVec = dict()
    fileMap = dict()
    with open('./Technical/clearVec.pickle', 'rb') as file:
        clearVec = pickle.load(file)
    with open('./Technical/fileMapping.pickle', 'rb') as file:
        fileMap = pickle.load(file)
#And then let's compare this vector with all others
#We are using cos-distance

    results = dict()
    for num, i in enumerate(clearVec):
        if i == '':
            continue
        results[num] = 1 - cosine(clearVec[i], searchVec)

    sortedResults = sorted(results, key=lambda x: results[x], reverse=True)
    
#Now we get a dict with i = number of doc, sortedResults[i] - cos-distance
#Let's get top 3 document names

    answer = [fileMap[i] for i in sortedResults[:3]]
    return answer
