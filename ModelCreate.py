from bs4 import BeautifulSoup
import numpy as np
import os
import pymorphy2
import re
from stop_words import get_stop_words
import pickle

AllowedWords = '[а-я]+'

wordNormal = pymorphy2.MorphAnalyzer()
stop_words = {wordNormal.normal_forms(i)[0] for i in get_stop_words('ru')}


MODEL_NAME = 'finaleModel.bin'

#This is the part where all docs and discussions get cleaned and stored
#Also test and train txt was created here

fileMap = dict()

for num,name in enumerate(os.listdir("./Dirty/Data/")):
    fileMap[num] = name
    with open("./Dirty/Data/"+name, 'r') as doc:
        soup = BeautifulSoup(doc.read(), 'html.parser')
        with open("./Clean/Data/"+str(num)+'.txt', 'w') as clean:
            clean.write(' '.join(soup.getText(' ').strip().split()))
            
text = ''
for name in os.listdir("./Clean/Data/"):
    try:
        with open("./Clean/Data/"+name, 'r',) as doc:
            for word in re.findall(AllowedWords, doc.read().lower()):
                normal = wordNormal.normal_forms(word)[0]
                text += str(normal) + ' '
            text += '\n'
    except:
        print("haha")
with open('./Technical/test.txt', 'w') as file:
        file.write(text)
        
for num, name in enumerate(os.listdir("./Dirty/Discussions/")):
    with open("./Dirty/Discussions/"+name, 'r') as doc:
        soup = BeautifulSoup(doc.read(), 'html.parser')
        with open("./Clean/Discussions/"+str(num)+'.txt', 'w') as clean:
            clean.write(' '.join(soup.getText(' ').strip().split()))
            
text = ''
for name in os.listdir("./Clean/Discussions/"):
    try:
        with open("./Clean/Discussions/"+name, 'r') as doc:
            for word in re.findall(AllowedWords, doc.read().lower()):
                normal = wordNormal.normal_forms(word)[0]
                text += str(normal) + ' '
            text += '\n'
    except:
        print("haha")
with open('./Technical/train.txt', 'w') as file:
    file.write(text)


#This is part where model was created
    
command = """./sent2vec/fasttext sent2vec -input \
    ./Technical/train.txt -output ./Model/{} -minCount 2 -dim 700\
        -epoch 9 -lr 0.15 -wordNgrams 2 -loss ns -neg 10 -thread 2\
            -t 0.000005 -dropoutK 4 -minCountLabel 10""".format(MODEL_NAME)
os.system(command)

#Here we create vectors for each doc
command = """./sent2vec/fasttext print-sentence-vectors ./Model/{} < ./Technical/test.txt""".format(MODEL_NAME)
output = os.popen(command).readlines()

#Cleaned all docs
clearVec = dict()
with open('./Technical/test.txt', 'r') as test:
    text = test.readlines()
    for num, i in enumerate(output):
        clearVec[(text[num].strip())] = [float(j) for j in i.strip().split()]

#Let's store our clearVec with pickle

with open('./Technical/clearVec.pickle', 'wb') as file:
    pickle.dump(clearVec, file)
    
#And dont forgot our file mapping file

with open('./Technical/fileMapping.pickle', 'wb') as file:
    pickle.dump(fileMap, file)


