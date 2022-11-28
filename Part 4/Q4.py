#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import glob
from nltk.stem import PorterStemmer
from nltk.tokenize import WhitespaceTokenizer
import pickle
import numpy as np
import collections
import pandas as pd
import sys


# In[2]:


def remove_special_characters(text):
    regex = re.compile('[^a-zA-Z0-9\s]')
    text_returned = re.sub(regex,' ',text)
    return text_returned


# In[3]:


files = glob.glob('../Data files required/english-corpora/*.txt', recursive = True)
ps = PorterStemmer()
tk = WhitespaceTokenizer()
data = {}
i = 0
for file in files:
#     print(file,'\n')
    file1 = open(file,"r")
    text1 = file1.read()
    cleaned = remove_special_characters(text1)
    cleaned = cleaned.replace("\n"," ")
    cleaned = cleaned.replace("\t"," ")
    cleaned = re.sub('\s+',' ',cleaned)
    cleaned = cleaned.lower()
    token = tk.tokenize(cleaned)
    stemming_output = ' '.join([ps.stem(w) for w in token])
    name = file[16:-4]
    data[name] = stemming_output
#     print(data[name])
    i += 1
    if (i%300 == 0):
        print('Document no equals ',i)
#         break
    break # Kept intentionally
# print(data['C00010'])


# '''Since the above code was taking time for executing so I just ran it once and saved it as a pickle. So in order to stop it from running again I kept break in for loop intentionally and loaded the previous saved output in the variable required. '''

# In[4]:


with open('../Data files required/Processed_corpos_IR.pkl', 'rb') as f:
    data = pickle.load(f)


# #### Making an index dictionary to number every document: 

# In[5]:


index_dict = {}
i = 0
for file in data.keys():
    key = file
    index_dict[key] = i
    i += 1
    # if i == 5000:
    #     break
# print(index_dict)


# In[6]:


index_dict2 = {v: k for k, v in index_dict.items()}
# print(index_dict2)


# #### Making a dict having info about no of words in each document.
# #### Also calculating average no of words accross all documents :

# In[7]:


No_W_doc_dict = {}
sum_ = 0
for doc in data.keys():
    No_W_doc_dict[doc] = len(data[doc])
    sum_ += len(data[doc])
Average_no_words = sum_ / 8351
    # print(doc)
    # break


# ## Making posting list : 

# In[8]:


unique_words = []
unique_words_dict = {}
i = 0
for key in data.keys():
    i += 1
    for word in data[key].split():
        if unique_words_dict.get(word, 0)  == 0:
            unique_words.append(word)
            unique_words_dict[word] = 1
        else:
            pass
#     if i%300 == 0:
#         print('Document no ',i,' and unique words till now equals ',len(unique_words))
# len(unique_words)


# In[9]:


# unique_words = []
# unique_words_dict = {}
temp_dict = {el:{} for el in unique_words}
# dict2 = {}
i = 0
for key in data.keys():
    i += 1
    for word in data[key].split():
        count = data[key].count(word)
        temp_dict[word][key] = count
        break # Kept intentionally
    if i%300 == 0:
        print('Document completed till now equals ',i)
    break # Kept intentionally


# In[10]:


with open('../Data files required/Saved_dictionary_IR.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)


# ### Making dict to fetch index no for every word and vice-versa

# In[11]:


word_index = {}
i = 0
for word in unique_words:
    word_index[word] = i
    i += 1

# word_index


# In[12]:


index_word = {v: k for k, v in word_index.items()}
# len(index_word)


# ### Calculating IDF for every word and storing it :

# In[13]:


word_IDF_dict = {}
for word in unique_words:
    word_IDF_dict[word] = np.log2((8351 + 1) / (len(loaded_dict[word].keys()) + 1))


# ### Making document norm required for TF-IDF model :

# In[14]:


document_norm_dict = {}
for doc in data.keys():
    vector = np.zeros(len(unique_words), dtype = int)
    for word in unique_words:
        indx = word_index[word]
        try:
            tf = loaded_dict[word][doc]
        except:
            tf = 0
        vector[indx] = tf * word_IDF_dict[word]
        break   #  Kept intentionally
    norm = np.linalg.norm(vector)
    document_norm_dict[doc] = norm
    break       #  Kept intentionally

# document_norm_dict['S00602'][100000]


# In[15]:


with open('../Data files required/document_norm_dict.pkl', 'rb') as f:
    document_norm_dict = pickle.load(f)


# ## Question 2

# ### Taking input file :

# In[16]:


if len(sys.argv)<=1:
    print('Error. Please try writing in this format: python3 code.py test.csv')
    exit()

input_file = pd.read_csv(sys.argv[1], sep="\t", header = None)

# input_file = pd.read_csv("TXT.txt",sep="\t", header=None)

# input_file


# In[17]:


def clean(query):
    regex = re.compile('[^a-zA-Z0-9\s]')
    cleaned = re.sub(regex,' ',query)
    cleaned = cleaned.replace("\n"," ")
    cleaned = cleaned.replace("\t"," ")
    cleaned = re.sub('\s+',' ',cleaned)
    cleaned = cleaned.lower()
    token = tk.tokenize(cleaned)
    processed_query = ' '.join([ps.stem(w) for w in token])
    return processed_query


# ### (a) Boolean retrival model : 

# In[18]:


# print(query)
column = ['QueryId', 'Iteration', 'DocId', 'Relevance']
BRS_output = pd.DataFrame(columns=column, index=range(99)) 
a = 0
for r in range(40):
    query = input_file.iat[r,1]
    query = clean(query)
    connecting_words = []
    cnt = 1
    different_words = []
    for word in query.split():
        if word.lower() != "and" and word.lower() != "or" and word.lower() != "not":
            different_words.append(word.lower())
        else:
            connecting_words.append(word.lower())
    # print(different_words)
    # print(connecting_words)

    bool_dict = {}
    # arr2 = np.zeros(3, dtype=bool)
    for word in different_words:
        # print(word)
        if word in unique_words:
            i = 0
            bool_dict[word] = np.zeros(8351, dtype=bool)
            documents = list(loaded_dict[word].keys())
            indices = np.zeros(len(documents))
            for x in documents:
                try:
                    indices[i] = index_dict[x]
                    i += 1
                except:
                    x = x[:-4]
                    indices[i] = index_dict[x]
                    i += 1
            for ind in indices:
                ind = int(ind)
                bool_dict[word][ind] = 1

    # print(len(documents), len(indices))
    # print(bool_dict)
    # bool_dict['cold'][40]

    query_list = query.split()
    if len(query_list) == 1:
        value = query_list[0]
    # query_list
    
    ########## Processing NOT in query ###########

    i = 0
    for word in query_list:
        if word == 'not':
            next_word = query_list[i+1]
            query_list.remove(word)
            bool_dict[next_word] = ~ bool_dict[next_word]
        i += 1
    # print(query_list)
    # bool_dict
    
    ########## Processing AND in query ###########

    i = 0
    j = 1
    while 'and' in  query_list:
        word = query_list[i]
    #     print(word)
        if word == 'and':
            prev_word = query_list[i-1]
            # print("i equals : ",i)
            next_word = query_list[i+1]
            # print('Prev and next equals : ',prev_word,' ',next_word)
            value = 'ans' + str(j)
            j += 1
            query_list[i] = value
            # print(query_list)
            query_list.remove(prev_word)
            # print("i equals : ",i)
            # print(query_list)
            query_list.remove(next_word)
            bool_dict[value] =  bool_dict[next_word] & bool_dict[prev_word]
            i = 0
        else :
            i += 1

    # print("Query list equals : ",query_list)
    # print(bool_dict)
    # print(j)
    
    ########## Processing OR in query ###########

    i = 0
    while 'or' in  query_list:
        word = query_list[i]
    #     print(word)
        if word == 'or':
            prev_word = query_list[i-1]
            # print("i equals : ",i)
            next_word = query_list[i+1]
            # print('Prev and next equals : ',prev_word,' ',next_word)
            value = 'ans' + str(j)
            j += 1
            query_list[i] = value
            # print(query_list)
            query_list.remove(prev_word)
            # print("i equals : ",i)
            # print(query_list)
            query_list.remove(next_word)
            bool_dict[value] =  bool_dict[next_word] | bool_dict[prev_word]
            i = 0
        else :
            i += 1
        # if i >= len(query_list):
        #     i = 0

    # print("Query list equals : ",query_list)
    # print(bool_dict)
    # print(j)
    
    ########## Processing words without any operator between them in query ###########
    
    i = 0
    while len(query_list) != 1:
        word = query_list[i]
        # print(word)
        prev_word = query_list[i]
        # print("i equals : ",i)
        next_word = query_list[i+1]
        # print('Prev and next equals : ',prev_word,' ',next_word)
        value = 'ans' + str(j)
        j += 1
        query_list[0] = value
        # query_list.remove(prev_word)
        # print("i equals : ",i)
        # print(query_list)
        query_list.remove(next_word)
        # print(query_list)

        bool_dict[value] =  bool_dict[next_word] & bool_dict[prev_word]
        # i = 0

    # print("Query list equals : ",query_list)
    # print(bool_dict)
    # print(j)

    ans = bool_dict[value].copy()
    indx = np.where(ans)[0]
    # final_ans = "Appropriate documents for given query are :  "
    # for x in indx:
    #     final_ans += index_dict2[x] + ', '
    # final_ans

    count = 0

    for x in indx:
        Qid = "Q" + str(r+1)
        BRS_output.loc[a].QueryId = Qid
        BRS_output.loc[a].Iteration = 1
        BRS_output.loc[a].DocId = index_dict2[x]
        BRS_output.loc[a].Relevance = 1        
        count += 1
        a += 1
        if count == 5:
            break

BRS_output = BRS_output.dropna()


# In[19]:


BRS_output.to_csv('BRS_output.csv', index=False)


# ### (b) (20 marks) Implement a system from the Tf-Idf family with appropriate forms for the functions and tuned parameters. A query is matched using cosine similarity. : 

# In[ ]:


# query_vector = []

# column = ['QueryId', 'Iteration', 'DocId', 'Relevance']
TfIdf_output = pd.DataFrame(columns=column, index=range(100)) 
a = 0
for r in range(40):
    query = input_file.iat[r,1]
    query = clean(query)
    query_list = query.split()
    query_vector = []
    for word in query_list:

        if word in unique_words:
            idf_word = word_IDF_dict[word]
        else:
            idf_word = 0

        tf_idf = query_list.count(word) * idf_word
        query_vector.append(tf_idf)
    #     q_norm+=tf_idf**2
    # q_norm=math.sqrt(q_norm)

    query_norm = np.linalg.norm(query_vector)
    query_vector = np.array(query_vector) / query_norm
    # query_vector

    # score = np.zeros(len(data.keys()))
    score = {}
    for i in index_dict2.keys():
        doc_vector = []
        for word in query_list:
            doc = index_dict2[i]
            if word in unique_words:
                idf_word = word_IDF_dict[word]
                try:
                    tf = loaded_dict[word][doc]
                except:
                    tf = 0
                tf_idf = tf * idf_word
            else:
                tf_idf = 0

            doc_vector.append(tf_idf)
        doc_vector = np.array(doc_vector) / document_norm_dict[doc]
        key = np.dot(query_vector,doc_vector)
        score[key] = doc

    # score = sorted(score.items(),key=lambda x:x[1],reverse=True)

    score_dict = collections.OrderedDict(sorted(score.items(),reverse=True))
    # score_dict

    # score_dict = collections.OrderedDict(sorted(score.items(),reverse=True))
    # od.keys()
    count = 0
    for k,v in score_dict.items():
        Qid = "Q" + str(r+1)
        TfIdf_output.loc[a].QueryId = Qid
        TfIdf_output.loc[a].Iteration = 1
        TfIdf_output.loc[a].DocId = v
        TfIdf_output.loc[a].Relevance = 1        
        count += 1
        a += 1
        if count == 5:
            break
TfIdf_output = TfIdf_output.dropna()
# TfIdf_output


# In[ ]:


TfIdf_output.to_csv('TfIdf_output.csv', index=False)


# ### (c) (20 marks) Implement a system from the BM25 family with appropriate forms for the functions and tuned parameters.

# #### Calculating IDF (for every word ) required for BM25 :

# In[ ]:


word_IDF_dict_BM25 = {}
for word in unique_words:
    freq = len(loaded_dict[word].keys())
    num = 8531 - freq + 0.5
    den = freq + 0.5
    word_IDF_dict_BM25[word] = np.log(1 + num/den)


# ##### BM25 started : 

# In[ ]:


BM25_output = pd.DataFrame(columns=column, index=range(100)) 
a = 0
for r in range(40):
    query = input_file.iat[r,1]
    query = clean(query)
    query_list = query.split()
    score_BM25 = {}
    k = 1.2
    b = 0.75
    for doc in data.keys():
        Sum = 0
        for word in query_list:
            try:
                tf = loaded_dict[word][doc]
                idf = word_IDF_dict_BM25[word]
                num = idf * tf * (k + 1)
                den = tf + k * (1 - b + b*No_W_doc_dict[doc]/Average_no_words )
                Sum += num/den
            except:
                pass
        score_BM25[Sum] = doc

    score_dict_BM25 = collections.OrderedDict(sorted(score_BM25.items(),reverse=True))

    count = 0
    for k,v in score_dict_BM25.items():
        Qid = "Q" + str(r+1)
        BM25_output.loc[a].QueryId = Qid
        BM25_output.loc[a].Iteration = 1
        BM25_output.loc[a].DocId = v
        BM25_output.loc[a].Relevance = 1        
        count += 1
        a += 1
        if count == 5:
            break
            
BM25_output = BM25_output.dropna()

BM25_output.to_csv('BM25_output.csv', index=False)

# with pd.option_context('display.max_rows', None,'display.max_columns', None):
#     display(BM25_output)
# BM25_output


# ### Finished
