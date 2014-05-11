# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 10:13:16 2014

@author: maelrazavet
"""

import sgmllib
import cgi, sys
import json
import string
import sys

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np

#import gensim

import pandas

import nltk
from nltk.corpus import stopwords

class ExtractTags(sgmllib.SGMLParser):

    def __init__(self, verbose=0):
        sgmllib.SGMLParser.__init__(self, verbose)
        self.data = None
        self.dict = {};
        self.res = [];
        self.tags = ['title', 'body', 'topics', 'dateline', 'reuters'];
        
    def handle_data(self, data):
        if self.data is not None:
            self.data.append(data)

    def unknown_starttag(self, tag, attrs):
        self.data = [];
        if tag == 'reuters':
            text = ""
            for attr, value in attrs:
                text = text + " %s='%s'" % (attr, cgi.escape(value))
            self.dict[tag] = text;
            
        
    def unknown_endtag(self, tag):
        if tag in self.tags:
            if tag != 'reuters':
                self.dict[tag] = self.data;
            else:
                self.res.append(self.dict);
                self.dict = {};
            self.data = [];
        

def parseData(s):
    file = open(s)
    p = ExtractTags()
    p.feed(file.read())
    p.close()
    return p.res

def collection(data):
    #loop through the 21 SGML files and store the data into data
    for i in range(0,22):
        if i < 10:
            data = data + parseData("./reuters21578/reut2-00" + str(i) + ".sgm");
        else:
            data = data + parseData("./reuters21578/reut2-0" + str(i) + ".sgm"); 
    return data

def preprocessData(data):
    res = []
    i=0
    wnl = nltk.WordNetLemmatizer() #WordNet database for lemmatizer
    tfidf = TfidfVectorizer()
    for item in data:
        if 'body' in item:
            item['body'] = item['body'][0].lower()
            #punctuation removal
            item['body'] = item['body'].translate(None, string.punctuation)
            #Text tokenization
            item['body'] = nltk.word_tokenize(item['body'])
            #stopword removal
            item['body'] = [w for w in item['body'] if not w in stopwords.words('english')]
            #lemmatization
            item['body'] = [wnl.lemmatize(t) for t in item['body']]
            #pos-tagger
            #item['body'] = nltk.pos_tag(item['body'])
            #entities = nltk.ne_chunk(item['body'])
            #print(entities)            
            res.append(item)
    #tfs = tfidf.fit_transform(item['body'])    
    return res
    
def featureSelection(data):
    all_words = []
    for item in data:
        for w in item['body'][1:-1].split(','):
            all_words.append(w)
    freq_dist = nltk.FreqDist(w.lower() for w in all_words)
    return freq_dist.keys()[:1000]       
    
def splitTrainTestData(data):
    #initialisation of two lists
    training = []
    testing = []    
    #loop through the entire list of documents
    for item in data:
        if item['reuters'].split(' ')[2].split('=')[1] == "'TRAIN'":
            training.append(data.index(item))
        else:
            testing.append(data.index(item))
    print("Cardinality of the training set: " + str(len(training)) + " = " + str(len(training)/float(len(data)) * 100) + "%" )
    print("Cardinality of the testing set: " + str(len(testing)) + " = " + str(len(testing)/float(len(data)) * 100) + "%" )
    return training, testing

def extract10Topics(data):
    #list of the 10 most popular topics
    topics = ['earn', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat', 'corn']
    res = []
    for item in data:
        if len(item['topics']) > 0:
            if item['topics'][2:-2] in topics:
                res.append(item)
    print("The data is composed of " + str(len(res)) + " documents. \n")
    return res
    
    

def storeInCSV(data):
    df = pandas.DataFrame(data)
    df.to_csv("final.csv")
    
#return a list of dictionnaries
def getFromCSV(file):
    df = pandas.read_csv(file)
    #delete the first columun which replicates the index column
    df = df.drop(df.columns[0], axis=1)  
    data = [];
    for i, row in enumerate(df.values):
        dict = {};
        dict['body'] = row[0]
        dict['dateline'] = row[1]
        dict['reuters'] = row[2]
        dict['title'] = row[3]
        dict['topics'] = row[4]
        data.append(dict) 
    return data
    
def runMenu(data):                 
    print("********************* Start of the program *********************")
    print("* 1 - Collect the data from the 21 SGML files                  *\n")
    print("* 2 - Pre-processing the data                                  *\n")
    print("* 3 - Split the data into a training and testing set           *\n")
    print("* 4 - Select the 10 most popular topics                        *\n")
    print("* 5 - Launch option 2 to 4 in one shot with data from option 6 *\n")
    print("* 6 - Load data from CSV file                                  *\n")
    print("****************************************************************")
    option = input("Please, select the option you desire:\n")
    print("You selected the option: " + str(option))
    
    if option == 1:
        print("Collection in process...")
        data = collection(data)
        print("Collection of data successffully completed!\n")
        print("The data is composed of " + str(len(data)) + " documents. \n")    
        runMenu(data)
        
    elif option == 2:
        if len(data) == 0:
            print("You first must collect the data \n")
            runMenu(data)     
        else:
            print("Pre-processing in process...")
            data = preprocessData(data)
            print("Pre-processing completed\n")
            
    elif option == 3:
        if len(data) == 0:
            print("You first must collect the data \n")
            runMenu(data)     
        else:        
            print("Splitting in process...")
            training, testing = splitTrainTestData(data)
            print("Splitting completed\n")
        
    elif option == 4:
        if len(data) == 0:
            print("You first must collect the data \n")
            runMenu(data)     
        else:          
            print("Extraction of the 10 most popular topics in process...")
            data = extract10Topics(data)
            print("Extraction completed \n")
        
    elif option == 5:
        print("Collection in process...")
        data = getFromCSV("test.csv")    
        print("Collection of data successffully completed!\n")
        print("The data is composed of " + str(len(data)) + " documents. \n")  
        print("Extraction of the 10 most popular topics in process...")
        data = extract10Topics(data)
        print("Extraction completed \n")
        for w in data[0]['body'][1:-1].split(','):
            print w
            break;        
        test = featureSelection(data)
        print("Splitting in process...")
        training, testing = splitTrainTestData(data)
        print("Splitting completed\n")    
    
    elif option == 6:
        print("Collection in process...")
        data = getFromCSV("test.csv")    
        print("Collection of data successffully completed!\n")        
    else:
        print("end")
        sys.exit()

#We launch the menu when we launch the program
if __name__ == '__main__':
    #initialisation of a list
    data = [];     
    #runMenu(data)
    print("Collection in process...")
    data = collection(data)
    print("Collection of data successffully completed!\n")    
    print("Pre-processing in process...")
    data = preprocessData(data)
    print("Pre-processing completed\n")    
    storeInCSV(data)
