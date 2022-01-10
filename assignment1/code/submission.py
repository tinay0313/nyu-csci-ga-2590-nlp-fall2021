import json
import collections
import argparse
import random

from util import *

random.seed(42)

def clean_tokens(tokens):
    """Return tokens that are not punctuation or stopwords in lowercase
    Parameters:
        tokens(list)
    Returns:
        A list of tokens(string)
    """
    special_char = ["!!","?!","??","!?","`","``","''","-lrb-","-rrb-","-lsb-","-rsb-",",",".",":",";","\"","'","-","!","#","###","$","%","&","(",")","*","..","...","?","@","[","]","^","{","}","+","<",">"]
    
    cleaned_tokens =[]
    for tok in tokens:
        tok = tok.lower()
        if tok not in special_char:
            cleaned_tokens.append(tok)
    return cleaned_tokens

def vectorize(tokens, vocab_list):
    vector = {}
    for w in vocab_list:
        key = w
        value = tokens.count(w)
        vector[w] = value
    return vector

def extract_unigram_features(ex):
    """Return unigrams in the hypothesis and the premise.
    Parameters:
        ex : dict
            Keys are gold_label (int, optional), sentence1 (list), and sentence2 (list)
    Returns:
        A dictionary of BoW featurs of x.
    Example:
        'I love it', 'I hate it' --> {'I':2, 'it':2, 'hate':1, 'love':1}
    """
    # BEGIN_YOUR_CODE
    bow = {}

    vocab_list = []
    s1 = ex['sentence1']
    s2 = ex['sentence2']

    for i in range(len(s1)):
        vocab_list.append(s1[i])
    
    for i in range(len(s2)):
        vocab_list.append(s2[i])

    tokens = []
    for i in range(len(s1)):
        tokens.append(s1[i])
    
    for i in range(len(s2)):
        tokens.append(s2[i])
    
    bow = vectorize(tokens, vocab_list)
    return bow
    # END_YOUR_CODE

def extract_custom_features(ex):
    """Design your own features.
    """
    # BEGIN_YOUR_CODE
    bow = {}
    
    vocab_list = []
    s1 = ex['sentence1']
    s2 = ex['sentence2']

    s1 = clean_tokens(s1)
    s2 = clean_tokens(s2)
    
    vocab_list.append(s1[0])
    vocab_list.append(s2[0])
    for i in range(1, len(s1)):
        vocab_list.append(s1[i - 1] + "_" + s1[i])
        vocab_list.append(s1[i])
    
    for i in range(1, len(s2)):
        vocab_list.append(s2[i - 1] + "_" + s2[i])
        vocab_list.append(s2[i])

    tokens = []
    tokens.append(s1[0])
    tokens.append(s2[0])
    for i in range(1, len(s1)):
        word = s1[i - 1] + "_" + s1[i]
        tokens.append(word)
        tokens.append(s1[i])
    
    for i in range(1, len(s2)):
        word = s2[i - 1] + "_" + s2[i]
        tokens.append(word)
        tokens.append(s2[i])
    
    bow = vectorize(tokens, vocab_list)
    return bow
    # END_YOUR_CODE

def learn_predictor(train_data, valid_data, feature_extractor, learning_rate, num_epochs):
    """Running SGD on training examples using the logistic loss.
    You may want to evaluate the error on training and dev example after each epoch.
    Take a look at the functions predict and evaluate_predictor in util.py,
    which will be useful for your implementation.
    Parameters:
        train_data : [{gold_label: {0,1}, sentence1: [str], sentence2: [str]}]
        valid_data : same as train_data
        feature_extractor : function
            data (dict) --> feature vector (dict)
        learning_rate : float
        num_epochs : int
    Returns:
        weights : dict
            feature name (str) : weight (float)
    """
    # BEGIN_YOUR_CODE
    weights = {}
    x = []
    y_actual = []

    for i in range(len(train_data)):
        x.append(feature_extractor(train_data[i]))
        y_actual.append(train_data[i]['gold_label'])
    
    dataset_sz = len(x)

    for epoch in range(num_epochs):
        for i in range(dataset_sz):
            y_pred = predict(weights, x[i])
            # d_weights saves the gradient for each word
            d_weights = {}
            
            for word in x[i].keys():
                d_weights[word] = (y_actual[i] - y_pred) * x[i][word]
            
            increment(weights ,d_weights, learning_rate)

    return weights
    # END_YOUR_CODE