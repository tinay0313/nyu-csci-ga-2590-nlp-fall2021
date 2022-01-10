import numpy as np
import argparse
import time

SPECIAL_SYMBOL = '###'
SENTENCE_BREAK_SYMBOL = '###/###'

def train_and_test(entrain,endev,entest):
    #ftrain_string = './data/' + entrain
    ftrain_string = entrain
    train_file = open(ftrain_string, 'r')
    train_data = train_file.read()

    #fdev_string = './data/' + endev
    fdev_string = endev
    dev_file = open(fdev_string, 'r')
    dev_data = dev_file.read()

    input = train_data + dev_data
    input = input.split()

    word_tag_list = []
    tag_tag_list = []
    tag_list = []
    tag_set = []
    word_list = []
    prev_tag = SPECIAL_SYMBOL

    emission_probability = {}
    transition_probability = {}

    for line in input:
        word_tag_list.append(line)
        word, tag = line.split('/')
        word_list.append(word)
        tag_list.append(tag)
        tags = prev_tag + '/' + tag
        tag_tag_list.append(tags)
        prev_tag = tag

    # count occurrences of each unique tag
    # tag_dict_lambda1: key = tag, value = occurrence 
    # tag_dict_lambda2: key = tag, value = occurrence
    tag_dict_lambda1 = {}
    tag_dict_lambda2 = {}
    for t in tag_list:
        if t in tag_dict_lambda1: continue
        tag_set.append(t)
        value = tag_list.count(t)
        tag_dict_lambda1[t] = value
        tag_dict_lambda2[t] = value

    # count occurrences of each unique word
    # word_dict: key = word, value = occurrence 
    word_dict = {}
    for w in word_list:
        if w in word_dict: continue
        value = word_list.count(w)
        word_dict[w] = value
    
    # count occurrence of each unique word_tag pair
    # word_tag: key = word/tag, value = occurrence
    word_tag = {}
    for w in word_tag_list:
        if w in word_tag or w == SENTENCE_BREAK_SYMBOL: continue
        value = word_tag_list.count(w)
        word_tag[w] = value
    
    # add smoothing to all (seen and unseen) word_tag combinations
    # calculate emission probability by doing
    # emission_probability[word_tag] = count(word_tag) / count(tag)
    lambda1 = 0.03
    for w in word_dict:
        for t in tag_dict_lambda1:
            curr_word_tag = w + '/' + t
            if(curr_word_tag in word_tag):
                word_tag[curr_word_tag] += lambda1
            else:
                word_tag[curr_word_tag] = lambda1
            tag_dict_lambda1[t] += lambda1
            word_dict[w] += lambda1
            
            emission_probability[curr_word_tag] = word_tag[curr_word_tag] / tag_dict_lambda1[t]
    
    # count occurrences of each unique tag_tag pair
    # tag_tag: key = tag/tag, value = occurrence
    tag_tag = {}
    for t in tag_tag_list:
        if t in tag_tag: continue
        value = tag_tag_list.count(t)
        tag_tag[t] = value

    # add smoothing to all (seen and unseen) tag1_tag2 combinations
    # calculate transition probability by doing
    # transition_probablity[tag1_tag2] = count(tag1_tag2) / count(tag1)
    lambda2 = 1
    for t1 in tag_dict_lambda2:
        for t2 in tag_dict_lambda2:
            curr_tag_tag = t1 + '/' + t2
            if(curr_tag_tag in tag_tag):
                tag_tag[curr_tag_tag] += lambda2
            else:
                tag_tag[curr_tag_tag] = lambda2
            tag_dict_lambda2[t2] += lambda2
            transition_probability[curr_tag_tag] = tag_tag[curr_tag_tag] / tag_dict_lambda2[t1]

    start_time = time.time()
    
    # ftest_string = './data/' + entest
    ftest_string = entest
    test_file = open(ftest_string, 'r')
    test_data = test_file.read()
    test_input = test_data.split()

    # test_input_tag_sequence[i] stores the tag at position i
    # of test_input 
    # instead of storing the tag itself, save the tag's index
    # in tag_set
    groundtruth_tag_sequence = np.zeros((len(test_input)))
    for i in range(0, len(test_input)):
        tag = (test_input[i].split('/'))[1]
        groundtruth_tag_sequence[i] = tag_set.index(tag)
    
    pi = np.zeros((len(test_input) + 1, len(tag_set)))
    
    # init pi[0, t'] where t' is the start symbol ###
    for t in range(0, len(tag_set)):
        curr_word_tag = (test_input[0].split('/'))[0] + '/' + tag_set[t]
        curr_tag_tag = SPECIAL_SYMBOL + '/' + tag_set[t]
        emission = 0
        if(curr_word_tag in emission_probability):
            emission = np.log(emission_probability[curr_word_tag])
        transition = np.log(transition_probability[curr_tag_tag])
        pi[0, t] = emission + transition
    
    for i in range(1, len(test_input)):
        for t in range(0, len(tag_set)):
            curr_word_tag = (test_input[i].split('/'))[0] + '/' + tag_set[t]
            emission = 0
            if(curr_word_tag in emission_probability):
                emission =  np.log(emission_probability[curr_word_tag])
            max_score = -float("inf")
            for t_prime in range(0, len(tag_set)):
                curr_tag_tag = tag_set[t_prime] + '/' + tag_set[t]
                score = emission + np.log(transition_probability[curr_tag_tag]) + pi[i - 1, t_prime]
                max_score = max(score, max_score)
            pi[i, t] = max_score
                
    predicted_tag_sequence = np.argmax(pi, 1)
    error = 0
    for i in range(0, len(groundtruth_tag_sequence)):
        if(groundtruth_tag_sequence[i] - predicted_tag_sequence[i] != 0.):
            error += 1
    
    accuracy = (1 - (error / len(groundtruth_tag_sequence))) * 100
    print(accuracy)


    out_stream = ''
    for i in range(0, len(groundtruth_tag_sequence)):
        out_stream += test_input[i].split('/')[0] + '/' + tag_set[predicted_tag_sequence[i]] + '\n'

    out_file = open('output.txt', 'w')
    out_file.write(out_stream)
    out_file.close()
    print("Run time: %s seconds" % (time.time() - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train data name')
    parser.add_argument('--dev', help='dev data name')
    parser.add_argument('--test', help='test data name')
    args = parser.parse_args()

    train_and_test(args.train,args.dev,args.test)