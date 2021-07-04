from math import log as ln


def vocabulary(data):
    """
    Creates the vocabulary from the data.
    :param data: List of lists, every list inside it contains words in that sentence.
                 len(data) is the number of examples in the data.
    :return: Set of words in the data
    """
    vocab = []
    for lst in data:
        for word in lst:
            vocab.append(word)
    return set(vocab)
    

def estimate_pi(train_labels):
    """
    Estimates the probability of every class label that occurs in train_labels.
    :param train_labels: List of class names. len(train_labels) is the number of examples in the training data.
    :return: pi. pi is a dictionary. Its keys are class names and values are their probabilities.
    """
    pi = {}
    n = len(train_labels)
    for label in train_labels:
        if label in pi:
            pi[label] += 1/n
        else:
            pi[label] = 1/n
    return pi
    
def estimate_theta(train_data, train_labels, vocab):
    """
    Estimates the probability of a specific word given class label using additive smoothing with smoothing constant 1.
    :param train_data: List of lists, every list inside it contains words in that sentence.
                       len(train_data) is the number of examples in the training data.
    :param train_labels: List of class names. len(train_labels) is the number of examples in the training data.
    :param vocab: Set of words in the training set.
    :return: theta. theta is a dictionary of dictionaries. At the first level, the keys are the class names. At the
             second level, the keys are all the words in vocab and the values are their estimated probabilities given
             the first level class name.
    """
    theta = {}
    n_vocab = len(vocab)
    classes = set(train_labels)
    for c in classes:
        v = {}
        for word in vocab:
            v[word] = 0
        theta[c] = v
    for i in range(len(train_data)):
        for word in train_data[i]:
            theta[train_labels[i]][word] += 1

    for label in theta:
        count = 0
        missing = 0
        for word in theta[label]:
            count += theta[label][word] 
        for word in theta[label]:

            theta[label][word] = (theta[label][word]+1)/(count+n_vocab)

    return theta

def test(theta, pi, vocab, test_data):
    """
    Calculates the scores of a test data given a class for each class. Skips the words that are not occurring in the
    vocabulary.
    :param theta: A dictionary of dictionaries. At the first level, the keys are the class names. At the second level,
                  the keys are all of the words in vocab and the values are their estimated probabilities.
    :param pi: A dictionary. Its keys are class names and values are their probabilities.
    :param vocab: Set of words in the training set.
    :param test_data: List of lists, every list inside it contains words in that sentence.
                      len(test_data) is the number of examples in the test data.
    :return: scores, list of lists. len(scores) is the number of examples in the test set. Every inner list contains
             tuples where the first element is the score and the second element is the class name.
    """
    scores = []
    for words in test_data:
        sub = []
        for label in pi:
            result = ln(pi[label])
            for word in words:
                result += ln(theta[label][word])
            sub.append((result,label))
        scores.append(sub)

    return scores




