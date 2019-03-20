import os
from gensim import corpora,models
from tqdm import tqdm
import logging
from nltk.corpus import wordnet
import numpy as np
import math
from scipy import spatial
import sklearn.metrics
from scipy.spatial import distance
import scipy
import matplotlib.pyplot as plt
import glob
import pickle

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
    def __iter__(self):
        for line in open(self.dirname):
            yield line.split()


class MyCorpus(object):
    def __init__(self, dirname, dictionary):
        self.dirname = dirname
        self.dictionary = dictionary
    def __iter__(self):
        for line in open(self.dirname):
            yield self.dictionary.doc2bow(line.lower().split())


def lda_preparation(corpus_doc, lda_dir):

    """
    Prepare documents for lda parsing
    :param corpus_doc: document-level version of lda in text format
    :param lda_dir: output directory for dictionary and bow
    """

    dictionary = corpora.Dictionary((line.lower().split() for line in open(corpus_doc, 'r')), prune_at=None)
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    dictionary.compactify()
    dictionary.save(os.path.join(lda_dir, 'lda_dict'))
    dictionary.save_as_text(os.path.join(lda_dir, 'lda_dictext'), sort_by_word=True)
    # Corpus parsing
    bow_corpus = MyCorpus(corpus_doc, dictionary)
    corpora.MmCorpus.serialize(os.path.join(lda_dir, 'lda_bow'), bow_corpus)


def normalize_(embedding_vector):

    """
    Normalize input vector
    :param: embedding_vector: non-normalized vector
    :return: normalizes vector
    """

    mag = math.sqrt(sum([x**2 for x in embedding_vector]))
    embedding_vector = [vv/mag for vv in embedding_vector]
    return np.array(embedding_vector)


def similarity_distances(generic, topic_specific, max_threshold, method):

    """
    Implement different distance metrics
    :param: generic : Matrix vocab_size_big*vocab_size_big which contains
    the pairwise similarities for the generic vocabulary
    :param: topic_specific: Matrix vocab_size_topic*vocab_size_topic which contains
    the pairwise similarities for the topic specific vocabulary
    :param: method str: distance metric
    :return: list that contains the distances
    """

    distances = []

    assert generic.shape[0] >= topic_specific.shape[0]

    for i in range(max_threshold):
        if method == 'cosine':
            distances += [(1 - spatial.distance.cosine(normalize_(generic[i]), normalize_(topic_specific[i])))]
        elif method == 'l2':
            distances += [distance.euclidean(generic[i], topic_specific[i])]
    return distances


def load_norm(generic, topic, max_words, extra_restriction=[]):

    """
    Load number embeddings from generic and topic-specific models (also apply normalization)

    :param: generic: pre-trained model of generic word embeddings
    :param: topic: pre-trained model of topic word embeddings
    :return: matrix with number normalized embeddings
    """

    gemb = []
    temb = []
    words = []
    if extra_restriction != []:
        word_check_list = extra_restriction
    else:
        word_check_list = topic.wv.index2word[:max_words]
    for word in word_check_list:
        if word in generic.wv.vocab:
            embedding_vector_g = normalize_(generic.wv[word])
            gemb.append(embedding_vector_g)

            embedding_vector_t = normalize_(topic.wv[word])
            temb.append(embedding_vector_t)

            words.append(word)
    counter = len(gemb)
    return np.array(gemb), np.array(temb), counter, words


def topic_corpora(lda, dictionary, files, corpus):

    """
    Clustering the sentence-level of the corpus to topic-specific corpora
    using a soft scheme with threshold equal to 0.1
    :param lda: trained lda model
    :param dictionary: dictionary of lda model
    :param files: output directories
    :param corpus: sentence-level of corpus
    """

    lines = open(corpus).readlines()
    logging.info('Clustering of corpus (sentence-level) to topic-based corpora')
    with tqdm(total=len(lines)) as pbar:
        for line in lines:
            pbar.update(1)
            bow = dictionary.doc2bow(line.strip().split())
            if len(bow) != 0:
                dist = list(sorted(lda[bow], key=lambda x: x[1]))
                [files[d[0]].write(line) for d in dist if d[1] > 0.1]
    for i in range(len(files)):
        files[i].close()


def monosemy(word):

    """
    Check if a word is monosemous according to WordNet
    :param word: word to investigate
    :return: True if the word is monosemous, False otherwise
    """

    a = wordnet.synsets(word)
    noun = wordnet.synsets(word, pos=wordnet.NOUN)
    verb = wordnet.synsets(word, pos=wordnet.VERB)
    adj = wordnet.synsets(word, pos=wordnet.ADJ)
    adv = wordnet.synsets(word, pos=wordnet.ADV)
    # A word is defined as monosemous, if it has one sense per syntactic category
    if len(wordnet.synsets(word)) != 0:
        if (len(noun) < 2) and (len(verb) < 2) and (len(adj) < 2) and (len(adv) < 2):
            return True
    return False


def distribution_sim(gdsm_sims, tdsm_sims, words):

    '''
     Distribution of similarity values
    :param gdsm_sims: generic similarities
    :param tdsm_sims: topic similarities
    :param words: words under investigation
    '''

    mon = []
    pol = []

    for (g_sims, t_sims), word in zip(zip(gdsm_sims, tdsm_sims), words):
        if monosemy(word):
            mon.append(abs(t_sims-g_sims))
        else:
            pol.append(abs(t_sims-g_sims))

    plt.plot(np.mean(mon, axis=0), label = 'mon')
    plt.plot(np.mean(pol, axis=0), label = 'pol')
    plt.legend()
    plt.show()


def procrustes(x, y):

    """
    Find a transformation matrix between 2 TDSMs, using Procrustes Analysis
    :param x: source space
    :param y: target space
    :return: transformation matrix
    """

    X = np.array(x)
    Y = np.array(y)
    dot_ = Y.T.dot(X)
    # Diagonalize matrix
    u, s, vt = np.linalg.svd(dot_)
    W = vt.T.dot(u.T)
    return W

