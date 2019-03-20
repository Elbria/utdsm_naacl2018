#!/usr/bin/env python
from gensim.models import Word2Vec
import numpy as np
import argparse
import logging
import os
import utils
from utils import *
from tqdm import tqdm
from sklearn import preprocessing
import pickle
import random

def main():
    parser = argparse.ArgumentParser(description='Unified model of multiple topic-based embeddings')
    parser.add_argument('--corpus_doc', help='the corpus file in document format')
    parser.add_argument('--corpus_sent', help='the corpus file in sentence format')
    parser.add_argument('--size', help='the vector size for the word2vec models', type=int, default=300)
    parser.add_argument('--window', help='the window size for the word2vec models', type=int, default=5)
    parser.add_argument('--cbow', help='choose word2vec model (use 1 for skip-gram model, 0 for cbow model)',
                        choices=[0,1], type=int, default=0)
    parser.add_argument('--topics', help='the number of topics', type=int, default=50)
    parser.add_argument('--anchors', help='number of anchors', type=int, default=5000)
    parser.add_argument('--anchors-selection', help='flag for self learning implementation',
                        choices = ['unsupervised','random'], type=str, default='random')
    parser.add_argument('--output', help='output directory')
    parser.add_argument('--tdsms', help='flag for creation and training of multiple tdsms', default=True)
    parser.add_argument('--mappings', help='flag for mappings of tdsms under the same space', default=True)
    parser.add_argument('--iterations', help='number of times to perform random projections', default=10)
    parser.add_argument('--verbose', help="increase output verbosity", default=True)
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    logging.info('Runnning the %s version of the code with %s anchor points...' %(args.anchors_selection,args.anchors))
    # Create output directories
    dsms_dir = os.path.join(args.output,'dsms', str(args.size))
    print(args.output)
    if not os.path.exists(dsms_dir):
        os.makedirs(dsms_dir)

    tdsms_dir = os.path.join(dsms_dir, str(args.topics))
    if not os.path.exists(tdsms_dir):
        os.makedirs(tdsms_dir)

    lda_dir = os.path.join(args.output,'lda')
    if not os.path.exists(lda_dir):
        os.makedirs(lda_dir)

    crp_dir = os.path.join(args.output, 'topic-corpora', str(args.topics))
    if not os.path.exists(crp_dir):
        os.makedirs(crp_dir)

    matrices = os.path.join(args.output, 'matrices', str(args.anchors_selection),
                            str(args.topics) + 'T_' + str(args.size) +  'd_' +  str(args.anchors) + 'a')
    if not os.path.exists(matrices):
        os.makedirs(matrices)

    # This part of code implements a Mixture of Topic-based Distributional Semantic Models
    # TODO: ADD REFERENCE
    if args.tdsms:
        # Step 1: Build Global-DSM model
        sentences = MySentences(args.corpus_sent)  # a memory-friendly iterator
        gdsm = Word2Vec(sentences=sentences, size=args.size, window=args.window, sg=args.cbow, max_vocab_size=800000)
        gdsm.save(os.path.join(dsms_dir, 'gdsm.model'))

        # Step 2: Topic-based DSMs
        #Train LDA over corpus (document-level version).
        #Topics are defined as distributions over a vocabulary.

        lda_preparation(args.corpus_doc, lda_dir)
        ccorpus = corpora.MmCorpus(os.path.join(lda_dir, 'lda_bow'))
        dictionary = corpora.Dictionary.load(os.path.join(lda_dir, 'lda_dict'))
        lda = models.LdaMulticore(ccorpus, id2word=dictionary, num_topics=args.topics, workers=3, iterations=200)
        lda.save(os.path.join(lda_dir, str(args.topics)))

        #Split generic corpus (sentence level version) to topic-specific corpora using the trained LDA model.
        #We adopt a soft clustering scheme (threshold for clustering is set to 0.1).

        files = {}
        for topic in range(0, args.topics):
            files[topic] = open(os.path.join(crp_dir, str(topic)), 'w')
        topic_corpora(lda, dictionary, files, args.corpus_sent)

        #Creation of Topic-based Distributional Semantic Models obtained via
        #running Word2Vec over the topic-based corpora.

        tdsm = {}
        for topic in range(0, args.topics):
            sentences = utils.MySentences(os.path.join(crp_dir, str(topic)))
            tdsm[topic] = Word2Vec(sentences=sentences, size=args.size, window=args.window, sg=args.cbow, max_vocab_size=100000, workers=8)
            tdsm[topic].save(os.path.join(tdsms_dir, str(topic)))

    # This part of code implements mappings from Topic-based DSMs to the Generic-DSM
    if args.mappings:
        # Perform iterations number of random projections
        if args.anchors_selection == 'random':

            gdsm = Word2Vec.load(os.path.join(dsms_dir, 'gdsm.model'))
            for topic in range(0, args.topics):

                tdsm = Word2Vec.load(os.path.join(tdsms_dir, str(topic)))
                max_sim_subset_threshold = max(len(tdsm.wv.vocab), 20000)

                logging.info('Vocabulary size for topic %d: %d' % (topic, max_sim_subset_threshold))
                gdsm_emb, tdsm_emb, count, words = utils.load_norm(gdsm, tdsm, max_sim_subset_threshold)
                logging.info('Vocab size for intersection of tdsm-gdsm: %d' % (count))

                logging.info('Anchor searching for %d tdsm', topic)

                # Anchor points selection
                for i in range(args.iterations):
                    anchors_ = random.sample(range(count), args.anchors)  # Random anchors

                    tdsm_anchors, gdsm_anchors = [], []
                    for anc in anchors_:
                        tdsm_anchors.append(tdsm_emb[anc])
                        gdsm_anchors.append(gdsm_emb[anc])

                    logging.info('Semantic mapping for %d tdsm', topic)
                    W = utils.procrustes(tdsm_anchors, gdsm_anchors)
                    if not os.path.exists(os.path.join(matrices, str(i))):
                        os.makedirs(os.path.join(matrices, str(i)))
                    np.save(os.path.join(os.path.join(matrices, str(i)), str(topic)), W)
                    del tdsm_anchors, gdsm_anchors, W

        elif args.anchors_selection == 'unsupervised':

            gdsm = Word2Vec.load(os.path.join(dsms_dir, 'gdsm.model'))
            logging.info("Loaded generic-based dsm")

            word_freq_cross_topic = dict()
            for topic in range(0, args.topics):
                tdsm = Word2Vec.load(os.path.join(tdsms_dir, str(topic)))
                for wd in tdsm.wv.vocab.keys():
                    if wd in word_freq_cross_topic:
                        word_freq_cross_topic[wd] += 1
                    else:
                        word_freq_cross_topic[wd] = 1
            c = 0
            unified_topic_vocab = list()
            for k, v in word_freq_cross_topic.items():
                # If the word exists in all topics
                if v == args.topics:
                    c += 1
                    unified_topic_vocab += [k]

            # Distributions of similarities initialization
            total_sims = np.zeros((len(unified_topic_vocab), len(unified_topic_vocab)))

            for topic in tqdm(range(0, args.topics)):
                tdsm = Word2Vec.load(os.path.join(tdsms_dir, str(topic)))

                max_sim_subset_threshold = max(len(tdsm.wv.vocab), 20000)

                logging.info('Vocabulary size for topic %d: %d' % (topic, max_sim_subset_threshold))

                gdsm_emb, tdsm_emb, count, words = utils.load_norm(gdsm, tdsm, max_sim_subset_threshold,
                                                                   extra_restriction=unified_topic_vocab)
                logging.info("Finished loading normalized embeddings")

                logging.info('Vocab size for intersection of tdsm-gdsm: %d' % (count))

                logging.info('Anchor searching for %d tdsm', topic)

                # Similarity matrices for source and target spaces
                tdsm_sims = (tdsm_emb).dot(tdsm_emb.T)

                tdsm_sims = preprocessing.minmax_scale(tdsm_sims.T).T
                total_sims += tdsm_sims
                del tdsm_sims, tdsm

            logging.info("Finished aggregating similarity matrices of the respective topics")
            total_sims /= args.topics
            gdsm_sims = (gdsm_emb).dot(gdsm_emb.T)

            gdsm_sims = preprocessing.minmax_scale(gdsm_sims.T).T

            # Anchor points selection
            sim_sim = utils.similarity_distances(gdsm_sims, total_sims, count, method='l2')
            anchors_ = sorted(range(len(sim_sim)), key=lambda i: sim_sim[i])[:args.anchors]

            logging.info("Anchor points calculated...")
            for topic in tqdm(range(0, args.topics)):
                tdsm = Word2Vec.load(os.path.join(tdsms_dir, str(topic)))
                max_sim_subset_threshold = max(len(tdsm.wv.vocab), 20000)
                logging.info('Vocabulary size for topic %d: %d' % (topic, max_sim_subset_threshold))
                gdsm_emb, tdsm_emb, count, words = utils.load_norm(gdsm, tdsm, max_sim_subset_threshold,
                                                                   extra_restriction=unified_topic_vocab)
                tdsm_anchors, gdsm_anchors = [], []
                for anc in anchors_:
                    tdsm_anchors.append(tdsm_emb[anc])
                    gdsm_anchors.append(gdsm_emb[anc])

                logging.info('Semantic mapping for %d tdsm', topic)
                W = utils.procrustes(tdsm_anchors, gdsm_anchors)
                np.save(os.path.join(matrices, str(topic)), W)

if __name__ == '__main__':
    main ()

