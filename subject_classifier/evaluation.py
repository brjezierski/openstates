import os
dname = os.path.dirname(os.path.abspath(__file__))
os.chdir(dname)

import json as j
import pandas as pd
import re
import pickle 
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec 
from scipy.sparse import csr_matrix, find
from operations import add_subjects, get_subject_list

class Score:

    def __init__(self, test_set, predicted_set, debug=False): 
        self.test_set = test_set
        self.predicted_set = predicted_set
        self.debug = debug

    def F1(self):
        '''
        Calculate F1 score between the original and predicted labels.
        '''
        pickle_in = open('data/subject_list.pickle', 'rb')
        subject_list = pickle.load(pickle_in)

        test_labels = self.test_set[subject_list].astype(str).astype(int)
        predicted_labels = self.predicted_set[subject_list].astype(str).astype(int)
        f1 = f1_score(test_labels, predicted_labels, average='micro')
        return f1

    def adjusted_F1(self, similarity_threshold):
        '''
        Calculate and return the modified F1 score where a label is considered correct if its cosine similarity with the original is over the similarity threshold.
        '''
        add_subjects(self.test_set)
        add_subjects(self.predicted_set)
        col_names =  ['predicted_subjects', 'test_subjects']
        subjects_comp_df  = pd.DataFrame(columns = col_names)
        subjects_comp_df['predicted_subjects'] = self.predicted_set['subjects']
        subjects_comp_df['test_subjects'] = self.test_set['subjects']

        none_predicted_count = 0
        correctly_labelled_count = 0
        incorrectly_labelled_count = 0
        original_labels_count = 0

        for index, row in subjects_comp_df.iterrows():
            pred_subjects = row['predicted_subjects'].split(', ')
            test_subjects = row['test_subjects'].split(', ')
            original_labels_count += len(test_subjects)

            for pred_subj in pred_subjects:
                if self.debug:
                    print('PREDICTED ', pred_subj)
                if pred_subj == '  ':
                    none_predicted_count += 1
                    continue
                correctly_labelled = False
                for test_subj in test_subjects:
                    # if test subject not assigned
                    if len(test_subj) == 2:
                        none_predicted_count += 1
                        continue
                    cos_sim = self.get_similarity(pred_subj, test_subj)
                    if cos_sim == None:
                        print('None: ', pred_subj, ', ', test_subj)
                    elif cos_sim > similarity_threshold:
                        if self.debug:
                            print('{}: {}, {}'.format(cos_sim, pred_subj, test_subj))
                        correctly_labelled_count += 1
                        correctly_labelled = True
                        continue
                if not correctly_labelled:
                    incorrectly_labelled_count += 1
        
        precision = correctly_labelled_count/(none_predicted_count+correctly_labelled_count+incorrectly_labelled_count)
        recall = correctly_labelled_count/original_labels_count
        score = 2*(precision*recall)/(precision+recall)
        return score

    def get_similarity(self, subject1, subject2):
        '''
        Return cosine similarity between two subjects.
        '''

        # stem subjects and exclude stopwords
        subjects = [subject1, subject2]
        stemmer = SnowballStemmer('english')
        stop_words = stopwords.words('english')
        subjects = list(map(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if stemmer.stem(i) not in stop_words]).lower(), subjects))

        # get tfidf_vectorizer
        pickle_in = open('data/word2vec_vectorizer.pickle', 'rb')
        tfidf_vectorizer = pickle.load(pickle_in)
        tfidf_matrix = tfidf_vectorizer.transform(subjects)

        # get Word2Vec
        pickle_in = open('data/word2vec_model.pickle', 'rb')
        word2vec_model = pickle.load(pickle_in)

        # create a list of words from the subjects
        words = []
        subject_indices = []
        index = 0
        for subject in subjects:
            for word in subject.split():
                words.append(word)
                subject_indices.append(index)
            index += 1

        # use tfidf scores where possible, sometimes the number of scores doesn't correspond to the number of words
        # i.e. when words repeat in a subject, e.g. 'SCHOOLS AND SCHOOL DISTRICTS'
        # otherwise take the unweighted average
        if len(tfidf_matrix.data) == len(words):
            tfidf_scores = tfidf_matrix.data
        else:
            tfidf_scores = np.ones(len(words))

        # for each word, get a word vector weighted by tfidf score and assign it to the subject index in a dictionary
        word_vec_values = {}
        for (word, tfidf_score, subject_index) in zip(words, tfidf_scores, subject_indices):
            try:
                word_vector = word2vec_model.wv[word]
            # catch words not available in the Word2Vec training corpus
            except KeyError as e:
                if self.debug:
                    print(e)
                return 0

            if subject_index not in word_vec_values:
                word_vec_values[subject_index] = []
            word_vec_values[subject_index].append(word_vector*tfidf_score)

        # for each subject get a weighted average word vector over the words
        average_vec_values = {}
        for subject_ind in word_vec_values:
            average_vec_values[subject_ind] = sum(word_vec_values[subject_ind])/len(word_vec_values[subject_ind])

        # get cosine similarity between the subject vectors
        cos_similarity = cosine_similarity(average_vec_values[0].reshape(1, -1), average_vec_values[1].reshape(1, -1))

        return cos_similarity

def build_word2vec(corpus_file):
    '''
    Build a Word2Vec model from the corpus in a specified file.
    '''
    # get the text corpus for Word2Vec
    pickle_in = open(corpus_file, 'rb')

    data_arr = pickle.load(pickle_in).array
    data = []
    for sentence in data_arr:
        sentence_to_list = sentence.split(' ')
        data.append(sentence_to_list)

    word2vec_model = Word2Vec(data)
    with open('data/word2vec_model.pickle', 'wb') as f:
        pickle.dump(word2vec_model, f)

def build_word2vec_vectorizer(corpus_file):
    '''
    Build a vectorizer and train it on the corpus file.
    '''
    pickle_in = open(corpus_file, 'rb')

    # ngrams_range = 1 because we want to obtain a tf_idf value for each word for vector multiplication
    vectorizer = TfidfVectorizer(ngram_range = (1,1))
    wa_bill_titles_subjects = pickle.load(pickle_in)
    vectorizer.fit(wa_bill_titles_subjects)
    with open('data/word2vec_vectorizer.pickle', 'wb') as f:
        pickle.dump(vectorizer, f)

    return vectorizer


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Build evaluation model.')
    parser.add_argument('--build_vectorizer', dest='build_vectorizer', action='store_true')
    parser.add_argument('--build_word2vec', dest='build_word2vec', action='store_true')
    parser.set_defaults(build_vectorizer=False)
    parser.set_defaults(build_word2vec=False)

    args = parser.parse_args()

    corpus_file = 'data/word2vec_corpus.pickle'

    if args.build_vectorizer:
        build_word2vec_vectorizer(corpus_file)
    if args.build_word2vec:
        build_word2vec(corpus_file)

