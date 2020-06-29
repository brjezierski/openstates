import os
dname = os.path.dirname(os.path.abspath(__file__))
os.chdir(dname)

import json as j
import pandas as pd
import re
import numpy as np
import pickle
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from scipy.sparse import csr_matrix, find
from operations import get_subject_list, add_subjects, get_predicted_sample, export_titles_subjects
from evaluation import Score

class MLModel:
    data = pd.DataFrame()
    
    def preprocess(self, file):
        """
        Read the data, convert to json, apply stemmer and stopwords to extract essential information.

        Update dataframe
        """
        json_data = None

        with open(file) as data_file:
            lines = data_file.readlines()
            joined_lines = "[" + ",".join(lines) + "]"

            json_data = j.loads(joined_lines)

        self.data = pd.DataFrame(json_data)
        stemmer = SnowballStemmer('english')
        words = stopwords.words('english')
        self.data = self.data.fillna(0)

        # # strip all the numbers and other non-letter symbols and replace them by the blank
        # # then split into words
        # # stem into lemmas if not a stopword
        self.data['cleaned'] = self.data['title'].apply(lambda x: " ".join(
            [stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if stemmer.stem(i) not in words]).lower())

    def split(self):
        '''
        Split into test and train batches.
        '''

        train_set, test_set = train_test_split(self.data, test_size=0.2)
        return train_set, test_set

    def train(self, train_set, state):
        '''
        Train the training set on a OneVsRest linear support vector classifier.
        '''

        train_set_features = train_set['cleaned']

        with open('data/subject_list.pickle'.format(state), 'wb') as f:
            pickle.dump(get_subject_list(state), f)
        train_labels = train_set[get_subject_list(state)].astype(str).astype(int)

        vectorizer = TfidfVectorizer(
            min_df=0.001,  # minDF = 0.01 means "ignore terms that appear in less than 1% of the documents"
            ngram_range=(1, 3),
            stop_words="english",
            sublinear_tf=True)
        train_features_transformed = vectorizer.fit_transform(train_set_features)
        with open('data/vectorizer.pickle', 'wb') as f:
            pickle.dump(vectorizer, f)

        classifier = OneVsRestClassifier(LinearSVC(
            C=1.0,
            penalty='l2',
            max_iter=3000,
            dual=False))
        classifier.fit(train_features_transformed, train_labels)
        with open('data/classifier.pickle', 'wb') as f:
            pickle.dump(classifier, f)

    def predict(self, test_set, state, sample_size):
        '''
        Predict labels for the test set and create a dataframe to store the new labels
        '''
        test_features = test_set['cleaned']

        pickle_in = open('data/classifier.pickle', 'rb')
        classifier = pickle.load(pickle_in)

        pickle_in = open('data/vectorizer.pickle', 'rb')
        vectorizer = pickle.load(pickle_in)
        test_features_transformed = vectorizer.transform(test_features)

        pred_labels = classifier.predict(test_features_transformed)
        predicted_set = pd.DataFrame(pred_labels, columns=get_subject_list(state))
        # set indices
        predicted_set.index = list(test_set.index)
        predicted_set['title'] = test_set['title']
        sample = get_predicted_sample(predicted_set, sample_size)

        return predicted_set


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Multi-label classification.')
    parser.add_argument('--train_state', type=str, default='wa', help='State whose dataset is to be used for training.')
    parser.add_argument('--predict_state', type=str, default=None, help='State to be used for subject labeling.')
    parser.add_argument('--print_bills', type=int, default=0, help='Number of bills with predicted subjects to print.')
    parser.add_argument('--f1', dest='f1', action='store_true')
    parser.add_argument('--adjusted_f1', dest='adjusted_f1', action='store_true')
    parser.add_argument('--similarity_threshold', type=float, default=0.98, help='The similarity threshold for accepting a predicted subject.')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--export_titles', dest='export_titles', action='store_true')
    parser.set_defaults(f1=False)
    parser.set_defaults(adjusted_f1=False)
    parser.set_defaults(debug=False)
    parser.set_defaults(export_titles=False)

    args = parser.parse_args()

    model = MLModel()
    model.preprocess('bill_datasets/{}.json'.format(args.train_state))

    if args.export_titles:
        export_titles_subjects(model.data, 'wa', 'word2vec_corpus')

    # only train 
    if args.predict_state == None:
        model.train(model.data, args.train_state)
        exit()
    # split the data into training and testing sets
    elif args.train_state == args.predict_state:
        train_set, test_set = model.split()
        model.train(train_set, args.train_state)
        predicted_set = model.predict(test_set, args.train_state, args.print_bills)
    # otherwise train on the whole dataset
    else:
        model.train(model.data, args.train_state)
        model.preprocess('bill_datasets/{}.json'.format(args.predict_state))
        test_set = model.data
        predicted_set = model.predict(model.data, args.train_state, args.print_bills)

    score = Score(test_set, predicted_set, debug=args.debug)
    if args.f1:
        result = score.F1()
        print('F1: {}'.format(result))
    if args.adjusted_f1:
        result = score.adjusted_F1(args.similarity_threshold)
        print('Adjusted F1: {}'.format(result))