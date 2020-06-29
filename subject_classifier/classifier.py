
import os
dname = os.path.dirname(os.path.abspath(__file__))
os.chdir(dname)
import sys
sys.path.append(dname)

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import pickle 
import pandas as pd
import re
from operations import get_predicted_sample, get_subject_list
import sys

class SubjectClassifier:

    '''
    Predict labels for the text string
    '''
    def predict_subjects(self, text):
        stemmer = SnowballStemmer('english') # to stems
        words = stopwords.words('english')
        cleaned_text = []
        cleaned_text.append(" ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", text).split() if stemmer.stem(i) not in words]).lower())

        pickle_in = open('data/classifier.pickle', 'rb')
        classifier = pickle.load(pickle_in)

        pickle_in = open('data/vectorizer.pickle', 'rb')
        vectorizer = pickle.load(pickle_in)
        test_features_transformed = vectorizer.transform(cleaned_text)

        pred_labels = classifier.predict(test_features_transformed)
        pickle_in = open('data/subject_list.pickle', 'rb')
        subject_list = pickle.load(pickle_in)
        predicted_set = pd.DataFrame(pred_labels, columns=subject_list)

        # set indices
        predicted_set['title'] = text
        titl_subj_dict = get_predicted_sample(predicted_set, 1)
        return titl_subj_dict[text]

if __name__ == "__main__":
    title = 'An Act relating to the division of labor standards and safety; relating to the division of workers compensation; establishing the division of workers safety and compensation; relating to employment of a minor; and providing for an effective date.'
    classifier = SubjectClassifier()
    classifier.predict_subjects(title)