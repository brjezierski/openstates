import pandas as pd
import os
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import pickle
import json as j
import re

scriptpath = os.path.dirname(os.path.realpath(__file__))


def get_subject_list(state):
    '''
    Return subjects from the give file as a list. 
    '''
    file_name = 'subjects_sorted/{}.txt'.format(state)
    subjects_file = open(os.path.join(scriptpath, file_name))
    subjects_list = subjects_file.readlines()
    subjects_list = list(map((lambda x: x[0:-1]), subjects_list))
    subjects_file.close()
    return subjects_list


def save_results(data, filename):
    '''
    Save bill titles together with labels to a file.

    Return title-subject dictionary
    '''
    results_file = open(os.path.join(scriptpath, filename), 'w+')
    title_subjects_dict = {}
    for index, row in data.iterrows():
        results_file.write('\n\n' + row['title'])
        subjects = []
        for column_name, value in row.items():
            if value == 1:
                results_file.write('\n' + column_name)
                subjects.append(column_name)
        title_subjects_dict[row['title']] = subjects
    results_file.close()
    return title_subjects_dict


def add_subjects(dataframe):
    '''
    Add a subjects column to the dataframe to aggragate string values of all the subjects for each bill.
    '''
    count = 0
    for index, row in dataframe.iterrows():
        count += 1
        subjects = []
        for column_name, value in row.items():
            if value == '1' or value == 1:
                subjects.append(column_name)
        dataframe.at[index, 'subjects'] = str(subjects).replace('[', ' ').replace(']', ' ')


def get_predicted_sample(predicted_labels, sample_size):
    '''
    Print the tags for a sample of a given size from the classifier's prediction.

    Return the dictionary of titles and subjects
    '''
    title_subjects_dict = {}
    sample = predicted_labels.sample(n=sample_size)
    for index, row in sample.iterrows():
        print('\n', row['title'])
        subjects = []
        for column_name, value in row.items():
            if value == 1:
                print(column_name)
                subjects.append(column_name)
        title_subjects_dict[row['title']] = subjects
    return title_subjects_dict


def export_titles_subjects(dataframe, state, file_name):
    '''
    Create a new column in a dataframe to merge bill titles and subjects. Dump the pickle thereof.
    '''
    json_data = None

    with open('bill_datasets/{}.json'.format(state)) as data_file:
        lines = data_file.readlines()
        joined_lines = '[{}]'.format(','.join(lines))
        json_data = j.loads(joined_lines)

    dataframe = pd.DataFrame(json_data)
    stemmer = SnowballStemmer('english')
    words = stopwords.words('english')

    dataframe['subjects'] = ''
    add_subjects(dataframe)
    dataframe['cleaned'] = (dataframe['title'] + dataframe['subjects']).apply(lambda x: " ".join([stemmer.stem(i)
                                                                                                  for i in re.sub("[^a-zA-Z]", " ", x).split() if stemmer.stem(i) not in words]).lower())
    bill_titles_subjects = dataframe['cleaned']
    with open('data/{}.pickle'.format(file_name), 'wb') as f:
        pickle.dump(bill_titles_subjects, f)
