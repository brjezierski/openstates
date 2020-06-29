# Subject Extraction from Bill Titles

This program is used to extract subjects from a bill title from OpenStates data.

## Quick guide

To use the program for subject extraction import the following into a bill.py file
```
from subject_classifier.classifier import SubjectClassifier
```
To extract subjects from a bill title use the following class and method. predict_subjects method returns a list of strings.
```
SubjectClassifier().predict_subjects("bill_title")
```
## Content
### Directories
* data - collection of .pickle files created when training the classification and Word2Vec models as well as additional supporting files
* bill_datasets - collection of JSON files corresponding to different states with each one containing a bill title, bill number, and subjects represented by a key value pair where the key is the capitalized subject and the binary value shows whether the bill is marked by the given subject 
* subjects_sorted - collection of txt files corresponding to the states and containing a list of alphabetized subjects used when marking state legislation
* results:
    * ak_subjects_train_wa.txt - contains a list of 100 AK bills marked with predicted subjects using the model trained on WA dataset
    * ak_subjects_train_wa_ky.txt - contains a list of the same 100 AK bills marked with predicted subjects using the model trained on the dataset merged from WA and KY datasets
    * similarities.txt - contains cosine similarity values of some subject pairs

### Python files
* classifier - contains the method to predict subjects
* operations - contains supporting methods modifying pandas data frames
* evaluation - contains scoring methods and the method used to evaluate similarity between two subjects
* ML_model - contains training and predicting methods for the machine learning model

## Commands
### ML_model.py commands

**`--train_state "state"`**
Train the machine learning model on

**`--predict_state "state"`**
Predict the subjects using the ML model on the provided state

**`--print_bills "number"`**
Print the given number of bills with predictions

**`--f1`**
Print the F1 score

**`--adjusted_f1`**
Print the adjusted F1 score

**`--similarity_threshold "value"`**
Set the similarity threshold for the adjusted F1 score (between 0 and 1)

**`--debug`**
Print details of the scoring program execution

**`--export_titles`**
Export titles used by Word2Vec vectorizer, useful only when the data folder is empty

### Examples

1. Train the model on Washington dataset
```
python3 ML_model.py --train_state wa
```
2. Train the model on Washington+Kentucky dataset and predict subjects for Alaska
```
python3 ML_model.py --train_state wa_ky --predict_state ak
```
3. Train the model on Washington+Kentucky dataset and predict subjects for Alaska
```
python3 ML_model.py --train_state wa_ky --predict_state ak
```
4. Predict subjects for Alaska and print 15 bills with predictions
```
python3 ML_model.py --predict_state ak --print_bills 15
```
4. Print 15 Alaska bills with predictions
```
python3 ML_model.py --predict_state ak --print_bills 15
```
5. Calculate the adjusted F1 score for Alaska bills with predictions
```
python3 ML_model.py --predict_state ak --adjusted_f1
```
6. Split Washington dataset into training and testing batches, train on the former, and predict and calculate the F1 score for the latter.
```
python3 ML_model.py --train_state wa --predict_state wa --f1
```

7. When the data folder is empty, in order to use the entire program functionality, run the following commands.
```
python3 ML_model.py --train_state 'state' --export_titles
python3 evaluation.py --build_vectorizer --build_word2vec
```

### evaluation.py commands

**`--build_vectorizer`**
Build Word2Vec vectorizer, useful only when the data folder is empty.

**`--build_word2vec`**
Build Word2Vec model, useful only when the data folder is empty.

### Installing dependencies

Add additional Python dependencies using poetry.
```
pandas, sklearn, nltk, gensim
```
## Training

### Dataset

It was necessary to find appropriate datasets of bill titles with subjects. The criteria were the following: 
* all bills in a given state are labeled
* the labels are not too specific to a given state and can be generalized to all states
* there is a limited set of subjects that can be assigned to a state bill

Out of 50 state legislation websites, only a fraction contained subject labeling and only 8 had labels present throughout all the bills (RI, WA, AK, HI, ID, IA, KY, ME). This was narrowed down to 5 in the range of 200-400 different bill titles (RI, WA, AK, ID, KY).

To maximize the size of the dataset, WA was chosen as after scraping of the most recent session it had 3780 bills available. During testing another model was later trained on a WA dataset appended by KY with 1337 bills available. For testing AK was chosen.

During development, I investigated whether the model is better trained on strings of titles or strings of titles appended with subjects. The F1 score was significantly better for testing the model solely on title strings.

### Classifier 

Through literature research, One Vs Rest Linear Support Vector classifier was determined as the best classifier for a multi-label classification problem that subject extraction is. l2 penalty was chosen over l1 as the average F1 score was 0.60 vs 0.52 for l2 penalty.

### Vectorizer

TF-IDF (term frequency–inverse document frequency) vectorizer was trained on the set of bill titles. TFIDF was chosen for subject extraction as it is used to reflect how important a word is to a document (bill title in this context) in a corpus (dataset of bill titles).

Following parameters were determined as optimum during development and testing through the comparison of F1 scores:
min_df=0.001 - terms that appear in less than 0.1% of the documents are ignored
ngram_range=(1, 3) - any sequence of up to 3 neighboring words is considered as a phrase in training 

## Evaluating the model

In order to evaluate the model, F1 score was chosen initially. It is a good metric to evaluate the model when the testing data is from the same state as training data. However, when testing data is from a different state and labels differ significantly from the subject labels of the training state, F1 is not a good choice as it expectedly returns a very low scoreas very a small number of subjects is shared between subject lists of two states.  

### F1 score

Commonly used in natural language processing, it is interpreted as a weighted average of precision and recall where precision is the fraction of all retrieved correct labels among all the retrieved labels and recall is the fraction of all the retrieved correct labels among all the correct labels.

### Adjusted F1 score

I adjusted the F1 score in order to identify labels with similar meaning as equivalent ones, for example ELECTIONS and VOTING in order to have a better means of comparing results of predicting subjects on a state different than the training one.

### Word2Vec

Measuring the similarity between two strings was achieved using a Word2Vec model trained on the dataset containing strings of Washington bill titles appended with corresponding subjects. A trained Word2Vec model returns a vector representation of a given word if it was present in the training corpus. This is a common technique to obtain a semantic representation for words from the context they are used in. 

### TF-IDF Vectorizer

TF-IDF vectorizer was also used in the evaluation stage. As each subject can consist of more than one word, there is a need to obtain a vector for the whole phrase rather than just one word. This can be done by taking the average of the word vectors weighted by the tf-idf scores corresponding to each word as tf-idf score indicates the importance of a word in the phrase given a corpus of training data. In this case, the training data was the same as for the Word2Vec model (strings of Washington bill titles appended with subjects).

### Cosine similarity

Cosine similarity is the measure of the angle between two vectors. The smaller the angle the closer the vectors are to one another and in the context of word or phrase vectors, the closer the semantic values of two words are. Hence, this technique was used to determine the similarity between two subjects on the scale from 0 to 1 when 1 indicates the same meaning. Based on my observations of the cosine similarities in the WA Word2Vec model, I determined 0.98 to be a fair threshold over which the subjects can be replaced by one another.

### Results 

The model performed well when only one state was considered (i.e. splitting one dataset into testing and training samples). The F1 score in this case oscillated around 0.60. For testing on a different state (AK in this case), adjusted F1 score was used. When the training dataset included two states (in this case Washington and Kentucky), the adjusted F1 score was slightly lower than if the training dataset included just one state (Washington). This can be explained by an increased complexity of selecting an appropriate subject since the number of possible subjects increases significantly after adding a second state to the training dataset. The adjusted F1 score when predicting AK subjects for WA and KY dataset is 0.38 while for WA dataset it is 0.40. Therefore, I decided to use the model trained solely on WA for subject prediction.

## Future work

The most significant factor in increasing the quality of the classifier is increasing the quality and quantity of data. This could be achieved by creating a limited list of subjects that would be used to relabel bills from the already labeled states. This new multi-state subject-title dataset would then be used for training of the classifier. Relabeling could be achieved using cosine similarities and another machine learning algorithm.
In order to achieve more accurate cosine similarity values between two subjects, we would need to gather more data for Word2Vec model and tfidf-vectorizer training. An optimal dataset would contain titles and abstracts (and possibly bodies) of all scraped bills. 

## Built With

* [gensim](https://pypi.org/project/gensim/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [NLTK](https://www.nltk.org)
* [pandas](https://pandas.pydata.org)


## Authors

* **Bartłomiej Jezierski** - [GitHub](https://github.com/brjezierski)

