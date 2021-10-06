# Written by Stuart Ryan, October 2021
# Python packages for Natrual Language Processing
import re
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
# Python package for TensorFlow
#import tensorflow
# Python package for pandas
import pandas as pd
import re
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


'''
Email data gotten from: https://www.kaggle.com/veleon/ham-and-spam-dataset
Text data gotten from: https://www.kaggle.com/uciml/sms-spam-collection-dataset
'''

### Text Data Collecting and Preprocessing ###
texts_df = pd.read_csv(r"text_data\spam.csv")
print(texts_df)
texts_df.dropna(1, inplace=True) #Removing excess variable columns
print(texts_df)

def preprocessor(text): # Getting rid of emojis as they are commonplace in text messages 
    emojis = re.compile(
    "(["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "])")

    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    text = re.sub(emojis, r' \1 ', text)
    return text

texts_df['v1'] = texts_df['v1'].apply(preprocessor)
print(texts_df)

### Text Data Model Creation ###
nltk.download('stopwords')
stop = stopwords.words('english')
porter = PorterStemmer()
def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

# Approx. 80% of the data allocated to training
X_train = texts_df.loc[:4456, 'v1'].values
y_train = texts_df.loc[:4456, 'v2'].values
X_test = texts_df.loc[4456:, 'v1'].values
y_test = texts_df.loc[4456:, 'v2'].values
tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

# Look to expand the paramater grid to include as many relevant variables as possible
param_grid = [{'vect__ngram_range': [(1, 2), (2, 2)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__alpha': [0.0001, 0.01, 1.0]},
              {'vect__ngram_range': [(1, 2), (2, 2)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__alpha': [0.0001, 0.01, 1.0]},
              ]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', SGDClassifier(loss='log', random_state=1))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=2,
                           n_jobs=-1)

gs_lr_tfidf.fit(X_train, y_train)

print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)

bestModel = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % bestModel.score(X_test, y_test))

### Email Data Preprocessing ###


