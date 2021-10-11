# Written by Stuart Ryan, October 2021

# Python packages for Natrual Language Processing
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Python package for holding data
import pandas as pd

# Sklearn packages for model creation
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Python package for parsing data from email files
import email

'''
Email data gotten from: https://www.kaggle.com/veleon/ham-and-spam-dataset
Text data gotten from: https://www.kaggle.com/uciml/sms-spam-collection-dataset5
'''
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
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

    text = text.lower() # lowercase text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', '')) # getting rid of emoticons
    text = re.sub(emojis, r' \1 ', text) # getting rid of emojis
    return text

porter = PorterStemmer()
def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
        
def text_model_creation():
    ### Text Data Collecting and Preprocessing ###
    texts_df = pd.read_csv(r"text_data\spam.csv", encoding="ISO-8859-1")
    texts_df.dropna(1, inplace=True) #Removing excess variable columns
    texts_df['v2'] = texts_df['v2'].apply(clean_text)
    print(texts_df)

    ### Text Data Model Creation ###
    # 70% of the data allocated to training
    X = texts_df.v2
    y = texts_df.v1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 1)

    tfidf = TfidfVectorizer(strip_accents=None,
                            lowercase=False,
                            preprocessor=None)

    # Look to expand the paramater grid to include as many relevant variables as possible
    '''
    Parameter choices:
    TfIdfVectorizer:
        - ngram_range: Chose to consider 1. unigrams and 2. unigrams with bigrams for text messages as texts are short and often times contain few keywords
        - tokenizer: Testing to see if stemming would be beneficial given the informal nature of words in texts (slang words)
    SGDClassifier:
        - clf_penalty: Chose to use both l1 and l2 because not sure whether the built in feature selection of l2 will be more important than l1's ability to ignore outliers in our dataset
        - clf_loss: Comparing log to hinge to compare if SVM is better for classification than logisitc regression
    '''
    param_grid = [{'vect__ngram_range': [(1,1), (1, 2), (2,2)],
                'vect__tokenizer': [tokenizer, tokenizer_porter],
                'vect__use_idf':[False, True],
                'clf__penalty': ['l1', 'l2'],
                'clf__alpha': [.00001, .0001, .001, .01],
                'clf__loss': ['log', 'hinge']
                }
                ]

    lr_tfidf = Pipeline([('vect', tfidf),
                        ('clf', SGDClassifier(loss='log', random_state=1, max_iter=5, tol=None))])

    gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                            scoring='accuracy',
                            cv=5,
                            verbose=2,
                            n_jobs=-1) # Utilizes all cores on machine to speed up grid search

    gs_lr_tfidf.fit(X_train, y_train)

    print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
    print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)

    bestModel = gs_lr_tfidf.best_estimator_
    print('Test Accuracy: %.3f' % bestModel.score(X_test, y_test))

    '''
    The following was the best parameter set found using the above grid search:
    Best parameter set: {'clf__alpha': 1e-05, 'clf__loss': 'log', 'clf__penalty': 'l2', 'vect__ngram_range': (1, 2), 'vect__tokenizer': <function tokenizer_porter at 0x0000023946B04438>, 'vect__use_idf': True} 
    CV Accuracy: 0.984
    Test Accuracy: 0.983
    '''
def email_model_creation():
    ### Email Data Collecting and Preprocessing ###
    
    email_df = pd.read_csv(r"text_data\spam.csv", encoding="ISO-8859-1")
    email_df.dropna(1, inplace=True) #Removing excess variable columns
    #texts_df['v2'] = texts_df['v2'].apply(clean_text)
    #print(texts_df)
    pass

email_model_creation()
