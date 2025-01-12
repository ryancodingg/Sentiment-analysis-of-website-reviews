{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib as plt\n",
    "import string\n",
    "from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import classification_report, accuracy_score, confusion_matrix\n",
    "from sklearn.model_selection import GridSearchCV, cross_val_score\n",
    "from xgboost import XGBClassifier\n",
    "from sklearn.svm import SVC\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.stem import PorterStemmer\n",
    "import nltk \n",
    "\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h1 style=\"color:#0855b1\">Task 1 - Data Loading and Data Preparation</h1>\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load train dataset\n",
    "df_train = pd.read_csv(r'/Users/nglongvu1002/Documents/Documents/Swinburne - Master of Data Science/Machine Learning/Dataset/x_train.csv')\n",
    "y_train = pd.read_csv(r'/Users/nglongvu1002/Documents/Documents/Swinburne - Master of Data Science/Machine Learning/Dataset/y_train.csv')\n",
    "\n",
    "#Load test dataset\n",
    "df_test = pd.read_csv(r'/Users/nglongvu1002/Documents/Documents/Swinburne - Master of Data Science/Machine Learning/Dataset/x_test.csv')\n",
    "y_test = pd.read_csv(r'/Users/nglongvu1002/Documents/Documents/Swinburne - Master of Data Science/Machine Learning/Dataset/y_test.csv')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>amazon</th>\n",
       "      <th>Oh and I forgot to also mention the weird color effect it has on your phone.</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>amazon</td>\n",
       "      <td>THAT one didn't work either.</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>amazon</td>\n",
       "      <td>Waste of 13 bucks.</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>amazon</td>\n",
       "      <td>Product is useless, since it does not have eno...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>amazon</td>\n",
       "      <td>None of the three sizes they sent with the hea...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>amazon</td>\n",
       "      <td>Worst customer service.</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   amazon  \\\n",
       "0  amazon   \n",
       "1  amazon   \n",
       "2  amazon   \n",
       "3  amazon   \n",
       "4  amazon   \n",
       "\n",
       "  Oh and I forgot to also mention the weird color effect it has on your phone.  \n",
       "0                       THAT one didn't work either.                            \n",
       "1                                 Waste of 13 bucks.                            \n",
       "2  Product is useless, since it does not have eno...                            \n",
       "3  None of the three sizes they sent with the hea...                            \n",
       "4                            Worst customer service.                            "
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Set column names for train dataset\n",
    "df_train.columns = ['website_name', 'text']\n",
    "y_train.columns = ['is_positive_sentiment']\n",
    "\n",
    "#Set column names for test dataset \n",
    "df_test.columns = ['website_name', 'text']\n",
    "y_test.columns = ['is_positive_sentiment']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Print the train datasett\n",
    "df_train['is_positive_sentiment'] = y_train['is_positive_sentiment']\n",
    "df_test['is_positive_sentiment'] = y_test['is_positive_sentiment']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# <h1 style=\"color:#0855b1\"> Task 2: Feature Representation </h1>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package stopwords to\n",
      "[nltk_data]     /Users/nglongvu1002/nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>website_name</th>\n",
       "      <th>text</th>\n",
       "      <th>is_positive_sentiment</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>amazon</td>\n",
       "      <td>one didnt work either</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>amazon</td>\n",
       "      <td>wast 13 buck</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>amazon</td>\n",
       "      <td>product useless sinc enough charg current char...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>amazon</td>\n",
       "      <td>none three size sent headset would stay ear</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>amazon</td>\n",
       "      <td>worst custom servic</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2394</th>\n",
       "      <td>yelp</td>\n",
       "      <td>sweet potato fri good season well</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2395</th>\n",
       "      <td>yelp</td>\n",
       "      <td>could eat bruschetta day devin</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2396</th>\n",
       "      <td>yelp</td>\n",
       "      <td>ambienc perfect</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2397</th>\n",
       "      <td>yelp</td>\n",
       "      <td>order duck rare pink tender insid nice char ou...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2398</th>\n",
       "      <td>yelp</td>\n",
       "      <td>servic good compani better</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>2399 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     website_name                                               text  \\\n",
       "0          amazon                              one didnt work either   \n",
       "1          amazon                                       wast 13 buck   \n",
       "2          amazon  product useless sinc enough charg current char...   \n",
       "3          amazon        none three size sent headset would stay ear   \n",
       "4          amazon                                worst custom servic   \n",
       "...           ...                                                ...   \n",
       "2394         yelp                  sweet potato fri good season well   \n",
       "2395         yelp                     could eat bruschetta day devin   \n",
       "2396         yelp                                    ambienc perfect   \n",
       "2397         yelp  order duck rare pink tender insid nice char ou...   \n",
       "2398         yelp                         servic good compani better   \n",
       "\n",
       "      is_positive_sentiment  \n",
       "0                         0  \n",
       "1                         0  \n",
       "2                         0  \n",
       "3                         0  \n",
       "4                         0  \n",
       "...                     ...  \n",
       "2394                      1  \n",
       "2395                      1  \n",
       "2396                      1  \n",
       "2397                      1  \n",
       "2398                      1  \n",
       "\n",
       "[2399 rows x 3 columns]"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Download stopwords\n",
    "nltk.download('stopwords')\n",
    "\n",
    "# Initialize PorterStemmer\n",
    "stemmer = PorterStemmer()\n",
    "stop_words = set(stopwords.words('english'))\n",
    "\n",
    "#Preprocess the text data\n",
    "def preprocessing_text(text):\n",
    "    text = text.lower() #Convert to lowercase\n",
    "    text = text.translate(str.maketrans('', '', string.punctuation)) #Remove punctuation\n",
    "    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stop words\n",
    "    text = ' '.join([stemmer.stem(word) for word in text.split()])  # Perform stemming\n",
    "\n",
    "    return text\n",
    "\n",
    "# Apply properocessing to the text in both training and test datasets\n",
    "df_train['text'] = df_train['text'].apply(preprocessing_text)\n",
    "df_test['text'] = df_test['text'].apply(preprocessing_text)\n",
    "\n",
    "df_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['10' 'absolut' 'act' 'actor' 'actual' 'almost' 'also' 'alway' 'amaz'\n",
      " 'amp' 'anoth' 'anyon' 'anyth' 'around' 'art' 'atmospher' 'attent' 'avoid'\n",
      " 'aw' 'away' 'awesom' 'back' 'bad' 'bar' 'bare' 'batteri' 'beauti'\n",
      " 'believ' 'best' 'better' 'big' 'bit' 'black' 'bland' 'bluetooth' 'bore'\n",
      " 'bought' 'buffet' 'burger' 'buy' 'call' 'came' 'camera' 'cant' 'car'\n",
      " 'care' 'case' 'cast' 'cell' 'charact' 'charg' 'charger' 'cheap' 'check'\n",
      " 'chicken' 'clear' 'color' 'come' 'comfort' 'complet' 'consid' 'cool'\n",
      " 'could' 'couldnt' 'custom' 'custom servic' 'day' 'deal' 'definit'\n",
      " 'delici' 'design' 'dialogu' 'didnt' 'differ' 'direct' 'director'\n",
      " 'disappoint' 'dish' 'doesnt' 'done' 'dont' 'dont think' 'drop' 'ear'\n",
      " 'easi' 'eat' 'effect' 'either' 'end' 'enjoy' 'enough' 'especi' 'even'\n",
      " 'ever' 'everi' 'everyon' 'everyth' 'excel' 'expect' 'experi' 'face'\n",
      " 'fact' 'famili' 'fantast' 'far' 'fast' 'feel' 'felt' 'film' 'find' 'fine'\n",
      " 'first' 'fit' 'flavor' 'food' 'found' 'fresh' 'fri' 'friend' 'friendli'\n",
      " 'full' 'funni' 'game' 'gave' 'gener' 'get' 'give' 'go' 'go back' 'good'\n",
      " 'got' 'great' 'hand' 'happi' 'hard' 'hate' 'headset' 'hear' 'help'\n",
      " 'highli' 'highli recommend' 'hold' 'hope' 'horribl' 'hour' 'howev' 'huge'\n",
      " 'id' 'ill' 'im' 'imagin' 'impress' 'incred' 'interest' 'item' 'ive'\n",
      " 'ive ever' 'job' 'keep' 'kind' 'know' 'lack' 'last' 'least' 'left' 'life'\n",
      " 'like' 'line' 'littl' 'live' 'long' 'look' 'lot' 'love' 'low' 'made'\n",
      " 'make' 'man' 'manag' 'mani' 'may' 'mean' 'menu' 'minut' 'money'\n",
      " 'motorola' 'movi' 'much' 'music' 'must' 'need' 'never' 'new' 'next'\n",
      " 'nice' 'night' 'noth' 'old' 'one' 'one best' 'order' 'overal' 'part'\n",
      " 'peopl' 'perfect' 'perfectli' 'perform' 'phone' 'pictur' 'piec' 'pizza'\n",
      " 'place' 'play' 'pleas' 'plot' 'plug' 'poor' 'possibl' 'predict' 'pretti'\n",
      " 'price' 'probabl' 'problem' 'product' 'purchas' 'put' 'qualiti' 'quit'\n",
      " 'rate' 'rather' 'real' 'realli' 'reason' 'receiv' 'recept' 'recommend'\n",
      " 'restaur' 'return' 'review' 'right' 'said' 'salad' 'saw' 'say' 'scene'\n",
      " 'screen' 'script' 'seat' 'second' 'see' 'seem' 'seen' 'serious' 'serv'\n",
      " 'server' 'servic' 'set' 'sever' 'show' 'side' 'simpli' 'sinc' 'slow'\n",
      " 'small' 'soon' 'sound' 'sound qualiti' 'special' 'staff' 'star' 'start'\n",
      " 'stay' 'steak' 'still' 'stori' 'stupid' 'suck' 'super' 'sure' 'sushi'\n",
      " 'tabl' 'take' 'talk' 'tast' 'tell' 'terribl' 'that' 'thing' 'think'\n",
      " 'though' 'thought' 'time' 'total' 'tri' 'turn' 'two' 'understand' 'us'\n",
      " 'use' 'vega' 'wait' 'want' 'wasnt' 'wast' 'wast money' 'wast time'\n",
      " 'watch' 'way' 'well' 'went' 'white' 'whole' 'without' 'wonder' 'wont'\n",
      " 'word' 'work' 'work great' 'work well' 'wors' 'worst' 'worth' 'would'\n",
      " 'would recommend' 'write' 'year']\n",
      "['10' 'absolut' 'act' 'actor' 'actual' 'almost' 'also' 'alway' 'amaz'\n",
      " 'amp' 'anoth' 'anyon' 'anyth' 'around' 'art' 'atmospher' 'attent' 'avoid'\n",
      " 'aw' 'away' 'awesom' 'back' 'bad' 'bar' 'bare' 'batteri' 'beauti'\n",
      " 'believ' 'best' 'better' 'big' 'bit' 'black' 'bland' 'bluetooth' 'bore'\n",
      " 'bought' 'buffet' 'burger' 'buy' 'call' 'came' 'camera' 'cant' 'car'\n",
      " 'care' 'case' 'cast' 'cell' 'charact' 'charg' 'charger' 'cheap' 'check'\n",
      " 'chicken' 'clear' 'color' 'come' 'comfort' 'complet' 'consid' 'cool'\n",
      " 'could' 'couldnt' 'custom' 'custom servic' 'day' 'deal' 'definit'\n",
      " 'delici' 'design' 'dialogu' 'didnt' 'differ' 'direct' 'director'\n",
      " 'disappoint' 'dish' 'doesnt' 'done' 'dont' 'dont think' 'drop' 'ear'\n",
      " 'easi' 'eat' 'effect' 'either' 'end' 'enjoy' 'enough' 'especi' 'even'\n",
      " 'ever' 'everi' 'everyon' 'everyth' 'excel' 'expect' 'experi' 'face'\n",
      " 'fact' 'famili' 'fantast' 'far' 'fast' 'feel' 'felt' 'film' 'find' 'fine'\n",
      " 'first' 'fit' 'flavor' 'food' 'found' 'fresh' 'fri' 'friend' 'friendli'\n",
      " 'full' 'funni' 'game' 'gave' 'gener' 'get' 'give' 'go' 'go back' 'good'\n",
      " 'got' 'great' 'hand' 'happi' 'hard' 'hate' 'headset' 'hear' 'help'\n",
      " 'highli' 'highli recommend' 'hold' 'hope' 'horribl' 'hour' 'howev' 'huge'\n",
      " 'id' 'ill' 'im' 'imagin' 'impress' 'incred' 'interest' 'item' 'ive'\n",
      " 'ive ever' 'job' 'keep' 'kind' 'know' 'lack' 'last' 'least' 'left' 'life'\n",
      " 'like' 'line' 'littl' 'live' 'long' 'look' 'lot' 'love' 'low' 'made'\n",
      " 'make' 'man' 'manag' 'mani' 'may' 'mean' 'menu' 'minut' 'money'\n",
      " 'motorola' 'movi' 'much' 'music' 'must' 'need' 'never' 'new' 'next'\n",
      " 'nice' 'night' 'noth' 'old' 'one' 'one best' 'order' 'overal' 'part'\n",
      " 'peopl' 'perfect' 'perfectli' 'perform' 'phone' 'pictur' 'piec' 'pizza'\n",
      " 'place' 'play' 'pleas' 'plot' 'plug' 'poor' 'possibl' 'predict' 'pretti'\n",
      " 'price' 'probabl' 'problem' 'product' 'purchas' 'put' 'qualiti' 'quit'\n",
      " 'rate' 'rather' 'real' 'realli' 'reason' 'receiv' 'recept' 'recommend'\n",
      " 'restaur' 'return' 'review' 'right' 'said' 'salad' 'saw' 'say' 'scene'\n",
      " 'screen' 'script' 'seat' 'second' 'see' 'seem' 'seen' 'serious' 'serv'\n",
      " 'server' 'servic' 'set' 'sever' 'show' 'side' 'simpli' 'sinc' 'slow'\n",
      " 'small' 'soon' 'sound' 'sound qualiti' 'special' 'staff' 'star' 'start'\n",
      " 'stay' 'steak' 'still' 'stori' 'stupid' 'suck' 'super' 'sure' 'sushi'\n",
      " 'tabl' 'take' 'talk' 'tast' 'tell' 'terribl' 'that' 'thing' 'think'\n",
      " 'though' 'thought' 'time' 'total' 'tri' 'turn' 'two' 'understand' 'us'\n",
      " 'use' 'vega' 'wait' 'want' 'wasnt' 'wast' 'wast money' 'wast time'\n",
      " 'watch' 'way' 'well' 'went' 'white' 'whole' 'without' 'wonder' 'wont'\n",
      " 'word' 'work' 'work great' 'work well' 'wors' 'worst' 'worth' 'would'\n",
      " 'would recommend' 'write' 'year']\n"
     ]
    }
   ],
   "source": [
    "# Initialize CountVectorizer\n",
    "vectorizer = CountVectorizer(\n",
    "    max_features=1000,       # Limit vocabulary size to 1000 words\n",
    "    min_df=10,               # Exclude words that appear in fewer than 10 documents\n",
    "    max_df=0.5,              # Exclude words that appear in more than 50% of documents\n",
    "    ngram_range=(1, 2),      # Use unigrams and bigrams\n",
    "    binary=False             # Use count values, not binary\n",
    ")\n",
    "# Fit and transform the text data\n",
    "X_train_bow = vectorizer.fit_transform(df_train['text'])\n",
    "X_test_bow = vectorizer.transform(df_test['text'])\n",
    "\n",
    "# Initialize TfidfVectorizer\n",
    "tfidf_vectorizer = TfidfVectorizer(\n",
    "    max_features=1000,       # Limit vocabulary size to 1000 words\n",
    "    min_df=10,               # Exclude words that appear in fewer than 10 documents\n",
    "    max_df=0.5,              # Exclude words that appear in more than 50% of documents\n",
    "    ngram_range=(1, 2),      # Use unigrams and bigrams\n",
    ")\n",
    "\n",
    "# Fit and transform the text data\n",
    "X_train_tfidf = tfidf_vectorizer.fit_transform(df_train['text'])\n",
    "X_test_tfidf = tfidf_vectorizer.transform(df_test['text'])\n",
    "\n",
    "\n",
    "print(vectorizer.get_feature_names_out())  # For CountVectorizer\n",
    "print(tfidf_vectorizer.get_feature_names_out())  # For TfidfVectorizer\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Performance with CountVectorizer:\n",
      "Accuracy: 0.7529215358931552\n",
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.73      0.80      0.76       299\n",
      "           1       0.78      0.71      0.74       300\n",
      "\n",
      "    accuracy                           0.75       599\n",
      "   macro avg       0.75      0.75      0.75       599\n",
      "weighted avg       0.75      0.75      0.75       599\n",
      "\n",
      "Confusion Matrix:\n",
      " [[238  61]\n",
      " [ 87 213]]\n",
      "\n",
      "Performance with TfidfVectorizer:\n",
      "Accuracy: 0.7412353923205343\n",
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.71      0.81      0.76       299\n",
      "           1       0.78      0.68      0.72       300\n",
      "\n",
      "    accuracy                           0.74       599\n",
      "   macro avg       0.75      0.74      0.74       599\n",
      "weighted avg       0.75      0.74      0.74       599\n",
      "\n",
      "Confusion Matrix:\n",
      " [[241  58]\n",
      " [ 97 203]]\n"
     ]
    }
   ],
   "source": [
    "# Initialize the model\n",
    "model = RandomForestClassifier(random_state=42)\n",
    "\n",
    "# Train and evaluate the model using CountVectorizer (BoW)\n",
    "model.fit(X_train_bow, y_train['is_positive_sentiment'])\n",
    "y_pred_bow = model.predict(X_test_bow)\n",
    "\n",
    "# Evaluate performance with CountVectorizer\n",
    "print(\"Performance with CountVectorizer:\")\n",
    "print(\"Accuracy:\", accuracy_score(y_test['is_positive_sentiment'], y_pred_bow))\n",
    "print(\"Classification Report:\\n\", classification_report(y_test['is_positive_sentiment'], y_pred_bow))\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(y_test['is_positive_sentiment'], y_pred_bow))\n",
    "\n",
    "# Train and evaluate the model using TfidfVectorizer\n",
    "model.fit(X_train_tfidf, y_train['is_positive_sentiment'])\n",
    "y_pred_tfidf = model.predict(X_test_tfidf)\n",
    "\n",
    "# Evaluate performance with TfidfVectorizer\n",
    "print(\"\\nPerformance with TfidfVectorizer:\")\n",
    "print(\"Accuracy:\", accuracy_score(y_test['is_positive_sentiment'], y_pred_tfidf))\n",
    "print(\"Classification Report:\\n\", classification_report(y_test['is_positive_sentiment'], y_pred_tfidf))\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(y_test['is_positive_sentiment'], y_pred_tfidf))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h1 style=\"color:#0855b1\"> Task 3: Classification and Evaluation </h1>\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split the data into training and validation sets (80-20 split as an example)\n",
    "X_train_split, X_val, y_train_split, y_val = train_test_split(\n",
    "    X_train_bow, df_train['is_positive_sentiment'], test_size=0.2, random_state=42\n",
    ")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2 style =color:#4fa5d8>Logistic Regression Model </h2>\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic Regression Evaluation with Initial Parameters:\n",
      "Accuracy: 0.7979166666666667\n",
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.78      0.83      0.80       240\n",
      "           1       0.82      0.76      0.79       240\n",
      "\n",
      "    accuracy                           0.80       480\n",
      "   macro avg       0.80      0.80      0.80       480\n",
      "weighted avg       0.80      0.80      0.80       480\n",
      "\n",
      "Confusion Matrix:\n",
      " [[200  40]\n",
      " [ 57 183]]\n"
     ]
    }
   ],
   "source": [
    " #Initialize and train the Logistic Regression model with default parameters\n",
    "lr_classifier = LogisticRegression(random_state=42, max_iter=1000)\n",
    "lr_classifier.fit(X_train_split, y_train_split)\n",
    "\n",
    "#Make predictions on the validation set\n",
    "y_val_pred_lr = lr_classifier.predict(X_val)\n",
    "\n",
    "# Evaluate using accuracy, classification report, and confusion matrix\n",
    "print(\"Logistic Regression Evaluation with Initial Parameters:\")\n",
    "print(\"Accuracy:\", accuracy_score(y_val, y_val_pred_lr))\n",
    "print(\"Classification Report:\\n\", classification_report(y_val, y_val_pred_lr))\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(y_val, y_val_pred_lr))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 3 folds for each of 16 candidates, totalling 48 fits\n",
      "Best Parameters for Logistic Regression: {'C': 1, 'penalty': 'l1', 'solver': 'liblinear'}\n",
      "Logistic Regression Evaluation with Best Parameters:\n",
      "Accuracy: 0.8041666666666667\n",
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.78      0.85      0.81       240\n",
      "           1       0.83      0.76      0.80       240\n",
      "\n",
      "    accuracy                           0.80       480\n",
      "   macro avg       0.81      0.80      0.80       480\n",
      "weighted avg       0.81      0.80      0.80       480\n",
      "\n",
      "Confusion Matrix:\n",
      " [[203  37]\n",
      " [ 57 183]]\n"
     ]
    }
   ],
   "source": [
    "#Define the parameter grid for GridSearchCV\n",
    "param_grid_lr = {\n",
    "    'C': [0.1, 1, 10, 100],  # Inverse of regularization strength\n",
    "    'penalty': ['l1', 'l2'],  # Norm used in penalization\n",
    "    'solver': ['liblinear', 'saga']  # Solvers that support both l1 and l2 penalties\n",
    "}\n",
    "\n",
    "#Use GridSearchCV to find the best hyperparameters\n",
    "grid_search_lr = GridSearchCV(estimator=LogisticRegression(max_iter=1000, random_state=42), \n",
    "                              param_grid=param_grid_lr, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)\n",
    "grid_search_lr.fit(X_train_split, y_train_split)\n",
    "\n",
    "#Get the best parameters and model\n",
    "best_params_lr = grid_search_lr.best_params_\n",
    "best_lr_model = grid_search_lr.best_estimator_\n",
    "print(\"Best Parameters for Logistic Regression:\", best_params_lr)\n",
    "\n",
    "# Evaluate the best model on the validation set\n",
    "y_val_pred_best_lr = best_lr_model.predict(X_val)\n",
    "\n",
    "print(\"Logistic Regression Evaluation with Best Parameters:\")\n",
    "print(\"Accuracy:\", accuracy_score(y_val, y_val_pred_best_lr))\n",
    "print(\"Classification Report:\\n\", classification_report(y_val, y_val_pred_best_lr))\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(y_val, y_val_pred_best_lr))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic Regression Evaluation on Test Set with Best Parameters:\n",
      "Accuracy: 0.7495826377295493\n",
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.71      0.83      0.77       299\n",
      "           1       0.80      0.67      0.73       300\n",
      "\n",
      "    accuracy                           0.75       599\n",
      "   macro avg       0.76      0.75      0.75       599\n",
      "weighted avg       0.76      0.75      0.75       599\n",
      "\n",
      "Confusion Matrix:\n",
      " [[249  50]\n",
      " [100 200]]\n"
     ]
    }
   ],
   "source": [
    "X_test_bow = vectorizer.transform(df_test['text'])\n",
    "y_test_pred_best_lr = best_lr_model.predict(X_test_bow)\n",
    "\n",
    "print(\"Logistic Regression Evaluation on Test Set with Best Parameters:\")\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_test_pred_best_lr))\n",
    "print(\"Classification Report:\\n\", classification_report(y_test, y_test_pred_best_lr))\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(y_test, y_test_pred_best_lr))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2 style =color:#4fa5d8>Random Forest Model</h2>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest Evaluation with Initial Parameters:\n",
      "Accuracy: 0.8104166666666667\n",
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.80      0.83      0.81       240\n",
      "           1       0.83      0.79      0.81       240\n",
      "\n",
      "    accuracy                           0.81       480\n",
      "   macro avg       0.81      0.81      0.81       480\n",
      "weighted avg       0.81      0.81      0.81       480\n",
      "\n",
      "Confusion Matrix:\n",
      " [[200  40]\n",
      " [ 51 189]]\n"
     ]
    }
   ],
   "source": [
    "# Initialize and train the Random Forest model with default parameters\n",
    "rf_classifier = RandomForestClassifier(random_state=42)\n",
    "rf_classifier.fit(X_train_split, y_train_split)\n",
    "\n",
    "#Make predictions on the validation set\n",
    "y_val_pred_rf = rf_classifier.predict(X_val)\n",
    "\n",
    "#Evaluate using accuracy, classification report, and confusion matrix\n",
    "print(\"Random Forest Evaluation with Initial Parameters:\")\n",
    "print(\"Accuracy:\", accuracy_score(y_val, y_val_pred_rf))\n",
    "print(\"Classification Report:\\n\", classification_report(y_val, y_val_pred_rf))\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(y_val, y_val_pred_rf))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 3 folds for each of 216 candidates, totalling 648 fits\n",
      "Best Parameters for Random Forest: {'bootstrap': True, 'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 300}\n",
      "Random Forest Evaluation with Best Parameters:\n",
      "Accuracy: 0.7854166666666667\n",
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.72      0.93      0.81       240\n",
      "           1       0.90      0.65      0.75       240\n",
      "\n",
      "    accuracy                           0.79       480\n",
      "   macro avg       0.81      0.79      0.78       480\n",
      "weighted avg       0.81      0.79      0.78       480\n",
      "\n",
      "Confusion Matrix:\n",
      " [[222  18]\n",
      " [ 85 155]]\n"
     ]
    }
   ],
   "source": [
    "# Define the parameter grid for GridSearchCV\n",
    "param_grid_rf = {\n",
    "    'n_estimators': [100, 200, 300],        # Number of trees in the forest\n",
    "    'max_depth': [None, 10, 20, 30],        # Maximum depth of each tree\n",
    "    'min_samples_split': [2, 5, 10],        # Minimum number of samples required to split a node\n",
    "    'min_samples_leaf': [1, 2, 4],          # Minimum number of samples required at a leaf node\n",
    "    'bootstrap': [True, False]              # Whether bootstrap samples are used when building trees\n",
    "}\n",
    "\n",
    "#Use GridSearchCV to find the best hyperparameters\n",
    "grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42), \n",
    "                              param_grid=param_grid_rf, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)\n",
    "grid_search_rf.fit(X_train_split, y_train_split)\n",
    "\n",
    "#Get the best parameters and model\n",
    "best_params_rf = grid_search_rf.best_params_\n",
    "best_rf_model = grid_search_rf.best_estimator_\n",
    "print(\"Best Parameters for Random Forest:\", best_params_rf)\n",
    "\n",
    "#Evaluate the best model on the validation set\n",
    "y_val_pred_best_rf = best_rf_model.predict(X_val)\n",
    "\n",
    "print(\"Random Forest Evaluation with Best Parameters:\")\n",
    "print(\"Accuracy:\", accuracy_score(y_val, y_val_pred_best_rf))\n",
    "print(\"Classification Report:\\n\", classification_report(y_val, y_val_pred_best_rf))\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(y_val, y_val_pred_best_rf))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest Evaluation on Test Set with Best Parameters:\n",
      "Confusion Matrix:\n",
      " [[272  27]\n",
      " [133 167]]\n",
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.67      0.91      0.77       299\n",
      "           1       0.86      0.56      0.68       300\n",
      "\n",
      "    accuracy                           0.73       599\n",
      "   macro avg       0.77      0.73      0.72       599\n",
      "weighted avg       0.77      0.73      0.72       599\n",
      "\n",
      "Accuracy: 0.7328881469115192\n"
     ]
    }
   ],
   "source": [
    "# Make predictions on the test set using the best model\n",
    "y_test_pred_rf = best_rf_model.predict(X_test_bow)\n",
    "\n",
    "# Step 5: Evaluate the model on the test set\n",
    "print(\"Random Forest Evaluation on Test Set with Best Parameters:\")\n",
    "conf_matrix_test_rf = confusion_matrix(y_test, y_test_pred_rf)\n",
    "class_report_test_rf = classification_report(y_test, y_test_pred_rf)\n",
    "accuracy_test_rf = accuracy_score(y_test, y_test_pred_rf)\n",
    "\n",
    "print(\"Confusion Matrix:\\n\", conf_matrix_test_rf)\n",
    "print(\"Classification Report:\\n\", class_report_test_rf)\n",
    "print(\"Accuracy:\", accuracy_test_rf)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2 style =color:#4fa5d8> SVC Model </h2>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SVC Model Evaluation:\n",
      "Accuracy: 0.79375\n",
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.75      0.88      0.81       240\n",
      "           1       0.85      0.71      0.77       240\n",
      "\n",
      "    accuracy                           0.79       480\n",
      "   macro avg       0.80      0.79      0.79       480\n",
      "weighted avg       0.80      0.79      0.79       480\n",
      "\n",
      "Confusion Matrix:\n",
      " [[211  29]\n",
      " [ 70 170]]\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# Initialize and train the SVC model\n",
    "svc_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)  # You can customize parameters if needed\n",
    "svc_classifier.fit(X_train_split, y_train_split)  # Train the model\n",
    "\n",
    "# Make predictions on the validation set\n",
    "y_val_pred_svc = svc_classifier.predict(X_val)\n",
    "\n",
    "# Evaluate using accuracy, classification report, and confusion matrix\n",
    "print(\"SVC Model Evaluation:\")\n",
    "print(\"Accuracy:\", accuracy_score(y_val, y_val_pred_svc))\n",
    "print(\"Classification Report:\\n\", classification_report(y_val, y_val_pred_svc))\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(y_val, y_val_pred_svc))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 3 folds for each of 20 candidates, totalling 60 fits\n",
      "Best Parameters for SVC: {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}\n",
      "SVC Model Evaluation with Best Parameters:\n",
      "Accuracy: 0.79375\n",
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.75      0.88      0.81       240\n",
      "           1       0.85      0.71      0.77       240\n",
      "\n",
      "    accuracy                           0.79       480\n",
      "   macro avg       0.80      0.79      0.79       480\n",
      "weighted avg       0.80      0.79      0.79       480\n",
      "\n",
      "Confusion Matrix:\n",
      " [[211  29]\n",
      " [ 70 170]]\n"
     ]
    }
   ],
   "source": [
    "param_grid_svc = {\n",
    "    'C': [0.1, 1, 10, 100],       # Regularization parameter\n",
    "    'gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Kernel coefficient\n",
    "    'kernel': ['rbf']             # Using RBF kernel\n",
    "}\n",
    "\n",
    "grid_search_svc = GridSearchCV(estimator=SVC(), param_grid=param_grid_svc, \n",
    "                               cv=3, scoring='accuracy', verbose=2, n_jobs=-1)\n",
    "grid_search_svc.fit(X_train_split, y_train_split)\n",
    "\n",
    "#Get the best parameters and model\n",
    "best_params_svc = grid_search_svc.best_params_\n",
    "best_svc_model = grid_search_svc.best_estimator_\n",
    "print(\"Best Parameters for SVC:\", best_params_svc)\n",
    "\n",
    "#Evaluate the best model on the validation set\n",
    "y_val_pred_best_svc = best_svc_model.predict(X_val)\n",
    "\n",
    "print(\"SVC Model Evaluation with Best Parameters:\")\n",
    "print(\"Accuracy:\", accuracy_score(y_val, y_val_pred_best_svc))\n",
    "print(\"Classification Report:\\n\", classification_report(y_val, y_val_pred_best_svc))\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(y_val, y_val_pred_best_svc))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SVC Model Evaluation on Test Set:\n",
      "Confusion Matrix:\n",
      " [[254  45]\n",
      " [114 186]]\n",
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.69      0.85      0.76       299\n",
      "           1       0.81      0.62      0.70       300\n",
      "\n",
      "    accuracy                           0.73       599\n",
      "   macro avg       0.75      0.73      0.73       599\n",
      "weighted avg       0.75      0.73      0.73       599\n",
      "\n",
      "Accuracy: 0.7345575959933222\n"
     ]
    }
   ],
   "source": [
    "# Make predictions on the test set using the trained SVC model\n",
    "y_test_pred_svc = svc_classifier.predict(X_test_bow)\n",
    "\n",
    "# Evaluate the SVC model on the test set\n",
    "print(\"SVC Model Evaluation on Test Set:\")\n",
    "conf_matrix_test_svc = confusion_matrix(y_test, y_test_pred_svc)\n",
    "class_report_test_svc = classification_report(y_test, y_test_pred_svc)\n",
    "accuracy_test_svc = accuracy_score(y_test, y_test_pred_svc)\n",
    "\n",
    "print(\"Confusion Matrix:\\n\", conf_matrix_test_svc)\n",
    "print(\"Classification Report:\\n\", class_report_test_svc)\n",
    "print(\"Accuracy:\", accuracy_test_svc)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2 style =color:#4fa5d8>XGBoost </h2>\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 3 folds for each of 243 candidates, totalling 729 fits\n",
      "Best Parameters for XGBoost: {'colsample_bytree': 0.6, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 300, 'subsample': 1.0}\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/nglongvu1002/opt/anaconda3/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [22:18:26] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:740: \n",
      "Parameters: { \"use_label_encoder\" } are not used.\n",
      "\n",
      "  warnings.warn(smsg, UserWarning)\n"
     ]
    }
   ],
   "source": [
    "# Initialize the XGBoost model\n",
    "xgb_classifier = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')\n",
    "\n",
    "# Define the parameter grid for GridSearchCV\n",
    "param_grid_xgb = {\n",
    "    'n_estimators': [100, 200, 300],   # Number of boosting rounds\n",
    "    'max_depth': [3, 5, 7],            # Maximum depth of a tree\n",
    "    'learning_rate': [0.01, 0.1, 0.2], # Step size shrinkage\n",
    "    'subsample': [0.6, 0.8, 1.0],      # Fraction of samples used per tree\n",
    "    'colsample_bytree': [0.6, 0.8, 1.0] # Fraction of features used per tree\n",
    "}\n",
    "\n",
    "# Use GridSearchCV to find the best hyperparameters for XGBoost\n",
    "grid_search_xgb = GridSearchCV(\n",
    "    estimator=xgb_classifier,\n",
    "    param_grid=param_grid_xgb,\n",
    "    cv=3,\n",
    "    scoring='accuracy',\n",
    "    verbose=2,\n",
    "    n_jobs=-1\n",
    ")\n",
    "\n",
    "# Fit GridSearchCV on the training split\n",
    "grid_search_xgb.fit(X_train_split, y_train_split)\n",
    "\n",
    "# Get the best parameters and the best model for XGBoost\n",
    "best_params_xgb = grid_search_xgb.best_params_\n",
    "best_xgb_model = grid_search_xgb.best_estimator_\n",
    "print(\"Best Parameters for XGBoost:\", best_params_xgb)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "XGBoost Model Evaluation with Best Parameters on Validation Set:\n",
      "Accuracy: 0.8020833333333334\n",
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.76      0.88      0.82       240\n",
      "           1       0.86      0.72      0.79       240\n",
      "\n",
      "    accuracy                           0.80       480\n",
      "   macro avg       0.81      0.80      0.80       480\n",
      "weighted avg       0.81      0.80      0.80       480\n",
      "\n",
      "Confusion Matrix:\n",
      " [[211  29]\n",
      " [ 66 174]]\n"
     ]
    }
   ],
   "source": [
    "# Evaluate the best model on the validation set\n",
    "y_val_pred_best_xgb = best_xgb_model.predict(X_val)\n",
    "\n",
    "print(\"\\nXGBoost Model Evaluation with Best Parameters on Validation Set:\")\n",
    "print(\"Accuracy:\", accuracy_score(y_val, y_val_pred_best_xgb))\n",
    "print(\"Classification Report:\\n\", classification_report(y_val, y_val_pred_best_xgb))\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(y_val, y_val_pred_best_xgb))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "XGBoost Model Evaluation on Test Set:\n",
      "Confusion Matrix:\n",
      " [[256  43]\n",
      " [114 186]]\n",
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.69      0.86      0.77       299\n",
      "           1       0.81      0.62      0.70       300\n",
      "\n",
      "    accuracy                           0.74       599\n",
      "   macro avg       0.75      0.74      0.73       599\n",
      "weighted avg       0.75      0.74      0.73       599\n",
      "\n",
      "Accuracy: 0.7378964941569283\n"
     ]
    }
   ],
   "source": [
    "# Make predictions on the test set using the best model\n",
    "y_test_pred_best_xgb = best_xgb_model.predict(X_test_bow)\n",
    "\n",
    "# Evaluate the XGBoost model on the test set\n",
    "print(\"\\nXGBoost Model Evaluation on Test Set:\")\n",
    "conf_matrix_test_xgb = confusion_matrix(y_test, y_test_pred_best_xgb)\n",
    "class_report_test_xgb = classification_report(y_test, y_test_pred_best_xgb)\n",
    "accuracy_test_xgb = accuracy_score(y_test, y_test_pred_best_xgb)\n",
    "\n",
    "print(\"Confusion Matrix:\\n\", conf_matrix_test_xgb)\n",
    "print(\"Classification Report:\\n\", class_report_test_xgb)\n",
    "print(\"Accuracy:\", accuracy_test_xgb)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
