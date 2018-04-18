import nltk 
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.corpus import stopwords 
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk import Text 
import re 
import string 
import json 
import langid 
from sklearn.feature_extraction.text import TfidfVectorizer 
import h2o 
import pandas as pd
from h2o.estimators.naive_bayes import H2ONaiveBayesEstimator 
from h2o.estimators.random_forest import H2ORandomForestEstimator 
import random 


def tokenize(doc, lemmr):
    tokens = []
    stops = stopwords.words('english')

    doc = re.sub('['+string.punctuation+string.digits+']', ' ', doc) # Remove digits and punctuations 
    
    def is_good(token):
        return (len(token) >= 2 and not token in stops) 

    # Tokenize
    for sentence in sent_tokenize(doc): 
        for token in word_tokenize(sentence):
            # Ignore punctuation and stopwords
            if is_good(token): 
                lemma = lemmr.lemmatize(token.lower())
                if is_good(lemma):
                    tokens.append(lemma)
    return tokens

class Tokenizer():
    """
    Custom tokenizer for vectorization 
    Uses Lemmatization 
    """
    def __init__(self):
        self.lemmr = WordNetLemmatizer()

    def __call__(self, doc):
        return tokenize(doc, lemmr=self.lemmr)


TAGS = ['WiFi',  
        'Quietness', 
        'Bathroom',
        'Facilities',
        'Staff', 
        'Parking & Transport',  
        'Location', 
        'In-room facilities', 
        'Spa & Gym', 
        'Price', 
        'Cleanliness', 
        'Ambiance', 
        'Spaciousness',  
        'Food & Beverage', 
        'Breakfast', 
        'Bedding',  
        'Freebies', 
        'Views & Surroundings', 
        'Host' 
    ] 

def tag_index(tag): 
    TAGS_INDEX = {
        'WiFi': 0, 
        'Quietness': 1, 
        'Bathroom': 2,
        'Facilities': 3,
        'Staff': 4, 
        'Parking & Transport': 5,  
        'Location': 6, 
        'In-room facilities': 7, 
        'Spa & Gym': 8, 
        'Price': 9, 
        'Cleanliness': 10, 
        'Ambiance': 11, 
        'Spaciousness': 12,  
        'Food & Beverage': 13, 
        'Breakfast': 14, 
        'Bedding': 15,  
        'Freebies': 16, 
        'Views & Surroundings': 17, 
        'Host': 18  
    }   
    return TAGS_INDEX[tag] 


def transform_reviews(reviews): 
    """
    Transform reviews for specific hotel 
    human_reviews field -> list of {negative, positive, tags(list)}   
    """ 
    hotel_reviews = {}
    for cats, reviews in reviews.items():
        for review in reviews: 
            negative = review['negative'].encode('ascii', errors='ignore').decode()  
            positive = review['positive'].encode('ascii', errors='ignore').decode() 
            full = ' '.join([negative, positive]) 
            if full in hotel_reviews: 
                pass
            elif full.strip() is not '': #Not interested in reviews without text  
                hotel_reviews.update({full: {'negative': negative, 'positive': positive, 'tags': len(TAGS) * [0]}})
            else:
                continue
            for tag in cats.split(','):
                hotel_reviews[full]['tags'][tag_index(tag)] = 1 
    return [review for _, review in hotel_reviews.items()]        


def combine_reviews(records): 
    """ 
    Combine reviews for all hotels 
    """ 
    collection = [] 
    for record in records: 
        hotel_id = record['hotel_id']  
        reviews = transform_reviews(record['reviews'])  
        for review in reviews: 
            review.update({'hotel_id': hotel_id})
        collection.extend(reviews) 
    return collection


def clean_collection(collection, language='en'): 
    return [review for review in collection if get_lang(review['negative'] + review['positive']) == language]  


def extract_full(collection): 
    """ 
    Extracts a list of reviews full texts, and a dataframe with tags  
    """ 
    reviews_full = [] 
    reviews_tags = [] 
    reviews_hotel = []  
    for review in collection: 
        reviews_full.append(' '.join([review['negative'], review['positive']])) 
        reviews_tags.append(review['tags'])  
        reviews_hotel.append(review['hotel_id'])
    df_tags = pd.DataFrame(reviews_tags, columns=TAGS)
    df_hotels = pd.DataFrame(reviews_hotel, columns=['hotel_id'])

    return reviews_full, df_tags, df_hotels      


def extract_reviews(collection):
    """ 
    Extracts lists of positive/negative reviews, and a dataframe with tags  
    """ 
    reviews_negative = [] 
    reviews_positive = [] 
    reviews_tags = [] 
    reviews_hotel = []  
    for review in collection: 
        reviews_negative.append(review['negative']) 
        reviews_positive.append(review['positive']) 
        reviews_tags.append(review['tags'])  
        reviews_hotel.append(review['hotel_id'])
    df_tags = pd.DataFrame(reviews_tags, columns=TAGS)
    df_hotels = pd.DataFrame(reviews_hotel, columns=['hotel_id'])

    return reviews_negative, reviews_positive, df_tags, df_hotels           

def extract_scores(collection): 
    """ 
    Extracts scores for hotels ['reviews']['scores']   
    """ 
    hotel_scores = [] 
    hotels = [] 
    for review in collection: 
        hotels.append(review['hotel_id']) 
        hotel_scores.append(review['hotel_scores']) 
    scores = pd.DataFrame(hotel_scores, dtype='float')  
    scores['hotel_id'] = hotels 
    return scores[['hotel_id', 'cleanliness', 'facilities', 'free_wifi', 'staff', 'location', 'comfort', 'value_for_money']].groupby('hotel_id').mean()  

def fit_tfidf(collection): 
    vectorizer = TfidfVectorizer(input='content', stop_words='english', lowercase=True, tokenizer=Tokenizer(), min_df=0.02, max_df=1.0)  
    vectorizer.fit(collection) 
    return vectorizer  

def get_lang(text):
    return langid.classify(text)[0]
 
def load_data(reviews_vectors, reviews_tags, words): 
    """ 
    Vectorizes reviews using trained TF IDF vectorizer and loads it to H2O along with tags  
    Returns reference to H2O data frame containing TF IDF values and tags      
    """ 
    x = pd.DataFrame(reviews_vectors.toarray(), columns = words) 
    x_water = h2o.H2OFrame.from_python(x.to_dict('list')) 
    y = reviews_tags.copy() 
    y.columns = normalize_columns([*reviews_tags.columns])   
    y_water = h2o.H2OFrame.from_python(y.to_dict('list')) 
    frame_water = x_water.cbind(y_water.asfactor())
    return frame_water     

def vectorize_reviews(reviews_text, vectorizer): 
    """ 
    Vectorizes reviews using trained TF IDF vectorizer   
     
    """ 
    return vectorizer.transform(reviews_text)  
     

def train_classifiers(train_water, words, tags=TAGS):
    models = {}
    n_rows = train_water.nrow
    for tag in tags: 
        tag = normalize_column(tag) 
        n = train_water[tag].sum() 
        if n > 0 and n < n_rows:  
            model = H2ORandomForestEstimator(ntrees=100)
            model.train(training_frame = train_water, x = words, y = tag) 
            models.update({ tag: model }) 
    return models 


def classifiers_summary(models):
    for tag, model in models.items(): 
        print("====================\n")
        print(tag) 
        print(model.summary()) 


def classify_reviews(models, reviews_water): 
    """ 
    Classifies reviews using corresponding models  
    """ 
    predictions = {} 
    for tag, model in models.items(): 
        prediction = model.predict(reviews_water) 
        predictions.update({ tag: prediction })  
    return predictions 


def normalize_column(feature): 
    return re.sub('[^a-zA-Z0-9_.]', '', feature.strip().replace(" ", "_").replace("/", "_").replace("-", "_")) 


def normalize_columns(features): 
    if type(features) is not list: 
        features = [features] 
    return [re.sub('[^a-zA-Z0-9_.]', '', feature.strip().replace(" ", "_").replace("/", "_").replace("-", "_")) for feature in features]

    
def split_records(records, ratio): 
    train = [] 
    valid = []
    for record in records: 
        if random.random() <= ratio: 
            train.append(record) 
        else: 
            valid.append(record)    
    return train, valid          

    
def combine_predictions(positive_predictions, negative_predictions, hotels, tags, positive_reviews, negative_reviews): 
    """ 
    Combines predictions for all tags 
    Returns prediction for each tag for each review 
    """ 
    combined_predictions = pd.DataFrame() 
    combined_predictions['len_positive'] = [len(review.strip()) > 0 for review in positive_reviews] 
    combined_predictions['len_negative'] = [len(review.strip()) > 0 for review in negative_reviews] 
    for tag in TAGS: 
        if tag in positive_predictions and tag in negative_predictions: 
            combined_predictions['positive ' + tag] = (positive_predictions[tag].as_data_frame()['p1'] * combined_predictions['len_positive'] > negative_predictions[tag].as_data_frame()['p1'] * combined_predictions['len_negative']) * tags[tag] 
            combined_predictions['negative ' + tag] = (positive_predictions[tag].as_data_frame()['p1'] * combined_predictions['len_positive'] < negative_predictions[tag].as_data_frame()['p1'] * combined_predictions['len_negative']) * tags[tag] 
        else: 
            combined_predictions['positive ' + tag] = 0 
            combined_predictions['negative ' + tag] = 0   
        combined_predictions[tag] = tags[tag] 
    if positive_reviews is not None: 
        combined_predictions['positive'] = positive_reviews  
    if negative_reviews is not None: 
        combined_predictions['negative'] = negative_reviews  
    combined_predictions['hotel_id'] = hotels['hotel_id'] 
    return combined_predictions 


def predict_tag(combined_predictions): 
    """ 
    Scores each tag for each hotel 
    """ 
    scores = combined_predictions.groupby('hotel_id').sum() 
    for tag in TAGS: 
        scores['score ' + tag] = (1 + (scores['positive ' + tag] - scores['negative ' + tag]) / (scores['positive ' + tag] + scores['negative ' + tag])) * 0.5  
        #scores['score ' + tag] = scores['positive ' + tag] / scores[tag]  
    scores = scores[['score ' + tag for tag in TAGS]] 
    scores.fillna(0.5, inplace = True) #Assign score to tags which are not rated in a hotel reviews 
    scores.columns = TAGS  
    return scores 


def write_predictions(combined_predictions, combined_predictionsRF, file): 
    """ 
    Writes file containing reviews and their tags attributions 
    """ 
    f = open(file, 'w') 
    for i in range(combined_predictions.shape[0]): 
        review = combined_predictions.iloc[i] 
        reviewRF = combined_predictionsRF.iloc[i] 
        f.write('Hotel ')
        f.write(review['hotel_id']) 
        f.write('\n')
        f.write('Positive: ') 
        f.write(review['positive'])
        f.write('\n') 
        f.write('Negative: ') 
        f.write(review['negative']) 
        f.write('\n') 
        f.write('Tags NB: ') 
        for tag in TAGS: 
            if tag in combined_predictions.columns and review[tag] == 1: 
                if review['positive ' + tag] == 1: 
                    f.write(tag + '+ ') 
                elif review['negative ' + tag] == 1:  
                    f.write(tag + '- ') 
                else: 
                    f.write(tag + '  ')
        f.write('\n') 
        f.write('Tags RF: ')
        for tag in TAGS: 
            if tag in combined_predictions.columns and review[tag] == 1:
                if review['positive ' + tag] == 1: 
                    f.write(tag + '+ ') 
                elif review['negative ' + tag] == 1:  
                    f.write(tag + '- ') 
                else: 
                    f.write(tag + '  ')
        f.write('\n') 
        f.write('\n') 
    f.close() 


def write_scores(scores, file): 
    """ 
    Writes scores for hotels to file 
    """ 
    scores.to_csv(file)  
