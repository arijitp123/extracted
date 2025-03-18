# Importing the required Libraries
import pandas as pd
import re
import nltk 
import spacy
import string
import pickle as pk

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

# Now, loading the pickle files 

# Loading the Count Vectorizer
count_vector = pk.load(open('pickle_file/count_vector.pkl','rb'))            

# Loading the TFIDF Transformer
tfidf_transformer = pk.load(open('pickle_file/tfidf_transformer.pkl','rb')) 

# Loading the Classification Model
model = pk.load(open('pickle_file/model.pkl','rb'))         

# Loading the Cleaned DataFrame 
product_df = pk.load(open('pickle_file/lemmatized_review_df.pkl','rb'))    


#This function is predicting the sentiment of the product review comments
def model_predict(text):
    word_vector = count_vector.transform(text)
    tfidf_vector = tfidf_transformer.transform(word_vector)
    output = model.predict(tfidf_vector)
    return output


#This function will recommend the top 20 products based on the sentiment from model
def recommend_products(user_name):
    recommend_matrix = pk.load(open('pickle_file/user_final_rating.pkl','rb'))
    product_list = pd.DataFrame(recommend_matrix.loc[user_name].sort_values(ascending=False)[0:20])
    product_frame = product_df[product_df['name'].isin(product_list.index.tolist())]
    output_df = product_frame[['name','lemmatized_text']] 
    output_df['predicted_sentiment'] = model_predict(output_df['lemmatized_text'])
    return output_df

#This function will recommend the top 5 products based on the sentiment from model
def reco_prod_5(df):
    total_product=df.groupby(['name']).agg('count')
    rec_df = df.groupby(['name','predicted_sentiment']).agg('count')
    rec_df=rec_df.reset_index()
    merge_df = pd.merge(rec_df,total_product['lemmatized_text'],on='name')
    merge_df['%percentage'] = (merge_df['lemmatized_text_x']/merge_df['lemmatized_text_y'])*100
    merge_df=merge_df.sort_values(ascending=False,by='%percentage')
    output_products = pd.DataFrame(merge_df['name'][merge_df['predicted_sentiment'] ==  1][:5])
    return output_products

