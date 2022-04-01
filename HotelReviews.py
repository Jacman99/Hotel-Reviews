'''
======PATTERN MINING ON HOTEL REVIEW DATA======

Name: Jason Combs
Student ID: 301352433
Student Email: jcombs@sfu.ca
Class: CMPT459 - Introduction to Data Mining
Professor: Dr. Jian Pei
'''

import pandas as pd
import re
import collections
import nltk
from nltk.corpus import stopwords

# Package for Frequent Pattern Mining
import pyfpgrowth as fpg

# Packages useful for WordCloud
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# Function Definitions for Various Tasks
def remove_url_punctuation(X):
    # Replace URLs, punctuations, hastags found in a text 
    # string with nothing. Change to lowercase.
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    replace_url = url_pattern.sub(r'', str(X))
    punct_pattern = re.compile(r'[^\w\s]')
    no_punct = punct_pattern.sub(r'', replace_url).lower()
    return no_punct

def split_words(X):
    # Split tweets into word for NLP
    split_word_list = X.split(" ")
    return split_word_list

def remove_stopwords(X):
    # Remove Stop Words and other conditions
    global stop_words
    words = []
    for word in X:
        if word not in stop_words and len(word) > 2 and word != 'nan':
            words.append(word)
    return words

def detect_language(X):
    # Remove other languages using langdetect
    from langdetect import detect
    try:
        lang = detect(X)
        return(lang)
    except:
        return("other")

def reset_indices(df1, df2, df3):
    # Reset indexes of three dataframes
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    df3 = df3.reset_index(drop=True)
    return df1, df2, df3

def create_data(df):
    # Takes in data from dataframe converted from a dictionary, removes the ' 
    # strings in index, and returns the word patterns and their frequencies as 
    # index and support respectively
    index = []
    support = []
    c = len(df)
    for i in range(c):
        index.append(str(df.loc[i, 'Index']))
        support.append(df.loc[i, 'Support'])
        index[i] = index[i].replace("'", "")
        index[i] = index[i].replace(",)", ")")
    return index, support

def generate_wordcloud(words, mask):
    # Creates a WordCloud using a dictionary (words), and a PNG image (mask)
    word_cloud = WordCloud(width = 512, height = 512, background_color='white',\
        stopwords=STOPWORDS, mask=mask).generate_from_frequencies(words)
    plt.figure(figsize=(10,8), facecolor='white', edgecolor='blue')
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()

def generate_two_wordclouds(words1, words2, mask):
    # Creates a double WordCloud using two dictionarys (words1, words2), and a PNG image (mask)
    word_cloud = WordCloud(width = 300, height = 500, background_color='red',\
        stopwords=STOPWORDS, mask=mask).generate_from_frequencies(words1)
    word_cloud2 = WordCloud(width = 300, height = 500, background_color='lightgreen',\
        stopwords=STOPWORDS, mask=mask).generate_from_frequencies(words2)
    plt.figure(figsize=(18,18), facecolor='white', edgecolor='blue')
    gs = gridspec.GridSpec(1, 2)
    gs.update(wspace=0, hspace=0)
    plt.subplot(gs[0])        
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.subplot(gs[1])
    plt.imshow(word_cloud2)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()

#   Dataset D1
dfD1A = pd.read_csv('US_HotelReviews.csv')
dfD1B = pd.read_csv('US_HotelReviews_Jun19.csv')
dfD1B = dfD1B.drop(columns=['reviews.dateAdded']) # removing column P: 'reviews.dateAdded' from 'US_HotelReviews_Jun19.csv' since it appears to be a null column in the dataset
#   Dataset D2
dfD2 = pd.read_csv('EU_HotelReviews.csv')

# Data cleaning
#   Dataset D1 (USA 20K rows)
dfD1 = pd.concat([dfD1A, dfD1B], ignore_index=True) # combine dataframes
dfD1 = dfD1.drop_duplicates() # remove copies of rows
dfD1 = dfD1.drop(columns=['id', 'dateAdded', 'dateUpdated', 'address', 'categories', 'primaryCategories', 'keys', 'postalCode', 'reviews.dateSeen', 'reviews.sourceURLs',\
    'reviews.title', 'reviews.userCity', 'reviews.userProvince', 'reviews.username', 'sourceURLs', 'websites'])
old_names = ['reviews.date', 'reviews.rating', 'reviews.text'] 
new_names = ['date', 'rating', 'text']
dfD1.rename(columns=dict(zip(old_names, new_names)), inplace=True)
#   Dataset D2 (Italy 37K rows)
dfD2 = dfD2[dfD2['Hotel_Address'].str.contains('Italy')] # only take rows of hotels in Italy
dfD2 = dfD2.drop_duplicates() # remove copies of rows
dfD2 = dfD2.drop(columns = ['Additional_Number_of_Scoring', 'Reviewer_Nationality', 'Review_Total_Negative_Word_Counts',\
    'Total_Number_of_Reviews', 'Review_Total_Positive_Word_Counts', 'Average_Score', 'Tags', 'Total_Number_of_Reviews_Reviewer_Has_Given', 'days_since_review'])
old_names2 = ['Hotel_Address', 'Review_Date', 'Hotel_Name', 'Negative_Review', 'Positive_Review', 'Reviewer_Score', 'lat', 'lng'] 
new_names2 = ['address', 'date', 'name', 'negative_text', 'positive_text', 'rating', 'latitude', 'longitude']    
dfD2.rename(columns=dict(zip(old_names2, new_names2)), inplace=True)

# Removing any rows containing null values in the associated columns for all columns
#   Dataset D1
dfD1 = dfD1[pd.notnull(dfD1['city'])]
dfD1 = dfD1[pd.notnull(dfD1['country'])]
dfD1 = dfD1[pd.notnull(dfD1['latitude'])]
dfD1 = dfD1[pd.notnull(dfD1['longitude'])]
dfD1 = dfD1[pd.notnull(dfD1['name'])]
dfD1 = dfD1[pd.notnull(dfD1['province'])]
dfD1 = dfD1[pd.notnull(dfD1['rating'])]
dfD1 = dfD1[pd.notnull(dfD1['date'])]
dfD1 = dfD1[pd.notnull(dfD1['text'])]
#   Dataset D2
dfD2 = dfD2[pd.notnull(dfD2['address'])]
dfD2 = dfD2[pd.notnull(dfD2['date'])]
dfD2 = dfD2[pd.notnull(dfD2['name'])]
dfD2 = dfD2[pd.notnull(dfD2['negative_text'])]
dfD2 = dfD2[pd.notnull(dfD2['positive_text'])]
dfD2 = dfD2[pd.notnull(dfD2['rating'])]
dfD2 = dfD2[pd.notnull(dfD2['latitude'])]
dfD2 = dfD2[pd.notnull(dfD2['longitude'])]

# Doubling the rating of US to have ratings out of 10 in D1. This is to be consistent with Italy's data
dfD1['rating'] = dfD1['rating'] * 2

# Removing any rows with a rating greater than 10 or less than 0
#   Dataset D1
dfD1 = dfD1[dfD1['rating']>0]
dfD1 = dfD1[dfD1['rating']<=10]
#   Dataset D2
dfD2 = dfD2[dfD2['rating']>0]
dfD2 = dfD2[dfD2['rating']<=10]

# Filtering to only include data from the US via latitude and longitude
#   Dataset D1
dfD1 = dfD1[((dfD1['latitude']<=50.0) & (dfD1['latitude']>=24.0)) & ((dfD1['longitude']<=-65.0) & (dfD1['longitude']>=-122.0))]
#   Dataset D2
dfD2 = dfD2[((dfD2['latitude']<=46.0) & (dfD2['latitude']>=44.0)) & ((dfD2['longitude']<=10.0) & (dfD2['longitude']>=8.0))]
#   Resetting indices since we may have removed some rows
dfD1 = dfD1.reset_index(drop=True)
dfD2 = dfD2.reset_index(drop=True)

# Add 'bad0_good1' column
#   Dataset D1
dfD1["bad0_good1"] = dfD1["rating"].apply(lambda x: 0 if x <= 5 else 1)

lenD1 = len(dfD1)
num_pos_D1 = dfD1['bad0_good1'].sum()
print("Note: There are", lenD1 - num_pos_D1, "negative reviews and", num_pos_D1, "positive reviews out of", lenD1, "reviews in D1.")
print("We balance this by selecting the same amount of positive and negative reviews to mine on.")

# To speed up tokenization as well as balancing the negative and positive reviews in our data,
# we reduce the maximum number of reviews. Next steps of data cleaning may remove some reviews.
N1 = 2000 # maximum reviews availabe in D1: 19790
N2 = 5000 # maximum reviews availabe in D2: 37204
neg_dfD1 = dfD1.loc[dfD1['bad0_good1'] == 0].head(N1) # selecting first n1 negative reviews in D1
pos_dfD1 = dfD1.loc[dfD1['bad0_good1'] == 1].head(N1) # selecting first n1 positive reviews in D1
dfD2 = dfD2.head(N2) # selecting first n2 reviews in D2 (already split into positive/negative reviews)

# Tokenizing datasets D1 and D2
print(">> Beginning tokenization of D1 and D2...")
#   Remove punctuation, special characters etc in tweets
#       Dataset D1
neg_dfD1['tidy_tweet'] = neg_dfD1['text'].apply(remove_url_punctuation)
pos_dfD1['tidy_tweet'] = pos_dfD1['text'].apply(remove_url_punctuation)
#       Dataset D2
dfD2['tidy_tweet'] = dfD2['negative_text'].apply(remove_url_punctuation)
dfD2['tidy_tweet2'] = dfD2['positive_text'].apply(remove_url_punctuation)
print(">> Step 1/4 completed...")

#   Keep english tweets
#       Dataset D1
neg_dfD1['en'] = neg_dfD1['text'].apply(detect_language)
neg_dfD1 = neg_dfD1[neg_dfD1['en'] == 'en']
pos_dfD1['en'] = pos_dfD1['text'].apply(detect_language)
pos_dfD1 = pos_dfD1[pos_dfD1['en'] == 'en']
#       Dataset D2
dfD2['en'] = dfD2['negative_text'].apply(detect_language)
dfD2 = dfD2[dfD2['en'] == 'en']
dfD2['en'] = dfD2['positive_text'].apply(detect_language)
dfD2 = dfD2[dfD2['en'] == 'en']
print(">> Step 2/4 completed...")

#   Tokenize words in tweets
#       Dataset D1
neg_dfD1['word_list'] = neg_dfD1['tidy_tweet'].apply(split_words)
neg_dfD1 = neg_dfD1.drop(columns=['tidy_tweet', 'en'])
pos_dfD1['word_list'] = pos_dfD1['tidy_tweet'].apply(split_words)
pos_dfD1 = pos_dfD1.drop(columns=['tidy_tweet', 'en'])
#       Dataset D2
dfD2['neg_word_list'] = dfD2['tidy_tweet'].apply(split_words)
dfD2['pos_word_list'] = dfD2['tidy_tweet2'].apply(split_words)
dfD2 = dfD2.drop(columns=['tidy_tweet', 'tidy_tweet2', 'en'])
print(">> Step 3/4 completed...")

#   Removing stopwords
stop_words = set(stopwords.words('english'))
#       Dataset D1
neg_dfD1['word_list'] = neg_dfD1['word_list'].apply(remove_stopwords)
pos_dfD1['word_list'] = pos_dfD1['word_list'].apply(remove_stopwords)
#       Dataset D2
dfD2['neg_word_list'] = dfD2['neg_word_list'].apply(remove_stopwords)
dfD2['pos_word_list'] = dfD2['pos_word_list'].apply(remove_stopwords)

#   Resetting indices since some rows may have been removed
neg_dfD1, pos_dfD1, dfD2 = reset_indices(neg_dfD1, pos_dfD1, dfD2)
print(">> Step 4/4 completed...")

#   Removing rows in D2 containing both "positive" and "negative" in the pos_word_list and neg_word_list respectively
#   since these rows do not contain a review, just the default message for someone who did not leave a review.
lengthD2 = len(dfD2)
for i in range(lengthD2):
    if ('positive' in dfD2.loc[i, 'pos_word_list']) & (len(dfD2.loc[i, 'pos_word_list']) == 1) \
        & ('negative' in dfD2.loc[i, 'neg_word_list']) & (len(dfD2.loc[i, 'neg_word_list']) == 1):
        dfD2.drop([i])

#   Resetting indices since we may have removed some rows
neg_dfD1, pos_dfD1, dfD2 = reset_indices(neg_dfD1, pos_dfD1, dfD2)

#   Replace the 'positive' and 'negative' strings with the empty strings. 
for i in range(lengthD2):
    if ('negative' in dfD2.loc[i, 'neg_word_list']) & (len(dfD2.loc[i, 'neg_word_list']) == 1):
        dfD2.loc[i, 'neg_word_list'][0] = dfD2.loc[i, 'neg_word_list'][0].replace('negative', '')
    if ('positive' in dfD2.loc[i, 'pos_word_list']) & (len(dfD2.loc[i, 'pos_word_list']) == 1):
        dfD2.loc[i, 'pos_word_list'][0] = dfD2.loc[i, 'pos_word_list'][0].replace('positive', '')
print(">> Tokenization complete!\n")

print("Printing list of positive reviews from USA (D1)\n", neg_dfD1)
print("\nPrinting list of negative reviews from USA (D1)\n", pos_dfD1)
print("\nPrinting list of reviews from Italy (D2)\n", dfD2)

# Selecting the number of reviews to use for frequent pattern mining. This ensures the negative and positive
# reviews in D1 will be balanced since no more rows will be removed from this point on.
n1 = min(len(neg_dfD1), len(pos_dfD1)) # Freq pattern mine on n1 positive and negative reviews from D1
n2 = len(dfD2) # Freq pattern mine on n2 reviews from D2

print("\nNote: We now have", n1, "negative and positive reviews in D1 and", n2, "reviews in D2 after some data cleaning and balancing.\n")

# Creating list for Frequent Pattern Mining
#   Dataset D1
#       Negative reviews
neg_dfD1 = neg_dfD1.head(n1) # selecting first n1 negative reviews
neg_word_listD1 = list(neg_dfD1['word_list'])
neg_wlD1 = len(neg_word_listD1)
for i in range(neg_wlD1):
    neg_word_listD1[i] = list(set(neg_word_listD1[i])) # removing duplicate strings
#       Positive reviews
pos_dfD1 = pos_dfD1.head(n1) # selecting first n1 positive reviews
pos_word_listD1 = list(pos_dfD1['word_list'])
pos_wlD1 = len(pos_word_listD1)
for i in range(pos_wlD1):
    pos_word_listD1[i] = list(set(pos_word_listD1[i])) # removing duplicate strings

#   Dataset D2
#       Negative reviews
dfD2 = dfD2.head(n2) # selecting first n2 reviews
neg_word_listD2 = list(dfD2['neg_word_list'])
neg_wlD1 = len(neg_word_listD2)
for i in range(neg_wlD1):
    neg_word_listD2[i] = list(set(neg_word_listD2[i])) # removing duplicate strings
#       Positive reviews
pos_word_listD2 = list(dfD2['pos_word_list'])
pos_wlD1 = len(pos_word_listD2)
for i in range(pos_wlD1):
    pos_word_listD2[i] = list(set(pos_word_listD2[i])) # removing duplicate strings

# Minimum supports for Frequent Pattern Mining
min_sup = 50 # minimum support for D1
min_sup2 = 30 # minimum support for D2

# Creating frequent patterns
#   Dataset D1
#       Negative reviews
neg_patternsD1 = fpg.find_frequent_patterns(neg_word_listD1, min_sup)
neg_patternsD1 = pd.DataFrame.from_dict(neg_patternsD1, orient='index')
neg_patternsD1 = neg_patternsD1.reset_index() # changing index into column
neg_patternsD1 = neg_patternsD1.rename(columns={'index':'Index', 0:'Support'}) # renaming columns
neg_patternsD1 = neg_patternsD1.sort_values('Support', ascending=False)
neg_patternsD1 = neg_patternsD1.reset_index(drop=True) # reset index
print("D1 patterns in", len(neg_word_listD1), "negative reviews:\n", neg_patternsD1, "\n")
#       Positive reviews
pos_patternsD1 = fpg.find_frequent_patterns(pos_word_listD1, min_sup)
pos_patternsD1 = pd.DataFrame.from_dict(pos_patternsD1, orient='index')
pos_patternsD1 = pos_patternsD1.reset_index() # changing index into column
pos_patternsD1 = pos_patternsD1.rename(columns={'index':'Index', 0:'Support'}) # renaming columns
pos_patternsD1 = pos_patternsD1.sort_values('Support', ascending=False)
pos_patternsD1 = pos_patternsD1.reset_index(drop=True) # reset index
print("D1 patterns in", len(pos_word_listD1), "positive reviews:\n", pos_patternsD1, "\n")

#   Dataset D2
#       Negative reviews
neg_patternsD2 = fpg.find_frequent_patterns(neg_word_listD2, min_sup2)
neg_patternsD2 = pd.DataFrame.from_dict(neg_patternsD2, orient='index')
neg_patternsD2 = neg_patternsD2.reset_index() # changing index into column
neg_patternsD2 = neg_patternsD2.rename(columns={'index':'Index', 0:'Support'}) # renaming columns
neg_patternsD2 = neg_patternsD2.sort_values('Support', ascending=False)
neg_patternsD2 = neg_patternsD2.reset_index(drop=True) # reset index
neg_patternsD2 = neg_patternsD2.drop([0])
neg_patternsD2 = neg_patternsD2.reset_index(drop=True) # reset index
print("D2 patterns in", len(neg_word_listD2), "negative reviews:\n", neg_patternsD2, "\n")
#       Positive reviews
pos_patternsD2 = fpg.find_frequent_patterns(pos_word_listD2, min_sup2)
pos_patternsD2 = pd.DataFrame.from_dict(pos_patternsD2, orient='index')
pos_patternsD2 = pos_patternsD2.reset_index() # changing index into column
pos_patternsD2 = pos_patternsD2.rename(columns={'index':'Index', 0:'Support'}) # renaming columns
pos_patternsD2 = pos_patternsD2.sort_values('Support', ascending=False)
pos_patternsD2 = pos_patternsD2.reset_index(drop=True) # reset index
print("D2 patterns in", len(pos_word_listD2), "positive reviews:\n", pos_patternsD2, "\n")

# Data visualization: WordCloud
hotel_mask = np.array(Image.open("hotel2.png"))

#   Dataset D1
#       Negative reviews
WordCloud_neg_D1 = neg_patternsD1.head(100)
index_neg_D1, support_neg_D1 = create_data(WordCloud_neg_D1)
dict_neg_D1 = dict(zip(index_neg_D1, support_neg_D1))
#       Positive reviews
WordCloud_pos_D1 = pos_patternsD1.head(100)
index_pos_D1, support_pos_D1 = create_data(WordCloud_pos_D1)
dict_pos_D1 = dict(zip(index_pos_D1, support_pos_D1))

#   Creating WordCloud of negative vs positive reviews in D1
title_text = "Most frequent patterns of negative reviews (left in red) vs positive reviews (right in green) in USA (D1)"
print("Creating WordCloud for:\n  ", title_text)
generate_two_wordclouds(dict_neg_D1, dict_pos_D1, hotel_mask)

#   Dataset D1
#       Negative reviews
WordCloud_neg_D2 = neg_patternsD2.head(100)
index_neg_D2, support_neg_D2 = create_data(WordCloud_neg_D2)
dict_neg_D2 = dict(zip(index_neg_D2, support_neg_D2))
#       Positive reviews
WordCloud_pos_D2 = pos_patternsD2.head(100)
index_pos_D2, support_pos_D2 = create_data(WordCloud_pos_D2)
dict_pos_D2 = dict(zip(index_pos_D2, support_pos_D2))

#   Creating WordCloud of negative vs positive reviews in D2
title_text = "Most frequent patterns of negative reviews (left in red) vs positive reviews (right in green) in Italy (D2)"
print("Creating WordCloud for:\n  ", title_text)
generate_two_wordclouds(dict_neg_D1, dict_pos_D1, hotel_mask)

# Data visualization: Bar Charts
b = 10 # Use the first b most frequent patterns in bar chart
#   Dataset D1
#       Negative reviews
title_text = "Bar Chart of Top 10 Most Frequent Word Patterns of Negative Hotel Reviews in USA"
print("Creating Bar Chart for:\n   Most frequent patterns of negative reviews in USA (D1)")
plt.bar(index_neg_D1[:b], support_neg_D1[:b], width=0.8, color='r', label="Im a wiener")
plt.xticks(rotation=30)
plt.title(title_text)
plt.xlabel('Patterns')
plt.ylabel('Frequency')
plt.show()
#       Positive reviews
title_text = "Bar Chart of Top 10 Most Frequent Word Patterns of Positive Hotel Reviews in USA"
print("Creating Bar Chart for:\n   Most frequent patterns of positive reviews in USA (D1)")
plt.bar(index_pos_D1[:b], support_pos_D1[:b], width=0.8, color='g')
plt.xticks(rotation=30)
plt.title(title_text)
plt.xlabel('Patterns')
plt.ylabel('Frequency')
plt.show()
#   Dataset D2
#       Negative reviews
title_text = "Bar Chart of Top 10 Most Frequent Word Patterns of Negative Hotel Reviews in Italy"
print("Creating Bar Chart for:\n   Most frequent patterns of negative reviews in Italy (D2)")
plt.bar(index_neg_D2[:b], support_neg_D2[:b], width=0.8, color='r')
plt.xticks(rotation=30)
plt.title(title_text)
plt.xlabel('Patterns')
plt.ylabel('Frequency')
plt.show()
#       Positive reviews
title_text = "Bar Chart of Top 10 Most Frequent Word Patterns of Positive Hotel Reviews in Italy"
print("Creating Bar Chart for:\n   Most frequent patterns of positive reviews in Italy (D2)")
plt.bar(index_pos_D2[:b], support_pos_D2[:b], width=0.8, color='g')
plt.xticks(rotation=30)
plt.title(title_text)
plt.xlabel('Patterns')
plt.ylabel('Frequency')
plt.show()

# Saving data to current directory
neg_patternsD1.to_csv('neg_patternsD1') # Save dataframe
pos_patternsD1.to_csv('pos_patternsD1') # Save dataframe
neg_patternsD2.to_csv('neg_patternsD2') # Save dataframe
pos_patternsD2.to_csv('pos_patternsD2') # Save dataframe

print("\n>> Done! Program exited successfully.")