# Importing the required libraries and packages
from selenium import webdriver
from time import sleep
from bs4 import BeautifulSoup
import requests
import unicodedata
import re
import string
import pandas as pd
import fastparquet as fp
import sudachipy
from sudachipy import dictionary
from sudachipy import tokenizer
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from pprint import pprint

''' 1. Use the Selenium's Chromedriver to automatically access all the pages of the Tech blog's website
        and retrieve URLs of all the articles posted in the blog '''

# Using Selenium's Chromedriver to automatically access the company's tech blog and \
# collect the URLs of all the buttons used for navigating to the next page
main_page_url = 'https://www.flywheel.jp/topics-tag/tech/'
all_page_urls = []
article_url_list = []
driver = webdriver.Chrome('./chromedriver')
driver.get(main_page_url)

# Using BeautifulSoup library to parse the html content of the URLs
page_source = BeautifulSoup(driver.page_source, 'html.parser')
all_pages = page_source.find_all('a', class_='page larger')
for page in all_pages:
    all_page_urls.append(page['href'])
    
all_page_urls.append(main_page_url)
print(len(all_page_urls))

# Iterating through all the pages of the blog and retrieveing the URLs of existing articles \
# by manually inspecting the location of the URLs inside the html tags in advance
for url in all_page_urls:
    driver.get(url)
    sleep(5)
    page_source = BeautifulSoup(driver.page_source, 'html.parser')
    article_boxes = page_source.find_all('li', class_ = 'pb-100 pb-sp-70')
    for box in article_boxes:
        for ele in box.find_all('a', class_=False):
            article_url_list.append(ele['href'])
            print(f'The number of article URLs scraped is {len(article_url_list)}')
            
# Making a set to remove the retrived duplicate URLs
article_url_list = list(set(article_url_list)) 
print(len(article_url_list))

# Closing and quiting the driver
driver.close()
driver.quit()

'''________________________________________________________________'''

''' 2. Use the BeautifulSoup to parse the HTML content of the retrieved URLs, 
        and apply 'for loop' and 'regular expressions' to remove not meaningful content of the text '''

# Initializing an empty dictionary to store the URL for each article as a key, and article's preprocessed text as a value
article_dict = {}

for url in article_url_list:
    
    response = requests.get(url)

    soup = BeautifulSoup(response.text, 'html.parser')
    text_only = soup.find_all(string = True)
    
    reduced_text = ''
    blacklist = [
    '[document]',
    'noscript',
    'header',
    'html',
    'meta',
    'head',
    'input',
    'script',
    'style',
    ]

    # remove the unwanted tag elements from the article's text that appear in the blacklist
    for item in text_only:
        if item.parent.name not in blacklist:
            reduced_text += '{}'.format(item)

    match = re.search(r'contents start(.*?)contents end', reduced_text, flags = re.DOTALL)
    if match:
        # match the text to the group between the 'contents start' and 'contents end'
        text = match.group(1)
        # use the NKFC form to normalize Japanese texts
        text = unicodedata.normalize('NFKC', text)
        # remove special characters
        text = re.sub(r'[\t\n\r#。、「」・”─ –]', '', text)
        text = text.strip().lower()
        # remove a string with beginning of 'YYYY.MM.DD' end of 'tech|'
        pattern = re.compile(r'\d{4}\.\d{2}\.\d{2}.*?tech\|')
        text = re.sub(pattern, '', text)
        # remove URLs from the text
        url_pattern = re.compile(r'http[s]?:\/\/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        text = re.sub(url_pattern, '', text)
        # remove the first 3 characters from every text, i.e, 'ブログ'
        text = re.sub(r'.', '', text, count = 3)
        # remove the digits and punctuation marks 
        text = re.sub(r'\d', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        article_dict[url] = text
    else:
        print('No match found here - ', url)
        article_dict[url] = reduced_text

# Saving the 'article_dict' as a pandas dataframe 
article_df = pd.DataFrame(list(article_dict.items()), columns=['article_url', 'article_text'])

# Saving the dataframe as a parquet file for minimising data storage and optimising data query and processing
parquet_file = './article_data.parquet'
fp.write(parquet_file, article_df, compression='GZIP')

jp_stopwords = open('japanese_stopwords.txt', 'r')
jp_stopwords = jp_stopwords.read()
jp_stopwords = jp_stopwords.split('\n')

'''________________________________________________________________'''

''' 3. Conduct a topic modelling to cluster the document text into topics '''

class ArticleTopics:
    def __init__(self, document_text):
        self.document_text = document_text

    def sudachipy_tokenizer(self, document_text):
        ''' 
        Returns word tokens for a given Japanese text
        ---------------------------------------------
        Input: text (type string) 
        Output: string formatted word tokens in a list 
        '''

        # use the SudachiPy morphological analyzer to tokenize a document text written in Japanese
        tokenizer_object = sudachipy.Dictionary().create()
        tokens = tokenizer_object.tokenize(document_text)
        tokens = [token.surface() for token in tokens]

        # remove Japanese stopwords
        tokens = [token for token in tokens if token not in jp_stopwords]

        # remove words written in hiragana 
        kana_re = re.compile('^[ぁ-ゖ]+$')
        tokens = [token for token in tokens if not kana_re.match(token)]
        return tokens

    def get_topics(self):
        '''
        Returns: Topic distribution for a given text
        --------------------------------------------
        Input: document text (type string)
        Output: topics (number must be defined) with their associated words and word probabilities to be associated with that topic
        Output type: list of (integer, float) pairs
        '''
            
        # tokenize the text with a predefined 'sudachipy_tokenizer' function
        tokens = [self.sudachipy_tokenizer(document_text)]

        # create an id to word/token dictionary
        id2word = Dictionary(tokens)

        # create a list of unique words and their frequencies in the document text
        corpus = [id2word.doc2bow(token) for token in tokens]
            
        # build an LDA model with gensim
        lda_model = LdaModel(corpus = corpus,
                        id2word = id2word,
                        num_topics = 3, 
                        random_state = 0, 
                        chunksize = 100,
                        alpha = 'auto',
                        per_word_topics = True)

        for topic_index, topic_words in lda_model.print_topics():
            output_topics = print('Topic: {} \nWords: {}'.format(topic_index, topic_words))

        return output_topics


# Create instances of ArticleTopics class and retrieve the topics and tokens for each article
for document_text in article_df['article_text']:
    class_instance = ArticleTopics(document_text)
    topics = class_instance.get_topics()
    tokens = class_instance.sudachipy_tokenizer(document_text)
    print('Below are the tokens: \n', tokens)
