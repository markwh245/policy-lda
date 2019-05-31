"""
Phase 1 - Build dictory for Privacy
"""

import pandas as pd
import PyPDF2
import textract
import nltk
# nltk.download('punkt')  
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import glob   
import requests
from bs4 import BeautifulSoup

url = ['https://www.nab.com.au/common/privacy-policy',
       'https://www.westpac.com.au/privacy/privacy-policy',
       'https://www.bhp.com/info/privacy-policy',
       'https://www.apple.com/au/legal/privacy/en-ww/',
       'https://www.samsung.com/us/account/privacy-policy/',
       'https://www.nab.com.au/common/privacy-policy/gdpr'
       ]

text = ""
for x in url:
    req = requests.get(x)
    soup = BeautifulSoup(req.content, 'html.parser')
    text += soup.get_text("|", strip=True)

pdfs = []
from pathlib import Path
for filename in Path('privacy').glob('**/*.pdf'): 
       pdfFileObj = open(filename,'rb')
       pdfs.append(str(filename))
       pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
       num_pages = pdfReader.numPages
       count = 0
       while count < num_pages:
           pageObj = pdfReader.getPage(count)
           count +=1
           text += pageObj.extractText()
           if text != "":
               text = text
           else:
               text = textract.process(fileurl, method='tesseract', language='eng')


#The word_tokenize() function will break our text phrases into #individual words
tokens = word_tokenize(text)
punctuations = ['(',')',';',':','[',']',',']
stop_words = stopwords.words('english')
extra = ['function', 'group', 'var', 'null', 'void', 'return', 'length', 'typeof', 'else', 'type',
         'auguments', 'get', 'string', 'prototype', 'nodeType', 'slice', 'header', 'top', 'li',
         'style', 'Appendix','Table', 'owl', 'hover', 'pageination']
for i in extra:
    stop_words.append(i)
    

keywords = [word for word in tokens if not word in stop_words and not word in punctuations]


# Libraries for text preprocessing
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


#nltk.download('wordnet') 
from nltk.stem.wordnet import WordNetLemmatizer

##Convert to list from string
text = text.split()
    
    
#Lemmatisation
lem = WordNetLemmatizer()
text = [lem.lemmatize(word) for word in text if not word in stop_words] 
text = " ".join(text)

corpus = []
corpus.append(text)

#Word cloud
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stop_words,
                          max_words=100,
                          max_font_size=50, 
                          random_state=42
                         ).generate(str(corpus))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("privacy.png", dpi = 900)



from sklearn.feature_extraction.text import CountVectorizer
import re
cv = CountVectorizer(stop_words = stop_words, max_features=10000, ngram_range=(1,3))
X = cv.fit_transform(corpus)
list(cv.vocabulary_.keys())[:10]


#Most frequently occuring words
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    return words_freq[:n]
#Convert most freq words to dataframe for plotting bar plot
top_words = get_top_n_words(corpus, n=20)
top_df = pd.DataFrame(top_words)
top_df.columns=["Word", "Freq"]
#Barplot of most freq words
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
g = sns.barplot(x="Word", y="Freq", data=top_df)
g.set_xticklabels(g.get_xticklabels(), rotation=30)

#Most frequently occuring Bi-grams
def get_top_n2_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(2,2),  
            max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]
top2_words = get_top_n2_words(corpus, n=20)
top2_df = pd.DataFrame(top2_words)
top2_df.columns=["Bi-gram", "Freq"]
print(top2_df)
#Barplot of most freq Bi-grams
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
h=sns.barplot(x="Bi-gram", y="Freq", data=top2_df)
h.set_xticklabels(h.get_xticklabels(), rotation=45)

#Most frequently occuring Tri-grams
def get_top_n3_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(3,3), 
           max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]
top3_words = get_top_n3_words(corpus, n=20)
top3_df = pd.DataFrame(top3_words)
top3_df.columns=["Tri-gram", "Freq"]
print(top3_df)
#Barplot of most freq Tri-grams
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
j=sns.barplot(x="Tri-gram", y="Freq", data=top3_df)
j.set_xticklabels(j.get_xticklabels(), rotation=45)