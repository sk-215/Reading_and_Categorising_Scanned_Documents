import os
from PIL import Image
import base64
import http.client, urllib.request, urllib.parse, urllib.error, base64
import io
import json
import requests
import urllib
from gensim import corpora, models, similarities
from gensim.utils import tokenize
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string
import http
import shutil
import tqdm
from spellchecker import SpellChecker

def filter_for_english(text):
    dict_url = 'https://raw.githubusercontent.com/first20hours/' \
               'google-10000-english/master/20k.txt'

    dict_words = set(requests.get(dict_url).text.splitlines())

    english_words = tokenize(text)
    english_words = [w for w in english_words if w in list(dict_words)]
    english_words = [w for w in english_words if (len(w) > 1 or w.lower() == 'i')]
    return (' '.join(english_words))
def preprocess(document):
    clean = filter_for_english(document)
    clean = remove_stopwords(clean)
    clean = preprocess_string(clean)

    # Remove non-english words

    return (clean)

def read_and_return(foldername, fileext='.txt', delete_after_read=False):
    allfiles = os.listdir(foldername)
    allfiles = [os.path.join(foldername, f) for f in allfiles if f.endswith(fileext)]
    alltext = []
    for filename in allfiles:
        with open(filename, 'r', encoding='utf-8') as f:  # Specify encoding here
            alltext.append((filename, f.read()))
        if delete_after_read:
            os.remove(filename)
    return alltext



def run_lda(textlist,
            num_topics=10,
            return_model=False,
            preprocess_docs=True):
    '''
    Train and return an LDA model against a list of documents
    '''
    if preprocess_docs:
        doc_text = [preprocess(d) for d in textlist]
    dictionary = corpora.Dictionary(doc_text)

    corpus = [dictionary.doc2bow(text) for text in doc_text]
    tfidf = models.tfidfmodel.TfidfModel(corpus)
    transformed_tfidf = tfidf[corpus]

    lda = models.ldamulticore.LdaMulticore(transformed_tfidf, num_topics=num_topics, id2word=dictionary)

    input_doc_topics = lda.get_document_topics(corpus)

    return (lda, dictionary)


def find_topic(text, dictionary, lda):
    '''
    https://stackoverflow.com/questions/16262016/how-to-predict-the-topic-of-a-new-query-using-a-trained-lda-model-using-gensim

     For each query ( document in the test file) , tokenize the
     query, create a feature vector just like how it was done while training
     and create text_corpus
    '''

    text_corpus = []

    for query in text:
        temp_doc = tokenize(query.strip())
        current_doc = []
        temp_doc = list(temp_doc)
        for word in range(len(temp_doc)):
            current_doc.append(temp_doc[word])

        text_corpus.append(current_doc)
    '''
     For each feature vector text, lda[doc_bow] gives the topic
     distribution, which can be sorted in descending order to print the 
     very first topic
    '''
    tops = []
    for text in text_corpus:
        doc_bow = dictionary.doc2bow(text)
        topics = sorted(lda[doc_bow], key=lambda x: x[1], reverse=True)[0]
        tops.append(topics)
    return (tops)


def topic_label(ldamodel, topicnum):
    alltopics = ldamodel.show_topics(formatted=False)
    topic = alltopics[topicnum]
    topic = dict(topic[1])
    import operator
    return (max(topic, key=lambda key: topic[key]))

TOPICS = 2
OUTPUT_FOLDER = r'D:\for_lda'

if __name__ == '__main__':

    # First, read all the output .txt files
    print("Reading files")
    texts = read_and_return(OUTPUT_FOLDER)

    print("Building LDA topic model")
    # Second, train the LDA model (pre-processing is internally done)
    print("Preprocessing Text")
    textlist = [t[1] for t in texts]
    ldamodel, dictionary = run_lda(textlist, num_topics=TOPICS)

    # Third, extract the top topic for each document
    print("Extracting Topics")
    topics = []
    for t in texts:
        topics.append((t[0], find_topic([t[1]], dictionary, ldamodel)))

    # Convert topics to topic names
    for i in range(len(topics)):
        topnum = topics[i][1][0][0]
        # print(topnum)
        topics[i][1][0] = topic_label(ldamodel, topnum)
        # [(filename, topic), ..., (filename, topic)]

    # Create folders for the topics
    print("Copying Documents into Topic Folders")
    foundtopics = []
    for t in topics:
        foundtopics += t[1]
    foundtopics = set(foundtopics)
    topicfolders = [os.path.join(OUTPUT_FOLDER, f) for f in foundtopics]
    topicfolders = set(topicfolders)
    [os.makedirs(m) for m in topicfolders]

    # Copy files into appropriate topic folders
    for t in topics:
        filename, topic = t
        src = filename
        filename = filename.split("\\")
        dest = os.path.join(OUTPUT_FOLDER, topic[0])
        dest = dest + "/" + filename[-1]
        copystr = "copy " + src + " " + dest
        shutil.copyfile(src, dest)
        os.remove(src)

    print("Done")