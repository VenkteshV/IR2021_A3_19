import pandas as pd
import os
import re
import string
import nltk
from nltk.corpus import stopwords,wordnet
from nltk.tokenize import RegexpTokenizer, word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
wnl = WordNetLemmatizer()

stopwords_english = stopwords.words('english')
class_names = ['comp.graphics','sci.med','talk.politics.misc','rec.sport.hockey','sci.space']

re_email = re.compile(r"(?!\B)\S+@[^\s.]+[\.a-zA-Z]+")
re_url = re.compile(r"\b[a-zA-Z]+[^@]\.[\.a-zA-Z]+\b")
data ={'text':[],'label':[]}
def clean_meta(text):
    text = re.sub(r'(From:\s+[^\n]+\n)', '', text)
    text = re.sub(r'(Subject:[^\n]+\n)', '', text)
    text = re.sub(r'(Nntp-Posting-Host:\s+[^\n]+\n)', '', text)
    text = re.sub(r'(([\sA-Za-z0-9\-]+)?[A|a]rchive-name:[^\n]+\n)', '', text)
    text = re.sub(r'(Last-modified:[^\n]+\n)', '', text)
    text = re.sub(r'(Version:[^\n]+\n)', '', text)

    return text

def get_wordnet_tag(word_tag_tuple):
    if word_tag_tuple[1].startswith('J'):
        return wordnet.ADJ
    elif word_tag_tuple[1].startswith('V'):
        return wordnet.VERB
    elif word_tag_tuple[1].startswith('N'):
        return wordnet.NOUN
    elif word_tag_tuple[1].startswith('R'):
        return wordnet.ADV

def word_tokenizer(text):
    regexp_tokenizer=RegexpTokenizer(r'\b[a-zA-Z]+[^@]\.[\.a-zA-Z]+\b|(?!\B)\S+@[^\s.]+[\.a-zA-Z]+|\b[\w\'\-]+\b|\w+')
    return regexp_tokenizer.tokenize(text)
def lemmatizer(sent):
    words = word_tokenizer(sent)
    lemmatized_set = []
    pos_tagged = nltk.pos_tag(words)
    for word in pos_tagged:
        try:
            lemmatized_set.append(wnl.lemmatize(word[0], get_wordnet_tag(word)))
        except:
            lemmatized_set.append(wnl.lemmatize(word[0]))
            continue
    sent = ' '.join(lemmatized_set)
    return sent
def clean_sentence(sentence):
    input_text = sentence.split("Lines:")
    if len(input_text)>1:
        input_text = sentence.split("Lines:")
        split_text = "Lines:"
    else:
        input_text = sentence.split("Date:")
        split_text = "Date:"
    sentence = input_text[1]
    sentence = sentence.lower()
    sentence = sentence.strip()
    sentence = re.sub(re_url, '', sentence)
    sentence = re.sub(re_email, '', sentence)
    sentence = re.sub(f'[{re.escape(string.punctuation)}]', '', sentence)
    sentence = re.sub(r'(\d+)', ' ', sentence)
    sentence = re.sub(r'(\s+)', ' ', sentence)
    sentence = lemmatizer(sentence)
    return sentence

def remove_stopwords(sentence):
    words = sentence.split(" ")
    words = [word for word in words if word not in stopwords_english]
    return ' '.join(words)
def preprocess(text):
    text = clean_meta(text)
    text = clean_sentence(text)
    text = remove_stopwords(text)
    return text


def get_data(class_name):
    file_names = os.listdir("20_newsgroups/"+class_name)
    for i in range(len(file_names)):
        with open("20_newsgroups/"+class_name+"/"+file_names[i],encoding = "latin-1") as f_input:
            text= f_input.read()
            text = preprocess(text)
            data['text'].append(text)
            data['label'].append(class_name)



if __name__=="__main__":
    for class_name in class_names:
        get_data(class_name)
    data = pd.DataFrame(data)
    print(data['text'].values)
    print(data['label'].value_counts())