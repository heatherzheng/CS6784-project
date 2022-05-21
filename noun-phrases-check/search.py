import csv
import time
import json
import time
import requests
import urllib
import pandas as pd
from requests_html import HTML
from requests_html import HTMLSession
from selenium import webdriver

import nltk
from nltk import pos_tag
from nltk.util import transitive_closure
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

train = []
label = []

with open('test_2k.csv', newline='') as csvfile:
  spamreader = csv.reader(csvfile)
  flag = True
  for row in spamreader:
    if flag:
        flag = False
        continue
    train.append(row[0])
    if row[1]=="natural":
      label.append(1)
    else:
      label.append(0)

def get_source(url):
    """Return the source code for the provided URL.

    Args:
        url (string): URL of the page to scrape.

    Returns:
        response (object): HTTP response object from requests_html.
    """

    try:
        session = HTMLSession()
        #headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36'}
        #response = session.get(url, headers=headers)
        response = session.get(url)
        return response

    except requests.exceptions.RequestException as e:
        print(e)

def get_results(query):
    query = urllib.parse.quote_plus(query)
    #response = get_source("https://www.google.co.uk/search?q=" + query)
    response = get_source("https://www.google.co.uk/search?q=" + "\""+query+"\"")
    return response

def parse_results(response):
    css_identifier_result = ".tF2Cxc"
    css_identifier_title = "h3"
    css_identifier_link = ".yuRUbf a"
    css_identifier_text = ".VwiC3b"

    if response.status_code != 200:
      print("error",str(response.status_code))
      return None
    results = response.html.find(css_identifier_result)
    output = []
    for result in results:

        try:
          item = {
              'title': result.find(css_identifier_title, first=True).text,
              'link': result.find(css_identifier_link, first=True).attrs['href'],
              'text': result.find(css_identifier_text, first=True).text
          }
        except:
            item = {
              'title': result.find(css_identifier_title, first=True).text,
              'link': result.find(css_identifier_link, first=True).attrs['href'],
              'text': ""
            }
        finally:
          output.append(item)
    return output

def google_search(query):
    response = get_results(query)
    return parse_results(response)

def search_NN(sentences,length_thre=2):
  all = []
  for sent in nltk.sent_tokenize(sentences):
    pos = nltk.pos_tag(nltk.word_tokenize(sent))
    prev = -2
    phrase = ""
    for element in range(len(pos)):
      if "NN" in pos[element][1]:
        #print(pos[element],element,prev)
        #print(phrase)
        if element == prev +1:
          phrase = phrase + " " + pos[element][0]
        else:
          if phrase != "" and len(phrase.split())>=length_thre:
            all.append(phrase)
          phrase = pos[element][0]
        prev = element
  return all

lemmatizer = WordNetLemmatizer()

def match_content(phrase, title):
  words = phrase.split(" ")

  for word in words:
    if (not word.lower() in title.lower()) and not (word.lower() in title.replace("-","").lower()):
      return False
  return True


def check_char(input):
  if str.isalpha(input):
    return True
  if str.isnumeric(input):
    return True
  if input=="-":
    return True
  else:
    return False

def doc_to_vec(doc):
  result = {}
  s = set()
  for phrase in search_NN(doc):
    if phrase not in s:
      phrase = " ".join([i for i in [''.join(filter(check_char , word)) for word in phrase.split(" ")] if i!=" " and i!=""])
      if len(phrase.split())<2:
        continue
      phrase = [lemmatizer.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] else lemmatizer.lemmatize(i) for i,j in pos_tag(phrase.split())]
      phrase = " ".join(phrase)
      s.add(phrase)
      tmp = []
      tmp_1 = []
      title = []
      content = []
      search_result = google_search(phrase)
      if search_result ==None:
        return None
      for i in search_result:
        tmp.append(match_content(phrase, i['title']))
        tmp_1.append(match_content(phrase, i['text']))
        title.append(i['title'])
        if i['text']:
          content.append(i['text'])
        else:
          content.append("")
      result[phrase] = {'match_title':tmp,'match_content':tmp_1,'title':title, "content":content}

  return result

new_train = []
for i in range(100):
  result = doc_to_vec(train[i])
  if result==None:
    print("break at " + str(i))
    break
  new_train.append(result)

out_file = open("new_train_natural.json", "w")

json.dump(new_train, out_file, indent = 6)

out_file.close()

#print non-exist phrases
count = 0
index = 0
for i in new_train:
  flag = False
  pos,neg = 0,0
  for phrase in i:
    tmp_1 = []
    tmp_2 = []
    for j in i[phrase]['title']:
      tmp_1.append(match_content(phrase, j))
    for j in i[phrase]['content']:
      tmp_2.append(match_content(phrase, j))

    if not(any(tmp_1)) and  not(any(tmp_2)):
      flag = True
      neg +=1
      print(phrase)
    else:
      pos +=1
  if flag:
    count += 1
  print(pos,neg)
  print(train[index])
  index +=1
print(count)
print(len(new_train_1))
