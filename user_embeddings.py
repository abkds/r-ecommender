import json
from nltk.tokenize import RegexpTokenizer
import string

#data = json.loads("user_comments.json")
with open('user_comments.json','r') as fp:
    data = json.load(fp)
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
stop_words = set(stopwords.words('english'))
count = 0
tokenizer = RegexpTokenizer(r'\w+')
for e in data:
    if(count > 1000):
        break
    count+=1
    word_tokens = word_tokenize(data[e]['body']) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = [w for w in filtered_sentence if not w == '\n']
    filtered_sentence = [w for w in filtered_sentence if not w in string.punctuation]
    data[e]['body'] = filtered_sentence

from collections import defaultdict
all_user_comments = defaultdict(list)
count = 0
for e in data:
    curr_list = []
    count+=1
    curr_list = all_user_comments[data[e]['author']]
    for word in (data[e]['body']):
        curr_list.append(word)
    all_user_comments[data[e]['author']] = curr_list


from nltk.tokenize import word_tokenize
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# reviews_train_list = getReviewsList(reviews_train)

tagged_data = [TaggedDocument(words=(_d), tags=[str(i)]) for i,_d in all_user_comments.items()]
model = Doc2Vec(tagged_data, vector_size=50, window=3, min_count=5, epochs = 4, workers=8)

sims = model.docvecs.most_similar('YoungModern')
