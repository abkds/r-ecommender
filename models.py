
#%%
import implicit
import numpy as np
from tqdm import tqdm_notebook
import pandas as pd
import csv 
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from implicit.nearest_neighbours import bm25_weight
from implicit import alternating_least_squares
import umap


#%%
data = []
with open('interactions_30_ch_no_bots') as csvfile:
    datareader = csv.reader(csvfile, delimiter=' ')
    for subreddit, user, comments, _ in datareader:
        data.append([user, subreddit, int(comments)])


#%%
data = pd.DataFrame.from_records(data)


#%%
data.columns = ['user', 'subreddit', 'comments']


#%%
data['user'] = data['user'].astype("category")
data['subreddit'] = data['subreddit'].astype("category")


#%%
# create a sparse matrix of all the artist/user/play triples
comments = coo_matrix((data['comments'].astype(float), 
                   (data['subreddit'].cat.codes, 
                    data['user'].cat.codes)))

#%% [markdown]
# ### Latent Semantic Analysis

#%%
# toggle this variable if you want to recalculate the als factors
read_als_factors_from_file = True


#%%
if read_als_factors_from_file:
    subreddit_factors = np.load('subreddit_factors_als.npy')
    user_factors = np.load('user_factors_als.npy')
else:
    subreddit_factors, user_factors = alternating_least_squares(bm25_weight(comments), 20)


#%%
subreddit_factors, user_factors = alternating_least_squares(bm25_weight(comments), 20)


#%%
class TopRelated(object):
    def __init__(self, subreddit_factors):
        norms = np.linalg.norm(subreddit_factors, axis=-1)
        self.factors = subreddit_factors / norms[:, np.newaxis]
        self.subreddits = data['subreddit'].cat.categories.array.to_numpy()

    def get_related(self, subreddit, N=10):
        subredditid = np.where(self.subreddits == subreddit)[0][0]
        scores = self.factors.dot(self.factors[subredditid])
        best = np.argpartition(scores, -N)[-N:]
        best_ = [self.subreddits[i] for i in best]
        return sorted(zip(best_, scores[best]), key=lambda x: -x[1])


#%%
top_related = TopRelated(subreddit_factors)


#%%
top_related.get_related('OnePiece')


#%%
subreddit_factors.shape


#%%
subreddits_embedded = umap.UMAP().fit_transform(subreddit_factors)
subreddits_embedded.shape


#%%
subreddits_embedded


#%%
subreddits = data['subreddit'].cat.categories.array.to_numpy()


#%%
import random

indices = random.sample(range(len(subreddits)), 1000)


#%%
sampled_subreddits = subreddits[indices]
sampled_subreddits_embedded = subreddits_embedded[indices]


#%%
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

plotly.tools.set_credentials_file(username='abkds', api_key='KKuXHMUKu7EHg9kIZWrl')


# Create random data with numpy
import numpy as np

N = 500
xs = sampled_subreddits_embedded[:, 0]
ys = sampled_subreddits_embedded[:, 1]

# Create a trace
trace = go.Scatter(
    x = xs,
    y = ys,
    mode='markers+text',
    text=sampled_subreddits
)

data_ = [trace]

# Plot and embed in ipython notebook!
py.iplot(data_, filename='basic-scatter')

# or plot with: plot_url = py.plot(data, filename='basic-line')

#%% [markdown]
# ### Bayesian Personalized Ranking

#%%
from implicit.bpr import BayesianPersonalizedRanking

params = {"factors": 63}


#%%
import logging
import tqdm
import time
import codecs


#%%
model = BayesianPersonalizedRanking(**params)


#%%
model_name = 'bpr'
output_filename = 'subreddits_recs_bpr'


#%%
model.fit(comments)


#%%
def bpr_related_subreddits(subreddit):
    found = np.where(subreddits == subreddit)
    if len(found[0]) == 0:
        raise ValueError("Subreddit doesn't exist in the dataset.")
    _id = found[0][0]
    return [(subreddits[i], v) for i, v in model.similar_items(_id)]


#%%
bpr_related_subreddits('dogs')


#%%
users = data['user'].cat.categories.array.to_numpy()


#%%
write_bpr_recommendations = False


#%%
user_comments = comments.T.tocsr()
if write_bpr_recommendations:
    # generate recommendations for each user and write out to a file
    with tqdm.tqdm_notebook(total=len(users)) as progress:
        with codecs.open(output_filename, "w", "utf8") as o:
            for userid, username in enumerate(users):
                for subredditid, score in model.recommend(userid, user_comments):
                    o.write("%s\t%s\t%s\n" % (username, subreddits[subredditid], score))
                progress.update(1)

#%% [markdown]
# ### Sample user recommendations
# 
# We went through the user 'xkcd_transciber' list of subreddits, where he/she commented. Taking a view of the kind of subreddits followed by the user we see that the predictions are good. This is just one sample, we are saving the recommendations for all users in a file and will also write the AUC score function for getting the exact scores for the generated recommendations.

#%%
def recommend_for_user(username):
    sample_user_id = np.where(users == username)[0][0]
    return [(subreddits[i], v) for i, v in model.recommend(2293528, user_comments)]


#%%
recommend_for_user('xkcd_transcriber')


#%%
def subreddits_interacted_by_user(username):
    sample_user_id = np.where(users == username)[0][0]
    _idlist =  comments.getcol(sample_user_id)
    return [subreddits[idx] for idx, i in enumerate(_idlist.toarray()) if i != 0.0]


#%%
# sample 50 reddits with which xkcd_transcriber has interacted with.
random.sample(subreddits_interacted_by_user('xkcd_transcriber'), 50)


#%%
# set seed to get the same train and test set
np.random.seed(42)

filename = 'interactions_30_ch_no_bots'
train_filename = 'interactions_5'

def create_dataset():
    data = defaultdict(lambda: [])
    with open(filename) as csvfile:
        datareader = csv.reader(csvfile, delimiter=' ')        
        for subreddit, user, comments, _ in tqdm.tqdm_notebook(datareader):
            data[user].append((subreddit, comments))
    

    f_train = open(train_filename, 'a')
    
    for user, items in tqdm.tqdm_notebook(data.items()):
        np.random.shuffle(items)
        if len(items) >= 5:
            for item in items:
                line = ' '.join(list(map(str, [item[0], user, item[1]]))) + '\n'
                f_train.write(line)
    
    f_train.close()
        
create_dataset()


#%%
data = []
with open('interactions_5') as csvfile:
    datareader = csv.reader(csvfile, delimiter=' ')
    for subreddit, user, comments in datareader:
        data.append([user, subreddit, int(comments)])


#%%
data = pd.DataFrame.from_records(data)
data.columns = ['user', 'subreddit', 'comments']

data['user'] = data['user'].astype("category")
data['subreddit'] = data['subreddit'].astype("category")


#%%
# create a sparse matrix of all the artist/user/play triples
comments = coo_matrix((data['comments'].astype(float), 
                   (data['subreddit'].cat.codes, 
                    data['user'].cat.codes)))


#%%
comments


#%%
subreddits = data['subreddit'].cat.categories.array.to_numpy()
users = data['user'].cat.categories.array.to_numpy()


#%%
print('Number of users for BPR model: %s' % len(users))
print('Number of subreddits for BPR model: %s' % len(subreddits))

#%% [markdown]
# Create the index and the reverse index for the users and subreddits

#%%
def item_to_index(things):
    index = {}
    for idx, item in enumerate(things):
        index[item] = idx
    return index

def index_to_item(index):
    things = np.empty(len(index), dtype=object)
    for item, idx in index.items():
        things[idx] = item
    return things


#%%
subreddits_index = item_to_index(subreddits)
users_index = item_to_index(users)

#%% [markdown]
# ### Extracting test set
# 
# We will pluck out the test set, as per the strategy given in the paper [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/pdf/1205.2618.pdf), section 6.2 

#%%
def train_test_split(coo_comments):
    """
    Omits random user subreddit interactions, zeros them out 
    and appends them to the test list.
    """
    csr_comments = coo_comments.tocsr()
    
    data = defaultdict(lambda: [])
    with open('interactions_5') as csvfile:
        datareader = csv.reader(csvfile, delimiter=' ')        
        for subreddit, user, comments in tqdm.tqdm_notebook(datareader):
            data[user].append((subreddit, comments))
    
    train_set = []
    test_set = []
    
    for user, items in tqdm.tqdm_notebook(data.items()):
        np.random.shuffle(items)
        test_item = items[0]
        test_comments = items[1]
        
        test_subreddit = test_item[0]
        # zero out a user item interaction
        csr_comments[subreddits_index[test_subreddit], users_index[user]] = 0
        
        test_set.append([test_subreddit, user, int(comments)])
        
        for item in items[1:]:
            train_set.append([item[0], user, int(item[1])])
        
    csr_comments.eliminate_zeros()
    return train_set, test_set, csr_comments.tocoo()


#%%
train_set, test_set, comments = train_test_split(comments)

#%% [markdown]
# ### AUC Metric
# 
# We will implement the AUC Metric for evaluation of BPR based methods. We take the definition given in the paper [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/pdf/1205.2618.pdf), section 6.2 . AUC is defined as 
#  
# $$AUC = \frac{1}{| U |} \sum_u \frac{1}{|E(u)|} \sum_{(i, j) \in E(u)} \delta(\hat{x}_{ui} - \hat{x}_{uj}) $$
# 
# where $$E(u) := \{(i, j) | (u, i) \in S_{test} ∧ (u, j) \notin (S_{test} ∪ S_{train})\}$$
# 
# 

#%%
# create E(u) list for each user and store it use ids instead of names to store them
E_u = defaultdict(lambda : set())

for subreddit, user, _ in tqdm.tqdm_notebook(train_set):
        E_u[users_index[user]].add(subreddits_index[subreddit])
            
for subreddit, user, _ in tqdm.tqdm_notebook(test_set):
        E_u[users_index[user]].add(subreddits_index[subreddit])


#%%
# train the bpr model 
from implicit.bpr import BayesianPersonalizedRanking

params = {"factors": 63}


#%%
model = BayesianPersonalizedRanking(**params)


#%%
comments


#%%
model.fit(comments)


#%%
num_subreddits = len(subreddits)


#%%
def auc(test_set, user_factors, subreddit_factors, subreddits, users):
    """
    Returns the auc score on a test data set
    """
    num_users = len(test_set)
    
    total = 0
    
    # treat the signal as 1 as per the implicit bpr paper
    for subreddit, user, signal in tqdm.tqdm_notebook(test_set):  # outer summation
        # inner summation 
        # TODO: try to parallelize 
        u = users_index[user]
        i = subreddits_index[subreddit]
        
        x_ui = user_factors[u].dot(subreddit_factors[i])
        
        js = []
        
        for j in range(0, num_subreddits):
            if j != i and j not in E_u[u]:
                js.append(j)
                
        total += np.sum(np.heaviside(x_ui - user_factors[u].dot(subreddit_factors[js].T), 0)) / len(js)
            
        # for j in range(0, subreddits):
        #    numel = 0
        #    total_user = 0
        #    if j != i and j not in E_u[u]:
        #        numel += 1
        #        x_uj = user_factors[u].dot(subreddit_factors[j])
        #            total_user += heaviside(x_ui - x_uj)
        
        # total += (total_user * 1.0 / numel)
    
    return total / num_users


#%%
auc(test_set[:10000], model.user_factors, model.item_factors, subreddits, users)


#%%
def get_aucs_vs_factors():
    factors = [8, 16, 32, 64, 128]
    params_list = [{"factors": factor} for factor in factors]
    
    aucs = []
    
    for params in params_list:
        model = BayesianPersonalizedRanking(**params)
        model.fit(comments)
        aucs.append(auc(test_set[:20000], model.user_factors, model.item_factors, subreddits, users))
    
    return aucs


#%%
aucs_vs_factors = get_aucs_vs_factors()


#%%
aucs_vs_factors


#%%
def get_aucs_vs_factors_als():
    factors = [8, 16, 32, 64, 128]
    
    aucs = []
    
    for factor in factors:
        subreddit_factors, user_factors = alternating_least_squares(bm25_weight(comments), factor)
        aucs.append(auc(test_set[:20000], user_factors, subreddit_factors, subreddits, users))
    
    return aucs


#%%
aucs_als = get_aucs_vs_factors_als()


#%%
aucs_als


#%%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

with plt.xkcd():
    xs = np.array([8, 16, 32, 64, 128])
    ys = aucs_vs_factors
    axes = plt.axes()
    plt.semilogx(xs, ys, '-gD', xs, aucs_als, '-g^')
    axes.set_ylim([0.75, 1.0])
    axes.set_xticks([10, 20, 50, 100])
    axes.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes.set_xlabel('number of dimensions')
    axes.set_ylabel('AUC')
    plt.title('AUC scores for Reddit January Data')
    plt.grid()
    plt.show()


