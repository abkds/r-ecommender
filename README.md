# r/ecommender
A subreddit recommender system for users based on user and subreddit similarity using both implicit and explicit signals.

Abhishek Das, Janvi Palan, Nikhil Bhat, Sukanto Guha

## Motivation
Reddit is one of the biggest, most popular sites in the world, and we frequently use Reddit for staying up to date on subjects which interest us. Considering there are _540 million_ users on Reddit, we feel there is a need for a robust recommender system tailor made for Reddit users, so that they can discover new content and immprove their browsing goals, which may be different for different users.
## Dataset
* Our [dataset](https://www.reddit.com/r/datasets/comments/65o7py/updated_reddit_comment_dataset_as_torrents/) comprises of user comments on Reddit from the month of January 2015. It contains 57 million comments from reddit users. One interesting thing about our dataset is that we have more users than items(subreddits), which is unusual for Information Retrieval datasets **and also why algorithms for other datasets cannot be directly applied to Reddit.**
* Pre-processing  
  We crysallized down on three approaches, and we created one dataset for each:
  
  #### Dataset with user-subreddit interactions for Collaborative Filtering and ALS
    1. We removed user - subreddit interactions which were lesser than 30 characters and fewer than 5 comments
    2. We removed users which were bots and removed comments which were [deleted]
    3. Final dataset size:
      - Users = **735834**
      - Subreddits = **14842**
  
  #### Dataset with user comments grouped by user
    1. We removed user - subreddit interactions which were lesser than 10 characters and fewer than 3 comments.
    2. Final dataset size: **29 million comments** of the same users and subreddits
  
  #### Dataset with user comments grouped by subreddit
    1. We removed user - subreddit interactions which were lesser than 10 characters and fewer than 3 comments.
    2. Final dataset size: **29 million comments** of the same users and subreddits

For all three datasets, we performed comment filtering by removing stopwords, fixing punctuations and converting to lower case.

## Methodology
Subreddit recommendation is an important unsolved problem in the broad field of recommender systems, and we tried several methods and finally an ensemble appraoch to tackle this problem.
#### Approach 1: Collaborative Filtering 

  This approach involved Dataset 1 from above. We do not consider the actual words in a comment, but just the fact that the        user has commented on a subreddit as a signal that they like it. Using this model, the advantage was that it was simple to implement, gave us a good baseline, and is easily scalable. We do **not** consider how many times a user comments on a subreddit, just the fact that they have commented. The major drawback of this method is that it falls behind in terms of our evaluation metrics.
  
#### Approach 2: Collaborative Filtering - Alternating Least Squares Matrix Factorization(ALS - MF)
  This approach used Dataset 1. In ALS-MF, we theorize that the number of comments a user has on a subreddit is a strong indicator, and not just the fact that they have commented. The theory is that a user who has 50 comments on a subreddit finds the subreddit more relavant than someone who has 5 comments.

#### Approach 3: Bayesian Personalized Ranking (BPR)
  This approach invloves using [BPR](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf). BPR involves **(user,subreddit,subreddit)** triads. If User1 has commented on Subreddit1 but not Subreddit2, then then (User1,Subreddit1,Subreddit2) will have a positive value. We build such triads for all user and subreddit pairs to build the recommender system.
 
#### Approach 4: Textual BPR (Vanilla t-BPR)
  This approach uses Dataset 2 and 3. This approach uses the approach used in [Visual BPR](https://arxiv.org/pdf/1510.01784.pdf). In the paper, visual embeddings are founf for each item in the Amazon dataset. We use this aprroach by creating textual embeddings for both users and subreddits by concatenating over all the comments made on a subreddit and concatenating all comments made by a user,respectively. Each list of comments is labelled and embeddings are created via [gensim](https://radimrehurek.com/gensim/models/doc2vec.html). These embeddings were used to find the k-most similar subreddits for recommendation to the user. 
  
#### Approach 5: Textual BPR + Training (Learning t-BPR)
  This approach uses Dataset 2 and 3. The difference from Vanilla t-BPR is that the user-user embeddings are trained by our model from the data instead of using gensim for the same. This model was also based on the Visual BPR paper, and considers a Deep CNN model for training the lower dimension embeddings for each user.
  
#### Approach 6: Ensemble - Putting it all together
  In our project, we realized that combining different models like ALS with t-BPR may give better results than a similar model, as user recommendation should ideally take into account [user serendipity, novelty and diversity](http://ir.ii.uam.es/rim3/publications/ddr11.pdf). Choosing the best models is work in progress, and needs further insight on what user goals are when browsing Reddit.

## Evaluation
For evaluation, we split our data into two sets- training and test data. We initially have a list of subreddits a user subscribes to. We take out 10% of subreddits associated with a user and add them to our test set. Once training is complete, we test how many of the subreddits we removed in thte initial set are present in our recommendations. We used the following evaluation model to test our models:
#### Area-Under-the-Curve (AUC)
  This [evaluation criteria](https://wen.wikipedia.org/wiki/Receiver_operating_characteristic#/Area_under_the_curve) gives the probability that a classifier will rank a randomly chosen positive instance higher than a randomly chosen negative one. Because our models are _comparison based_, AUC works well as its defition pertains to the number of comparisons we perform correctly.
  
<img src="https://github.com/abkds/r-ecommender/blob/master/AUCs.png" height="350">

  

#### Techonolgies used - a brief list of libraries and languages
* Python 3
* Jupyter notebook
* gensim
* Colab - Google research
* implicit
* tqdm
* scipy
* nltk

#### [arxiv link](https://arxiv.org/abs/1905.01263)
