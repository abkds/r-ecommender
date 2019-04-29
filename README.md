# r/ecommender
A subreddit recommender system for users based on user and subreddit similarity using both implicit and explicit signals.

Abhishek Das, Janvi Palan, Nikhil Bhat, Sukanto Guha

## Motivation
Reddit is one of the biggest, most popular sites in the world, and we frequently use Reddit for staying up to date on subjects which interest us. Considering there are _540 million_ users on Reddit, we feel there is a need for a robust recommender system tailor made for Reddit users, so that they can discover new content and immprove their browsing goals, which may be different for different users.
## Dataset
* Our [dataset](https://www.reddit.com/r/datasets/comments/65o7py/updated_reddit_comment_dataset_as_torrents/) comprises of user comments on Reddit from the month of January 2015. It contains 57 million comments from reddit users. One interesting thing about our dataset is that we have more users than items(subreddits), which is unusual for Information Retrieval datasets **and also why algorithms for other datasets cannot be directly applied to Reddit.**
* Pre-processing  
  We crysallized down on three approaches, and we created one dataset for each:
  
  Dataset with user-subreddit interactions for Collaborative Filtering and ALS
    1. We removed user - subreddit interactions which were lesser than 30 characters and fewer than 5 comments
    2. We removed users which were bots and removed comments which were [deleted]
    3. Final dataset size:
      - Users = **735834**
      - Subreddits = **14842**
  
  Dataset with user comments grouped by user
    1. We removed user - subreddit interactions which were lesser than 10 characters and fewer than 3 comments.
    2. Final dataset size: **29 million comments** of the same users and subreddits
  
  Dataset with user comments grouped by subreddit
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
  

