# OSRS-Price-Predictions-for-Items-with-High-Trade-Volume
ML Project: Price Prediction for Runes in OSRS

For context, OSRS is Old School Runescape, which is a large MMORPG (Massibe Multiplayer Online Role-Playing Game) with an equally large virtual market, where players are able to trade gold for items from other players.

There was a recent update, where the developers wanted to introduce a new prayer book (new benefits to gameplay), which was unprecedented. Initial reports from the playerbase suggested that items relating to prayer training (bones being the most common) as going up significantly in price despite it being a heavily botted item (botted meaning that there are illegal bot-players being used to drive up the supply in the market).

Given this change, I was interested if the market for prayer items was subject to the EMH, or if I could actually predict the prices of these specific goods.

# Current Status of the Project

This project is currently finished its first iteration, having done a wide variety of ML predictive models for both regression and classification problems.

# Automated Rune Price Scraping.ipynb, df_concat.ipynb, rune_prices_merged.csv

Uses the API provided by the OSRS team to collect prices in 5 minute intervals, convert the information into dataframe objects, and then export as csv files. This was not used in the latest iteration of the prediction models, as we were interested in predicting bone prices (as there had been a recent update that caused a sudden shock in the prices, which we wanted to see if we could predict accurately in both the present and future periods).

The same holds for the python file where we concate the dataframes, and the csv output.

# Reddit Comment Web Scraping.ipynb, text_df.csv

Using the API provided by Reddit, I was able to scrape the comments, and replies to comments on the official developer thread discussing the update relating to the prayer book that the developers wanted to introduce. We were able to extract all of the comments and their associated user id and upvotes. This was then converted into a csv file (text_df.csv), which had 2261 separate observations.

# OSRS Update Sentiment Analysis.ipynb
The first step taken was to preprocess the comments, which was accomplished by removing common stopwords, abbreviations of phrases, and punctuations. With the comments properly cleaned we moved on to topic modeling.

I wanted to establish what topics were present in the comment thread before we conducted the sentiment analysis. Given that I did not have the topic labels associated with all of the words we expected to see, we instead used an unsupervised topic modeling method. First, I vectorized the word space using bag of bi-grams, and then used LDA model to find the optimal number of topics.

Once I determined the optimal number of topics, I went through them to give them proper names before assigning the documents (comments) in our corpus their corresponding dominant topic. I then used the VADER lexicon to determine the sentiment scores of the individual documents, before taking the average sentiment scores for each of the topics present in the corpus. If these topics had a significant positive/negative compound score (absolute value greater than or equal to 0.05). Since all of the relevant topics had positive sentiment scores (significant positive compound scores), I interpreted that the general reception to the update was positive, which provides some context regarding the change in prices of the related goods.

# ML Models:

Before describing the models we used, it is important to set up some context relating to our question, specifically if the market in OSRS follows the efficient market hypothesis (EMH), that is the item prices already account for all information available, and thus are perfectly priced.

To test this, we compare our predictive models in both classification and regression cases to the persistence model, if all of our models in the regression models are less accurate than the regression persistence model (higher test MSE), then we state that there is random walk, and as a result we cannot outperform (predict prices) the market. The same holds for classification, but instead we are looking if we can predict the direction of the change in the prices of these goods.

The outcome variables that we are trying to predict are the prices and indicator of direction in price change in the 'current period' (5 minute intervals). We use nowcasting, as we do not actually have the prices in a current period as it is based on the average prices (high and low prices at which the item is sold) in the 5 minutes after the specified period (for example, 9:45 AM would be the average of prices from 9:45 to 9:50 AM).

The predictor variables were composed of the prices and trade volume of the main good that was lagged by 1 period, along with the prices and trade volume of the set of substitute goods lagged by 1 period.

# Price Prediction Regression Models.ipynb

In the regression problem, we ran the following ML models: regression tree, random forest, gradient boosted tree and knn 

These models all performed worse than the persistence model, which I interpret as the market for goods related to the prayer book update (bones) following the EMH.

# Price Classification Prediction Models.ipynb

In the classification problem, we ran the following ML models: logistic regression, decision tree, random forest, gradient boosted tree and knn

All of these models performed better than the persistence model, having higher accuracy scores and AUCs, thus suggesting that it is possible to predict the direction in which prices change in this market.

# Next Steps

Given the non-linearity in the data I plan on implementing neural networks to account for this issue.


