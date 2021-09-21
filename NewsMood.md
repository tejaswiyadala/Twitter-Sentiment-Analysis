
# News Mood

### Analysis

* Based on the Bar graph result, CBS news has the highest positive sentiment score, and CNN has the highest negative Sentiment score
* Based on the range of sentiment values, the tweets from Fox News and New York Times are generally the most neutral tweets.
* By analysing the tweets on 03/08/2018, most of the tweets tweeted by media sources have positive sentiments.


```python
# Dependencies
import tweepy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from itertools import cycle
import seaborn as sns

# Twitter API keys
from config import *
```


```python
# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Target News Organizations
target_media = ["BBC","CBS","CNN","FoxNews","nytimes"]
```


```python
# Variables for holding sentiments
news_sentiment = []
```


```python
# Loop through each News Organization
for media in target_media:
    
    # Set Tweet count
    tweet_count = 0
    
    # Grab 100 tweets from the media timeline
    public_tweets = api.user_timeline(media, count = 100, result_type = "recent")
    
    # Loop through each tweet
    for tweet in public_tweets:
        
        # Increment tweet count
        tweet_count = tweet_count + 1
    
        # Run Vader Analysis on each tweet
        compound = analyzer.polarity_scores(tweet["text"])["compound"]
        pos = analyzer.polarity_scores(tweet["text"])["pos"]
        neu = analyzer.polarity_scores(tweet["text"])["neu"]
        neg = analyzer.polarity_scores(tweet["text"])["neg"]
             
        # Add each value to the list
        news_sentiment.append({"Tweet Account" : media,
                               "Date" : tweet["created_at"],
                               "Tweet" : tweet["text"],
                               "Tweet Ago" : tweet_count,
                               "compound_list":compound,
                               "positive_list":pos,
                               "negative_list":neg,
                               "neutral_list":neu})        
        
# Create a dataframe with the values in the list
sentiments_df = pd.DataFrame(news_sentiment)

# Display the first 5 rows of the dataframe
sentiments_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Tweet</th>
      <th>Tweet Account</th>
      <th>Tweet Ago</th>
      <th>compound_list</th>
      <th>negative_list</th>
      <th>neutral_list</th>
      <th>positive_list</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Thu Mar 08 21:28:00 +0000 2018</td>
      <td>This Ancient Egyptian pharaoh invested more in...</td>
      <td>BBC</td>
      <td>1</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Thu Mar 08 21:00:04 +0000 2018</td>
      <td>"No matter where in the world, when civilisati...</td>
      <td>BBC</td>
      <td>2</td>
      <td>-0.2732</td>
      <td>0.103</td>
      <td>0.845</td>
      <td>0.052</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Thu Mar 08 20:33:04 +0000 2018</td>
      <td>Jack Jarvis and Victor McDade are back for ano...</td>
      <td>BBC</td>
      <td>3</td>
      <td>0.4019</td>
      <td>0.000</td>
      <td>0.856</td>
      <td>0.144</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Thu Mar 08 20:03:04 +0000 2018</td>
      <td>Lucy and Lee are back! 🙌🎉\n\n#NotGoingOut | 9p...</td>
      <td>BBC</td>
      <td>4</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Thu Mar 08 18:00:06 +0000 2018</td>
      <td>What advice would YOU give your younger self? ...</td>
      <td>BBC</td>
      <td>5</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Export the data in the DataFrame into a CSV file.
sentiments_df.to_csv("Output/sentiments_data.csv", index = False)
```


```python
# Plot the 100 tweets from each media source with a timestamp "Tweets Ago". 
now = datetime.datetime.now()
sns.set_style("darkgrid")
colors = ['LightBlue','green','red', 'blue','yellow']

for i in np.arange(0,len(target_media)):
    media_source = sentiments_df.loc[sentiments_df["Tweet Account"] == target_media[i]]
    media_source = media_source.sort_values("Tweet Ago")
    plt.scatter(np.arange(len(media_source["compound_list"])), 
            media_source["compound_list"], color = colors[i],
            edgecolor="black", linewidths=1, marker="o",
            alpha=0.8, s =100,label= target_media[i])

plt.xlabel("Tweets Ago")
plt.ylabel("Tweet Polarity")
plt.title("Sentiment Analysis of Media Tweets ({})".format(now.strftime("%m-%d-%y")))

plt.xlim(105,-5)
plt.ylim(-1,1)
plt.legend(bbox_to_anchor=(1.05,1), loc="best", title="Media Sources")

# Save the figure
plt.savefig("Output/Media_Tweets.png")

plt.show()
```


![png](output_10_0.png)



```python
# Create a bar plot
now = datetime.datetime.now()

tips = sns.load_dataset("tips")
ax = sns.barplot(x="Tweet Account", y="compound_list", data = sentiments_df, palette=colors)

plt.title("Overall Media Sentiment Based on Twitter ({})".format(now.strftime("%m-%d-%y")))
plt.ylabel('Tweet Polarity')

# Save the figure
plt.savefig("Output/Media_Sentiments_Barplot.png")
plt.show()
```


![png](output_11_0.png)

