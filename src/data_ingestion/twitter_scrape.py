import tweepy
import os
from .twitter_db import TwitterDB

import os
from dotenv import load_dotenv

load_dotenv('./tokem.env')  
bearer_token = os.getenv("TWITTER_BEARER_TOKEN")


def scrape_user_tweets(username, max_results=5):

    client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
    user = client.get_user(username=username)
    user_id = user.data.id
    username_handle = user.data.username
    tweets = client.get_users_tweets(
        id=user_id,
        tweet_fields=["id", "text", "created_at", "lang"],
        max_results=max_results
    )

    twitter_records = []
    if tweets.data:
        for t in tweets.data:
            twitter_records.append({
                "text_id": t.id,
                "text": t.text,
                "user_id": user_id,
                'label': 0,
                "username": username_handle,
                "created_at": str(t.created_at),
                "source": "Twitter API v2",
            })
    return twitter_records

def scrape_keyword_tweets(query="#AI", max_results=10):
    client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

    tweets = client.search_recent_tweets(
        query=query,
        tweet_fields=["id", "text", "created_at", "lang"],
        max_results=max_results
    )

    twitter_records = []
    if tweets.data:
        for t in tweets.data:
            twitter_records.append({
                "text_id": t.id,
                "text": t.text,
                "user_id": t.author_id if hasattr(t, "author_id") else None,
                'label': 0,
                "username": None,  
                "created_at": str(t.created_at),
                "source": f"Twitter API v2 - Keyword ({query})",
            })
    return twitter_records


def scrape_time_range_tweets(query="#AI", start_time="2024-01-01T00:00:00Z", end_time="2024-01-10T00:00:00Z", max_results=10):
    client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

    tweets = client.search_recent_tweets(
        query=query,
        tweet_fields=["id", "text", "created_at", "lang"],
        start_time=start_time,
        end_time=end_time,
        max_results=max_results
    )

    twitter_records = []
    if tweets.data:
        for t in tweets.data:
            twitter_records.append({
                "text_id": t.id,
                "text": t.text,
                "user_id": t.author_id if hasattr(t, "author_id") else None,
                'label': 0,
                "username": None,
                "created_at": str(t.created_at),
                "source": f"Twitter API v2 - Time Range ({start_time} ~ {end_time})",
            })
    return twitter_records
