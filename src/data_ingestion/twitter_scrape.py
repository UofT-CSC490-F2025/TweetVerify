import tweepy
import os
from twitter_db import TwitterDB

bearer_token = os.getenv("TWITTER_BEARER_TOKEN")

def scrape_tweets(query="#AI", max_results=50):
    client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

    tweets = client.search_recent_tweets(
        query=query,
        tweet_fields=["id", "text", "created_at", "author_id", "lang"],
        max_results=max_results
    )

    twitter_records = []
    for t in tweets.data:
        twitter_records.append({
            "text_id": t.id,
            "text": t.text,
            "label": 0,  # human
            "author_id": t.author_id,
            "created_at": str(t.created_at),
            "lang": t.lang,
        })
    return twitter_records

if __name__ == "__main__":
    records = scrape_tweets("AI OR ChatGPT", max_results=100)
    db = TwitterDB(records)
    path = db.process_twitter_dataset()
    print("Twitter data saved at:", path)
