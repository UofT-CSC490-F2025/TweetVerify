import os
import hashlib
import datetime
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.utils.canonical_id import canonical_id

DATA_LAKE_ROOT = Path('datalake')
TWITTER_CURATED = DATA_LAKE_ROOT / 'curated/twitter'


class TwitterDB:
    def __init__(self, twitter_records: list):
        self.twitter_records = twitter_records

    def process_twitter_dataset(self) -> Path:
        df = pd.DataFrame(self.twitter_records)
        df['text'] = df['text'].astype(str).str.strip()
        df = df[df['text'].str.len() > 0]
        df['label'] = df.get('label', 0).fillna(0)
        df['text_id'] = df.apply(lambda r: canonical_id(
            'twitter', str(r.get('text_id', r.name))), axis=1)
        df = df.drop_duplicates(subset=['text'])
        path = TWITTER_CURATED / \
            f'twitter_curated_{datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")}.parquet'
        pq.write_table(pa.Table.from_pandas(df), path)
        print(f'Twitter dataset curated: {path}, records={len(df)}')
        self.df = df
        return path

    def save_to_csv(self, path) -> Path:
        self.df.to_csv(path, index=False, mode='a')
        print(f'Twitter dataset saved to CSV: {path}, records={len(self.df)}')
        return path
