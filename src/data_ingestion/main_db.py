
import datetime
from pathlib import Path
from pydoc import text

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.canonical_id import canonical_id

DATA_LAKE_ROOT = Path('datalake')
MAIN_CURATED = DATA_LAKE_ROOT / 'curated/main'


class MainDB:
    def __init__(self, twitter_parquet, llm_parquet):
        self.twitter_parquet = twitter_parquet
        self.llm_parquet = llm_parquet
        self.read_download_dataset()

    def read_download_dataset(self):
        self.Sentiment140 = pd.read_csv(
            'datasets/training.1600000.processed.noemoticon.csv', encoding='latin1')
        self.Sentiment140.rename(columns={
                                 "@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D": 'text'}, inplace=True)
        self.Sentiment140['label'] = 0
        self.Sentiment140 = self.Sentiment140[['text', 'label']]
        self.DAIGT = pd.read_csv(
            'datasets/train_v4_drcat_01.csv')[['text', 'label']]

    def merge_to_main(self) -> Path:
        df_twitter = pd.read_parquet(self.twitter_parquet)[['text', 'label']]
        df_llm = pd.read_parquet(self.llm_parquet)[['text', 'label']]
        df_main = pd.concat(
            [df_twitter, df_llm, self.Sentiment140, self.DAIGT], ignore_index=True)
        df_main['text_id'] = df_main.index
        path = MAIN_CURATED / \
            f'main_curated_{datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")}.parquet'
        pq.write_table(pa.Table.from_pandas(df_main), path)
        print(f'Main dataset created: {path}, records={len(df_main)}')
        return path

