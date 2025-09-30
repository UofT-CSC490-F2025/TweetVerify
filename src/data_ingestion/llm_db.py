import os
import hashlib
import datetime
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from utils.canonical_id import canonical_id

DATA_LAKE_ROOT = Path('datalake')
LLM_CURATED = DATA_LAKE_ROOT / 'curated/llm'


class LLMDB:
    def __init__(self, llm_records: list):
        self.llm_records = llm_records

    def process_llm_dataset(self) -> Path:
        df = pd.DataFrame(self.llm_records)
        df['text'] = df['text'].astype(str).str.strip()
        df = df[df['text'].str.len() > 0]
        df['label'] = df.get('label', 1).fillna(1)
        df['text_id'] = df.apply(lambda r: canonical_id('llm', str(r.get('text_id', r.name))), axis=1)
        df['prompt_name'] = df.get('prompt_name', None)
        df['model'] = df.get('model', None)
        df['source'] = 'llm'
        df = df.drop_duplicates(subset=['text'])
        path = LLM_CURATED / f'llm_curated_{datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")}.parquet'
        pq.write_table(pa.Table.from_pandas(df), path)
        print(f'LLM dataset curated: {path}, records={len(df)}')
        return path
