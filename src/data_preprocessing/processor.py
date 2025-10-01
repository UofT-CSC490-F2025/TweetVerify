import pandas as pd
import emoji
import re


class DataProcessor:
    def __init__(self, main_parquet):
        self.main_parquet = main_parquet
        self.data = self.load_data()
        self.processed_data = None

    def load_data(self):
        df_main = pd.read_parquet(self.main_parquet)
        return df_main

    def clean_data(self,
                   lower=True,
                   remove_url=True,
                   remove_user=True,
                   remove_hashtag=True,
                   remove_emoji=True,
                   strip_space=True):
        df = self.data.copy()
        df = df.dropna(subset=['text'])
        df = df.drop_duplicates(subset=['text'])
        df['text'] = df['text'].apply(
            lambda x: self.clean_tweet_text(
                x,
                lower=lower,
                remove_url=remove_url,
                remove_user=remove_user,
                remove_hashtag=remove_hashtag,
                remove_emoji=remove_emoji,
                strip_space=strip_space
            )
        )
        all_human_df = df[df['label'] == 0].copy()
        all_ai_df = df[df['label'] == 1].copy()
        all_human_chars = set(''.join(all_human_df['text'].tolist()))
        all_ai_chars = set(''.join(all_ai_df['text'].tolist()))
        chars_to_remove = ''.join(
            [c for c in all_ai_chars if c not in all_human_chars])
        if chars_to_remove:
            translation_table = str.maketrans('', '', chars_to_remove)
            all_ai_df['text'] = all_ai_df['text'].apply(
                lambda s: s.translate(translation_table))
        self.processed_data = pd.concat(
            [all_human_df, all_ai_df], ignore_index=True)
        return self.processed_data

    def save_data(self, output_path, file_type='parquet'):
        if self.processed_data is None:
            raise ValueError("Data not processed. Run clean_data() first.")
        if file_type == 'parquet':
            self.processed_data.to_parquet(output_path, index=False)
        elif file_type == 'csv':
            self.processed_data.to_csv(output_path, index=False)
        else:
            raise ValueError("file_type must be 'parquet' or 'csv'.")

    def get_data(self):
        if self.processed_data is None:
            raise ValueError("Data not processed. Run clean_data() first.")
        return self.processed_data

    def remove_urls(self, text):
        return re.sub(r'http\S+', '', str(text))

    def remove_user_mentions(self, text):
        return re.sub(r'@\w+', '', str(text))

    def remove_hashtags(self, text):
        return re.sub(r'#\w+', '', str(text))

    def remove_emojis(self, text):
        return emoji.replace_emoji(str(text), replace='')

    def strip_spaces(self, text):
        text = str(text).strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def to_lower(self, text):
        return str(text).lower()

    def clean_tweet_text(self, text,
                         lower=True,
                         remove_url=True,
                         remove_user=True,
                         remove_hashtag=True,
                         remove_emoji=True,
                         strip_space=True):
        if pd.isna(text):
            return ''
        if remove_url:
            text = self.remove_urls(text)
        if remove_user:
            text = self.remove_user_mentions(text)
        if remove_hashtag:
            text = self.remove_hashtags(text)
        if remove_emoji:
            text = self.remove_emojis(text)
        if strip_space:
            text = self.strip_spaces(text)
        if lower:
            text = self.to_lower(text)
        return text
