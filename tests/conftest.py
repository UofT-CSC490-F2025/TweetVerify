import sys
import os
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock gensim
class MockGensim:
    class models:
        class Word2Vec:
            @staticmethod
            def load(*args, **kwargs):
                mock = MagicMock()
                mock.wv.key_to_index = {}
                return mock
            def __init__(self, *args, **kwargs):
                pass

sys.modules['gensim'] = MockGensim
sys.modules['gensim.models'] = MockGensim.models

