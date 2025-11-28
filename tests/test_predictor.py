import math
import torch
import pytest
from unittest.mock import patch

from src.inference.predictor import Predictor


class FakeW2V:
    class WV:
        key_to_index = {}

    wv = WV()


class FakeW2VWithVocab:
    class WV:
        key_to_index = {"hello": 0, "world": 1}

    wv = WV()


class DummyTokenizer:
    def __call__(
        self,
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    ):
        # 简单返回全 0 的 input_ids 和全 1 的 attention_mask
        return {
            "input_ids": torch.zeros((1, max_length), dtype=torch.long),
            "attention_mask": torch.ones((1, max_length), dtype=torch.long),
        }


class DummyTokenizer128:
    def __call__(
        self,
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    ):
        return {
            "input_ids": torch.zeros((1, max_length), dtype=torch.long),
            "attention_mask": torch.ones((1, max_length), dtype=torch.long),
        }


class DummyBertModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return "bert"

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        logits = torch.tensor([[2.0, 1.0]])
        return logits.repeat(batch_size, 1)


class RecordingRNNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None  

    def get_name(self):
        return "lstm"

    def forward(self, x):
        # x: [batch_size, seq_len]
        self.last_input = x.detach().clone()
        batch_size = x.size(0)
        logits = torch.tensor([[2.0, 1.0]])
        return logits.repeat(batch_size, 1)


@patch("src.inference.predictor.Word2Vec")
def test_predict_single_bert(mock_w2v):
    mock_w2v.load.return_value = FakeW2V()

    device = "cpu"
    model = DummyBertModel()
    tokenizer = DummyTokenizer()

    predictor = Predictor(
        model=model,
        device=device,
        tokenizer=tokenizer,
        word2vec_model_path="dummy.model",
    )

    text = "hello world"
    pred, conf = predictor.predict(text)

    # 类型检查
    assert isinstance(pred, int)
    assert isinstance(conf, float)

    assert pred == 0

    expected_conf = torch.softmax(torch.tensor([[2.0, 1.0]]), dim=1).max().item()
    assert 0.0 <= conf <= 1.0
    assert math.isclose(conf, expected_conf, rel_tol=1e-5)


@patch("src.inference.predictor.Word2Vec")
def test_predict_single_rnn_and_indices_mapping(mock_w2v):
    mock_w2v.load.return_value = FakeW2VWithVocab()

    device = "cpu"
    model = RecordingRNNModel()
    tokenizer = None 

    predictor = Predictor(
        model=model,
        device=device,
        tokenizer=tokenizer,
        word2vec_model_path="dummy.model",
    )


    text = "hello world unknown"
    pred, conf = predictor.predict(text)

    # 输出类型 & 数值
    assert isinstance(pred, int)
    assert isinstance(conf, float)
    assert pred == 0  
    assert 0.0 <= conf <= 1.0

    assert model.last_input is not None
    expected_indices = torch.tensor([[1, 2, 0]], dtype=torch.long)
    assert torch.equal(model.last_input.cpu(), expected_indices)


@patch("src.inference.predictor.Word2Vec")
def test_predict_batch_bert(mock_w2v):
    mock_w2v.load.return_value = FakeW2V()

    device = "cpu"
    model = DummyBertModel()
    tokenizer = DummyTokenizer128()  # batch 分支里 max_length=128

    predictor = Predictor(
        model=model,
        device=device,
        tokenizer=tokenizer,
        word2vec_model_path="dummy.model",
    )

    texts = ["text A", "text B", "text C"]
    results = predictor.predict_batch(texts)

    assert len(results) == len(texts)

    expected_conf = torch.softmax(torch.tensor([[2.0, 1.0]]), dim=1).max().item()

    for pred, conf in results:
        assert isinstance(pred, int)
        assert isinstance(conf, float)
        assert pred == 0
        assert 0.0 <= conf <= 1.0
        assert math.isclose(conf, expected_conf, rel_tol=1e-5)


@patch("src.inference.predictor.Word2Vec")
def test_predict_batch_rnn(mock_w2v):
    mock_w2v.load.return_value = FakeW2VWithVocab()

    device = "cpu"
    model = RecordingRNNModel()
    tokenizer = None

    predictor = Predictor(
        model=model,
        device=device,
        tokenizer=tokenizer,
        word2vec_model_path="dummy.model",
    )

    texts = [
        "hello world",         
        "unknown hello world", 
    ]
    results = predictor.predict_batch(texts)

    assert len(results) == len(texts)

    expected_conf = torch.softmax(torch.tensor([[2.0, 1.0]]), dim=1).max().item()

    for pred, conf in results:
        assert isinstance(pred, int)
        assert isinstance(conf, float)
        assert pred == 0
        assert 0.0 <= conf <= 1.0
        assert math.isclose(conf, expected_conf, rel_tol=1e-5)

    assert model.last_input is not None
    assert model.last_input.dim() == 2
    assert model.last_input.size(0) == len(texts)
