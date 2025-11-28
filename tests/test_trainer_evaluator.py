import pytest
import torch
from unittest.mock import MagicMock, patch
from torch.utils.data import Dataset
from src.dataloader.bertdataset import BertDataset
from src.dataloader.featuredataset import FeatureDataset
from src.evaluator.evaluator import Evaluator
from src.trainer.trainer import Trainer

# --- BertDataset Tests ---
def test_bert_dataset():
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }
    
    dataset = BertDataset(
        texts=["Hello"],
        labels=[1],
        tokenizer=mock_tokenizer,
        max_len=10
    )
    
    assert len(dataset) == 1
    
    item = dataset[0]
    assert 'input_ids' in item
    assert 'attention_mask' in item
    assert 'label' in item
    assert item['label'] == 1
    
    # Check tokenizer call
    mock_tokenizer.assert_called_with(
        "Hello",
        truncation=True,
        padding='max_length',
        max_length=10,
        return_tensors='pt'
    )

# --- FeatureDataset Tests ---
def test_feature_dataset():
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode_plus.return_value = {
        'input_ids': torch.tensor([1, 2, 3]),
        'attention_mask': torch.tensor([1, 1, 1])
    }
    
    dataset = FeatureDataset(
        texts=["Hello"],
        labels=[1],
        features=[[0.1, 0.2]],
        tokenizer=mock_tokenizer,
        max_len=10
    )
    
    assert len(dataset) == 1
    
    item = dataset[0]
    assert 'input_ids' in item
    assert 'attention_mask' in item
    assert 'label' in item
    assert 'extra_features' in item
    assert item['label'] == 1
    assert torch.allclose(item['extra_features'], torch.tensor([0.1, 0.2]))
    
    # Check tokenizer call
    mock_tokenizer.encode_plus.assert_called_with(
        "Hello",
        add_special_tokens=True,
        max_length=10,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

# --- Evaluator Tests ---
@pytest.fixture
def mock_model():
    model = MagicMock()
    model.to.return_value = model
    return model

def test_evaluator_rnn(mock_model):
    mock_model.get_name.return_value = "rnn"
    
    # Mock dataloader yielding one batch
    # batch: (input, label)
    x = torch.randn(2, 10)
    t = torch.tensor([0, 1])
    
    with patch('src.evaluator.evaluator.DataLoader') as mock_loader:
        mock_loader.return_value = [(x, t)]
        
        evaluator = Evaluator(mock_model, [], torch.device('cpu'))
        
        # Mock model output (logits)
        mock_model.return_value = torch.tensor([
            [0.9, 0.1],
            [0.2, 0.8]
        ])
        
        acc, f1, auc = evaluator.accuracy()
        assert acc == 1.0

def test_evaluator_bert(mock_model):
    mock_model.get_name.return_value = "bert"
    
    batch = {
        "input_ids": torch.randn(2, 10),
        "attention_mask": torch.randn(2, 10),
        "label": torch.tensor([0, 1])
    }
    
    with patch('src.evaluator.evaluator.DataLoader') as mock_loader:
        mock_loader.return_value = [batch]
        
        evaluator = Evaluator(mock_model, [], torch.device('cpu'))
        
        # Mock output (logits)
        mock_model.return_value = torch.tensor([
            [0.9, 0.1],
            [0.9, 0.1]
        ])
        
        acc, f1, auc = evaluator.accuracy()
        assert acc == 0.5

def test_evaluator_roberta_extra(mock_model):
    mock_model.get_name.return_value = "roberta_extra"
    
    batch = {
        "input_ids": torch.randn(2, 10),
        "attention_mask": torch.randn(2, 10),
        "extra_features": torch.randn(2, 2),
        "label": torch.tensor([0, 1])
    }
    
    with patch('src.evaluator.evaluator.DataLoader') as mock_loader:
        mock_loader.return_value = [batch]
        
        evaluator = Evaluator(mock_model, [], torch.device('cpu'))
        
        # Mock output (logits)
        mock_model.return_value = torch.tensor([
            [0.9, 0.1],  # correct
            [0.2, 0.8]   # correct
        ])
        
        acc, f1, auc = evaluator.accuracy()
        assert acc == 1.0
        
        # Verify model called with extra_features
        mock_model.assert_called()
        call_args = mock_model.call_args
        assert len(call_args[0]) == 3 # input_ids, attention_mask, extra_features

# --- Trainer Tests ---
def test_trainer_rnn(mock_model, tmp_path):
    # Mock parameters to not be empty
    param = torch.nn.Parameter(torch.tensor([1.0]))
    mock_model.parameters.return_value = [param]
    mock_model.get_name.return_value = "rnn"

    # Mock data
    x = torch.randn(2, 10)
    t = torch.tensor([0, 1])

    # Mock model forward to return Tensor with grad
    mock_output = torch.randn(2, 2, requires_grad=True)
    mock_model.return_value = mock_output

    # Mock evaluator to return accuracy
    with patch('src.trainer.trainer.Evaluator') as mock_eval_cls, \
         patch('src.trainer.trainer.torch.utils.data.DataLoader') as mock_loader_cls, \
         patch('src.trainer.trainer.os.environ', {}), \
         patch('src.trainer.trainer.os.makedirs'): # Mock makedirs

        mock_eval_instance = MagicMock()
        mock_eval_instance.accuracy.side_effect = [(0.5, 0.5, 0.5), (0.8, 0.8, 0.8)] # Improving accuracy
        mock_eval_cls.return_value = mock_eval_instance

        # Mock dataloader
        mock_loader_cls.return_value = [(x, t)]

        # Mock torch.save and load
        with patch('torch.save') as mock_save, \
             patch('torch.load') as mock_load, \
             patch('src.trainer.trainer.os.remove') as mock_remove:

            mock_load.return_value = {} # Empty state dict

            trainer = Trainer(
                device=torch.device('cpu'),
                model=mock_model,
                train_data=[(1,1)], 
                val_data=[],
                num_epochs=2,
                model_save_dir=str(tmp_path)
            )
            # Force overwrite train_loader to ensure it behaves exactly as we want
            trainer.train_loader = [(x, t)]

            # Run training
            train_loss, val_acc = trainer.train_model()
            
            assert len(train_loss) == 2
            assert len(val_acc) == 2
            assert val_acc == [0.5, 0.8]
            assert mock_save.called

def test_trainer_bert(mock_model, tmp_path):
    mock_model.get_name.return_value = "bert"
    param = torch.nn.Parameter(torch.tensor([1.0]))
    mock_model.parameters.return_value = [param]
    
    # Mock forward output
    mock_output = torch.randn(2, 2, requires_grad=True)
    mock_model.return_value = mock_output
    
    batch = {
        "input_ids": torch.randn(2, 10),
        "attention_mask": torch.randn(2, 10),
        "label": torch.tensor([0, 1])
    }
    
    with patch('src.trainer.trainer.Evaluator') as mock_eval_cls:
    
        mock_eval_instance = MagicMock()
        mock_eval_instance.accuracy.return_value = (0.9, 0.9, 0.9)
        mock_eval_cls.return_value = mock_eval_instance

        # Mock loss
        with patch('torch.nn.CrossEntropyLoss') as mock_loss_cls, \
             patch('torch.save') as mock_save, \
             patch('torch.load') as mock_load, \
             patch('src.trainer.trainer.os.remove') as mock_remove, \
             patch('src.trainer.trainer.os.makedirs'):
            
            mock_load.return_value = {}
            
            mock_loss_fn = MagicMock()
            mock_loss_fn.return_value.item.return_value = 0.1
            mock_loss_cls.return_value = mock_loss_fn
    
            trainer = Trainer(
                device=torch.device('cpu'),
                model=mock_model,
                train_data=[1], 
                val_data=[],
                num_epochs=1,
                model_save_dir=str(tmp_path)
            )
            # Force overwrite train_loader
            trainer.train_loader = [batch]
    
            train_loss, val_acc = trainer.train_model()
        
        assert len(train_loss) == 1
        mock_model.assert_called() # Forward pass called
        assert mock_save.called

def test_trainer_roberta_extra(mock_model, tmp_path):
    mock_model.get_name.return_value = "roberta_extra"
    param = torch.nn.Parameter(torch.tensor([1.0]))
    mock_model.parameters.return_value = [param]
    
    mock_output = torch.randn(2, 2, requires_grad=True)
    mock_model.return_value = mock_output
    
    batch = {
        "input_ids": torch.randn(2, 10),
        "attention_mask": torch.randn(2, 10),
        "extra_features": torch.randn(2, 2),
        "label": torch.tensor([0, 1])
    }
    
    with patch('src.trainer.trainer.Evaluator') as mock_eval_cls:
        mock_eval_instance = MagicMock()
        mock_eval_instance.accuracy.return_value = (0.9, 0.9, 0.9)
        mock_eval_cls.return_value = mock_eval_instance

        with patch('torch.nn.CrossEntropyLoss') as mock_loss_cls, \
             patch('torch.save') as mock_save, \
             patch('torch.load') as mock_load, \
             patch('src.trainer.trainer.os.remove') as mock_remove, \
             patch('src.trainer.trainer.os.makedirs'):
            
            mock_load.return_value = {}
            mock_loss_fn = MagicMock()
            mock_loss_fn.return_value.item.return_value = 0.1
            mock_loss_cls.return_value = mock_loss_fn
    
            trainer = Trainer(
                device=torch.device('cpu'),
                model=mock_model,
                train_data=[1], 
                val_data=[],
                num_epochs=1,
                model_save_dir=str(tmp_path)
            )
            trainer.train_loader = [batch]
    
            train_loss, val_acc = trainer.train_model()
            
            # Verify called with extra features
            mock_model.assert_called()
            call_args = mock_model.call_args
            assert len(call_args[0]) == 3

def test_trainer_no_epochs(mock_model, tmp_path):
    """Test trainer with 0 epochs"""
    mock_model.get_name.return_value = "bert"
    mock_model.parameters.return_value = [torch.nn.Parameter(torch.tensor([1.0]))]
    
    with patch('src.trainer.trainer.Evaluator'), \
         patch('src.trainer.trainer.DataLoader'), \
         patch('src.trainer.trainer.os.makedirs'):
        
        trainer = Trainer(
            device=torch.device('cpu'),
            model=mock_model,
            train_data=[],
            val_data=[],
            num_epochs=0,
            model_save_dir=str(tmp_path)
        )
        
        trainer.train_model()
        # best_model_path should be empty, so no load called.
        # No errors should occur.

def test_trainer_init_defaults(mock_model):
    mock_model.get_name.return_value = "bert" # Not rnn/lstm
    
    # Test default model_save_dir from env
    with patch('src.trainer.trainer.os.environ', {'SM_MODEL_DIR': '/env/path'}), \
         patch('src.trainer.trainer.Evaluator'), \
         patch('src.trainer.trainer.DataLoader'), \
         patch('src.trainer.trainer.os.makedirs'):
        
        trainer = Trainer(
            device=torch.device('cpu'),
            model=mock_model,
            train_data=[],
            val_data=[]
        )
        assert trainer.model_save_dir == '/env/path'

def test_trainer_init_rnn_loader(mock_model):
    mock_model.get_name.return_value = "rnn"
    
    with patch('src.trainer.trainer.os.environ', {'SM_MODEL_DIR': '/env/path'}), \
         patch('src.trainer.trainer.Evaluator'), \
         patch('src.trainer.trainer.DataLoader') as mock_loader, \
         patch('src.trainer.trainer.os.makedirs'):
        
        Trainer(
            device=torch.device('cpu'),
            model=mock_model,
            train_data=[(1, 1)], # Non-empty
            val_data=[]
        )
        # Verify it used the torch.utils.data.DataLoader path with collate_fn
        mock_loader.assert_called()
        call_kwargs = mock_loader.call_args[1]
        assert 'collate_fn' in call_kwargs

def test_trainer_save_delete_old(mock_model, tmp_path):
    mock_model.get_name.return_value = "bert"
    # Mock parameters
    param = torch.nn.Parameter(torch.tensor([1.0]))
    mock_model.parameters.return_value = [param]
    # Mock output
    mock_model.return_value = torch.randn(2, 2, requires_grad=True)
    
    batch = {
        "input_ids": torch.randn(2, 10),
        "attention_mask": torch.randn(2, 10),
        "label": torch.tensor([0, 1])
    }
    
    with patch('src.trainer.trainer.Evaluator') as mock_eval_cls, \
         patch('src.trainer.trainer.DataLoader'), \
         patch('src.trainer.trainer.os.remove') as mock_remove, \
         patch('src.trainer.trainer.os.path.exists') as mock_exists, \
         patch('src.trainer.trainer.os.makedirs'):
         
        mock_eval = MagicMock()
        # Accuracy improves 0.5 -> 0.6 -> 0.7
        # best_model_path will be set after 0.5
        # when 0.6 comes, it should delete old best_model_path
        mock_eval.accuracy.side_effect = [(0.5, 0.5, 0.5), (0.6, 0.6, 0.6), (0.7, 0.7, 0.7)] 
        mock_eval_cls.return_value = mock_eval
        
        # Mock exists to True so it tries to delete
        mock_exists.return_value = True
        
        # Mock torch functions
        with patch('torch.save'), \
             patch('torch.nn.CrossEntropyLoss') as mock_loss_cls, \
             patch('torch.load'):
            
            # Setup loss to return a float item()
            mock_criterion = MagicMock()
            mock_criterion.return_value.item.return_value = 0.1
            mock_loss_cls.return_value = mock_criterion
            
            trainer = Trainer(
                device=torch.device('cpu'),
                model=mock_model,
                train_data=[1],
                val_data=[],
                num_epochs=2, # Need at least 2 epochs to trigger delete
                model_save_dir=str(tmp_path)
            )
            trainer.train_loader = [batch]
            
            trainer.train_model()
            
            # Should have tried to remove old model
            assert mock_remove.called

def test_trainer_no_improvement(mock_model, tmp_path):
    """Test case where validation accuracy decreases"""
    mock_model.get_name.return_value = "bert"
    mock_model.parameters.return_value = [torch.nn.Parameter(torch.tensor([1.0]))]
    mock_model.return_value = torch.randn(2, 2, requires_grad=True)
    
    batch = {
        "input_ids": torch.randn(2, 10),
        "attention_mask": torch.randn(2, 10),
        "label": torch.tensor([0, 1])
    }
    
    with patch('src.trainer.trainer.Evaluator') as mock_eval_cls, \
         patch('src.trainer.trainer.DataLoader'), \
         patch('torch.save') as mock_save, \
         patch('torch.nn.CrossEntropyLoss') as mock_loss_cls, \
         patch('src.trainer.trainer.os.remove'), \
         patch('torch.load'), \
         patch('src.trainer.trainer.os.makedirs'):
         
        mock_eval = MagicMock()
        # Accuracy decreases 0.5 -> 0.4
        mock_eval.accuracy.side_effect = [(0.5, 0.5, 0.5), (0.4, 0.4, 0.4)] 
        mock_eval_cls.return_value = mock_eval
        
        # Fix loss mocking
        mock_criterion = MagicMock()
        mock_loss_tensor = MagicMock()
        mock_loss_tensor.item.return_value = 0.1
        mock_loss_tensor.backward = MagicMock()
        mock_criterion.return_value = mock_loss_tensor
        mock_loss_cls.return_value = mock_criterion
        
        trainer = Trainer(
            device=torch.device('cpu'),
            model=mock_model,
            train_data=[1],
            val_data=[],
            num_epochs=2,
            model_save_dir=str(tmp_path)
        )
        trainer.train_loader = [batch]
        
        trainer.train_model()
        
        # Save should be called once for 0.5, but NOT for 0.4
        assert mock_save.call_count == 1

def test_trainer_load_best_model(mock_model, tmp_path):
    """Test loading best model at end"""
    mock_model.get_name.return_value = "bert"
    mock_model.parameters.return_value = [torch.nn.Parameter(torch.tensor([1.0]))]
    mock_model.return_value = torch.randn(2, 2, requires_grad=True)
    
    batch = {
        "input_ids": torch.randn(2, 10),
        "attention_mask": torch.randn(2, 10),
        "label": torch.tensor([0, 1])
    }
    
    with patch('src.trainer.trainer.Evaluator') as mock_eval_cls, \
         patch('src.trainer.trainer.DataLoader'), \
         patch('torch.save'), \
         patch('torch.nn.CrossEntropyLoss') as mock_loss_cls, \
         patch('torch.load') as mock_load, \
         patch('src.trainer.trainer.os.makedirs'):
         
        mock_eval = MagicMock()
        mock_eval.accuracy.return_value = (0.9, 0.9, 0.9)
        mock_eval_cls.return_value = mock_eval
        
        mock_criterion = MagicMock()
        mock_loss_tensor = MagicMock()
        mock_loss_tensor.item.return_value = 0.1
        mock_criterion.return_value = mock_loss_tensor
        mock_loss_cls.return_value = mock_criterion
        
        # Mock loaded state dict
        mock_load.return_value = {'state': 'dict'}
        
        trainer = Trainer(
            device=torch.device('cpu'),
            model=mock_model,
            train_data=[1],
            val_data=[],
            num_epochs=1,
            model_save_dir=str(tmp_path)
        )
        trainer.train_loader = [batch]
        
        trainer.train_model()
        
        # Should verify it loaded the model
        mock_load.assert_called()
        mock_model.load_state_dict.assert_called_with({'state': 'dict'})

def test_trainer_no_save(mock_model, tmp_path):
    """Test case where validation accuracy decreases"""
    mock_model.get_name.return_value = "bert"
    mock_model.parameters.return_value = [torch.nn.Parameter(torch.tensor([1.0]))]
    mock_model.return_value = torch.randn(2, 2, requires_grad=True)
    
    batch = {
        "input_ids": torch.randn(2, 10),
        "attention_mask": torch.randn(2, 10),
        "label": torch.tensor([0, 1])
    }
    
    with patch('src.trainer.trainer.Evaluator') as mock_eval_cls, \
         patch('src.trainer.trainer.DataLoader'), \
         patch('torch.save') as mock_save, \
         patch('torch.nn.CrossEntropyLoss') as mock_loss_cls, \
         patch('torch.load') as mock_load, \
         patch('src.trainer.trainer.os.makedirs'):
         
        mock_eval = MagicMock()
        # Accuracy decreases: 0.5 (saves), then 0.4 (doesn't save)
        mock_eval.accuracy.side_effect = [(0.5, 0.5, 0.5), (0.4, 0.4, 0.4)]
        mock_eval_cls.return_value = mock_eval
        
        mock_criterion = MagicMock()
        mock_loss_tensor = MagicMock()
        mock_loss_tensor.item.return_value = 0.1
        mock_criterion.return_value = mock_loss_tensor
        mock_loss_cls.return_value = mock_criterion
        
        trainer = Trainer(
            device=torch.device('cpu'),
            model=mock_model,
            train_data=[1],
            val_data=[],
            num_epochs=2,
            model_save_dir=str(tmp_path)
        )
        trainer.train_loader = [batch]
        
        trainer.train_model()
        
        # Should save exactly once (for the first epoch)
        assert mock_save.call_count == 1
        # Should NOT load (only loads if best path exists at end, which it does, but we mocked load)
        # Actually train_model attempts to load best model at end.
        assert mock_load.called

def test_trainer_scheduler_step(mock_model, tmp_path):
    """Verify scheduler step is called for BERT"""
    mock_model.get_name.return_value = "bert"
    mock_model.parameters.return_value = [torch.nn.Parameter(torch.tensor([1.0]))]
    mock_model.return_value = torch.randn(2, 2, requires_grad=True)
    
    batch = {
        "input_ids": torch.randn(2, 10),
        "attention_mask": torch.randn(2, 10),
        "label": torch.tensor([0, 1])
    }
    
    with patch('src.trainer.trainer.Evaluator') as mock_eval_cls, \
         patch('src.trainer.trainer.DataLoader'), \
         patch('src.trainer.trainer.get_linear_schedule_with_warmup') as mock_get_schedule, \
         patch('torch.optim.AdamW'), \
         patch('torch.nn.CrossEntropyLoss') as mock_loss_cls, \
         patch('src.trainer.trainer.os.makedirs'), \
         patch('torch.save'), \
         patch('torch.load'), \
         patch('src.trainer.trainer.os.remove'):
         
        mock_eval = MagicMock()
        mock_eval.accuracy.return_value = (0.5, 0.5, 0.5) # Return tuple
        mock_eval_cls.return_value = mock_eval

        mock_scheduler = MagicMock()
        mock_get_schedule.return_value = mock_scheduler
        
        # Configure loss to return a float
        mock_loss_instance = MagicMock()
        mock_loss_instance.item.return_value = 0.1
        # Also backward() needs to exist
        mock_loss_instance.backward = MagicMock()
        
        # CrossEntropyLoss() returns the loss function (callable)
        # Calling the loss function returns the loss tensor/object
        # So: criterion = CrossEntropyLoss() -> criterion is mock_loss_fn
        # criterion(outputs, labels) -> returns mock_loss_instance
        mock_loss_fn = MagicMock()
        mock_loss_fn.return_value = mock_loss_instance
        mock_loss_cls.return_value = mock_loss_fn
        
        trainer = Trainer(
            device=torch.device('cpu'),
            model=mock_model,
            train_data=[1],
            val_data=[],
            num_epochs=1,
            model_save_dir=str(tmp_path)
        )
        trainer.train_loader = [batch]
        
        trainer.train_model()
        
        # Verify scheduler step called
        assert mock_scheduler.step.called
