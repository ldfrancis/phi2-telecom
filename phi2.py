import torch
import torch.nn as nn
from transformers.activations import NewGELUActivation
from utils.lora_utils import add_adapter, disable_adapter, enable_adapter, freeze_model

PhiConfig = {
  "_name_or_path": "microsoft/phi-2",
  "architectures": [
    "PhiForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 50256,
  "embd_pdrop": 0.0,
  "eos_token_id": 50256,
  "hidden_act": "gelu_new",
  "hidden_size": 2560,
  "initializer_range": 0.02,
  "intermediate_size": 10240,
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 2048,
  "model_type": "phi",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "partial_rotary_factor": 0.4,
  "qk_layernorm": False,
  "resid_pdrop": 0.1,
  "rope_scaling": None,
  "rope_theta": 10000.0,
  "tie_word_embeddings": False,
  "torch_dtype": "float16",
  "transformers_version": "4.42.0.dev0",
  "use_cache": True,
  "vocab_size": 51200
}


class PhiSdpaAttention:
    def __init__(self) -> None:
        pass



class PhiMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation_fn = NewGELUActivation()
        self.fc1 = nn.Linear(2560, 10240)
        self.fc2 = nn.Linear(10240, 2560)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    

class PhiDecoderLayer(nn.Module):
    


class PhiForCausalLM:
    def __init__(self) -> None:
        self.model = None
        self.lm_head = None


