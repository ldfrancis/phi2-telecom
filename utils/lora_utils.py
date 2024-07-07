# LoRAParametrization Obtained from 
# https://github.com/hkproj/pytorch-lora/blob/main/lora.ipynb

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize


class LoRAParametrization(nn.Module):
    def __init__(self, features_in, features_out, rank=1, alpha=1, device='cpu'):
        super().__init__()
        # Section 4.1 of the paper: 
        #   We use a random Gaussian initialization for A and zero for B, so ∆W = BA 
        #   is zero at the beginning of training
        self.lora_A = nn.Parameter(torch.zeros((rank,features_out)).to(device))
        self.lora_B = nn.Parameter(torch.zeros((features_in, rank)).to(device))
        nn.init.normal_(self.lora_A, mean=0, std=1)
        
        # Section 4.1 of the paper: 
        #   We then scale ∆Wx by α/r , where α is a constant in r. 
        #   When optimizing with Adam, tuning α is roughly the same as tuning the 
        #   learning rate if we scale the initialization appropriately. 
        #   As a result, we simply set α to the first r we try and do not tune it. 
        #   This scaling helps to reduce the need to retune hyperparameters when we
        #   vary r.
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, original_weights):
        if self.enabled:
            # Return W + (B*A)*scale
            return original_weights + torch.matmul(self.lora_B, self.lora_A).\
                view(original_weights.shape) * self.scale
        else:
            return original_weights


def linear_layer_parametrization(layer, rank=1, lora_alpha=1):
    # Only add the parameterization to the weight matrix, ignore the Bias

    # From section 4.2 of the paper:
    #   We limit our study to only adapting the attention weights for downstream 
    #   tasks and freeze the MLP modules (so they are not trained in downstream tasks) 
    #   both for simplicity and parameter-efficiency.
    #   [...]
    #   We leave the empirical investigation of [...], and biases to a future work.
    
    features_in, features_out = layer.weight.shape
    device = layer.weight.device
    return LoRAParametrization(
        features_in, features_out, rank=rank, alpha=lora_alpha, device=device
    )


def lora_parametrize(module, rank=1, name="weight"):
    parametrize.register_parametrization(
        module, name, 
        linear_layer_parametrization(module, rank=rank, lora_alpha=rank)
    )


def add_adapter(model, rank, target_modules=["q_proj","k_proj","v_proj"]):
    for name, module in model.named_children():
        if name in target_modules:
            lora_parametrize(module, rank)
            module.parametrizations["weight"][0].enabled = True
        else:
            add_adapter(module, rank, target_modules)

def disable_adapter(model, idx=0):
    for name, module in model.named_children():
        if hasattr(module, "parametrizations"):
            module.parametrizations["weight"][idx].enabled = False
        else:
            disable_adapter(module, idx=idx)

def enable_adapter(model, idx=0):
    for name, module in model.named_children():
        if hasattr(module, "parametrizations"):
            module.parametrizations["weight"][0].enabled = True
        else:
            enable_adapter(module, idx=idx)

def freeze_model(model):
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False



if __name__=="__main__":

    class PhiMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.activation_fn = nn.ReLU()
            self.fc1 = nn.Linear(2560, 10240)
            self.fc2 = nn.Linear(10240, 2560)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            hidden_states = self.fc1(hidden_states)
            hidden_states = self.activation_fn(hidden_states)
            hidden_states = self.fc2(hidden_states)
            return hidden_states


    inp = torch.randn((4, 2560)).float()
    model = PhiMLP()
    model(inp)

    original_weights = {}
    for name, param in model.named_parameters():
        original_weights[name] = param.clone().detach()

    total_parameters_original = 0
    for index, (name,layer) in enumerate(model.named_children()):
        if "fc" in name:
            total_parameters_original += layer.weight.nelement() + layer.bias.nelement()
            print(f'Layer {index+1}: W: {layer.weight.shape} + B: {layer.bias.shape}')
    print(f'Total number of parameters: {total_parameters_original:,}')

    add_adapter(model, 32, target_modules=["fc1", "fc2"])

    total_parameters_lora = 0
    total_parameters_non_lora = 0
    for index, (name,layer) in enumerate(model.named_children()):
        if "fc" in name:
            total_parameters_lora += layer.parametrizations["weight"][0]\
                .lora_A.nelement() + layer.parametrizations["weight"][0]\
                    .lora_B.nelement()
            total_parameters_non_lora += layer.weight.nelement() + layer\
                .bias.nelement()
            print(
                f'Layer {index+1}: W: {layer.weight.shape} + B: {layer.bias.shape}\
                      + Lora_A: {layer.parametrizations["weight"][0].lora_A.shape} \
                        + Lora_B: {layer.parametrizations["weight"][0].lora_B.shape}'
            )
    # The non-LoRA parameters count must match the original network
    assert total_parameters_non_lora == total_parameters_original
    print(f'Total number of parameters (original): {total_parameters_non_lora:,}')
    print(f'Total number of parameters (original + LoRA): {total_parameters_lora + total_parameters_non_lora:,}')
    print(f'Parameters introduced by LoRA: {total_parameters_lora:,}')
    parameters_incremment = (total_parameters_lora / total_parameters_non_lora) * 100
    print(f'Parameters incremment: {parameters_incremment:.3f}%')

    freeze_model(model)

    optim = torch.optim.Adam(model.parameters())
    optim.zero_grad()
    logits = model(inp)
    loss = logits.mean()
    loss.backward()
    optim.step()

    assert torch.all(model.fc1.parametrizations.weight.original == original_weights['fc1.weight'])
    assert torch.all(model.fc2.parametrizations.weight.original == original_weights['fc2.weight'])
    
    enable_adapter(model)
    assert torch.equal(model.fc1.weight, model.fc1.parametrizations.weight.original + (model.fc1.parametrizations.weight[0].lora_B @ model.fc1.parametrizations.weight[0].lora_A) * model.fc1.parametrizations.weight[0].scale)
    
    disable_adapter(model)
    assert torch.equal(model.fc1.weight, original_weights['fc1.weight'])
