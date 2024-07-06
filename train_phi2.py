import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from utils.constants import PHI2_MODEL_ID
from utils.data_utils import get_train_val_dataset, train_collate_fn


model =  AutoModelForCausalLM.from_pretrained(
    PHI2_MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32
)

train_dataset, val_dataset = get_train_val_dataset()
train_dataloader = DataLoader(train_dataset, batch_size=8, collate_fn=train_collate_fn, shuffle=False)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
device = "cuda:0"
model.to(device)
for batch in train_dataloader:
    print(batch["input_ids"][0][:20])
    model(batch["input_ids"].to(model.device))
    break


