import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from utils.constants import PHI2_MODEL_ID
from utils.data_utils import get_train_val_dataset, train_collate_fn
import sys
from tqdm.auto import tqdm

from peft import LoraConfig, get_peft_model

model =  AutoModelForCausalLM.from_pretrained(
    PHI2_MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32
)
# print(model)
# print("\n\n\n")
# print(model.state_dict().keys())
# print(model.model.layers[0].mlp.config)
# add_adapter(model, rank=16)
# sys.exit(1)
device = "cuda:0"
model.to(device)


train_dataset, val_dataset = get_train_val_dataset()
train_dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=train_collate_fn, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=4, collate_fn=train_collate_fn, shuffle=False)

def get_training_parameters(model):
    params = []
    for name, param in model.named_parameters():
        if "lm_head" in name:
            params += [param]
            param.requires_grad = True
        else:
            param.requires_grad = False
    return params


config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    # modules_to_save=["lm_head"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

for epoch in range(3):
    progress_bar = tqdm(range(len(train_dataloader)))
    progress_bar.desc = "Train"
    train_loss = 0
    model.train()
    for i,batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        mask = batch["output_masks"].to(model.device)
        p_mask = batch["padding_masks"].to(model.device)
        logits = model(batch["input_ids"].to(model.device), attention_mask=p_mask).logits
        B, T, C = logits.shape
        loss = loss_fn(logits.view(-1, C), batch["output_ids"].to(model.device).view(-1))
        loss = torch.sum(loss * p_mask.view(-1))/torch.sum(p_mask)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        train_loss = train_loss +(1/(i+1)) * (loss.item()-train_loss)
        progress_bar.set_postfix({"loss": f"{train_loss:.4f}"})
        progress_bar.update(1)
        # if i > 50:
        #     break
            
        
    progress_bar.close()

    tqdm.write("Validation")
    progress_bar = tqdm(range(len(val_dataloader)))
    progress_bar.desc = "Val"
    model.eval()
    with torch.inference_mode():
        val_loss = 0
        for i,batch in enumerate(val_dataloader):
            mask = batch["output_masks"].to(model.device)
            p_mask = batch["padding_masks"].to(model.device)
            logits = model(batch["input_ids"].to(model.device), attention_mask=p_mask).logits
            B, T, C = logits.shape
            loss = loss_fn(logits.view(-1, C), batch["output_ids"].to(model.device).view(-1))
            loss = torch.sum(loss * p_mask.view(-1))/torch.sum(p_mask)
            val_loss = val_loss + (1/(i+1))*(loss.item()-val_loss)
            progress_bar.set_postfix({"loss": f"{val_loss:.4f}"})
            progress_bar.update(1)

            if i%20 == 0:
                # generate
                x = batch["input_ids"][0]
                valid_idxs = batch["output_masks"][0].nonzero()[0] 
                end_idx = valid_idxs[-1]
                start_idx = valid_idxs[0]
                targ = batch["output_ids"][0][start_idx-1:]
                quest = x[:start_idx]

                tqdm.write("--------------------------------------------------")
                tqdm.write(val_dataset.tokenizer.decode(quest, skip_special_tokens=True))
                tqdm.write(val_dataset.tokenizer.decode(targ, skip_special_tokens=True))
                tqdm.write("Prediction:")
                x_ = quest.unsqueeze(0).to(model.device)
                topk = 50
                new_tokens = []
                while x_.size(1) <= len(x):
                    logits = model(x_).logits
                    l_logits = logits[:, -1, :]
                    probs = l_logits.softmax(dim=-1)
                    t_p, t_i = probs.topk(topk, dim=-1)
                    idx = torch.multinomial(t_p, num_samples=1)
                    ix = torch.gather(t_i, dim=-1, index=idx)
                    x_ = torch.cat((x_, ix), dim=1)
                    new_tokens += [ix.item()]
                    if ix.item() == val_dataset.tokenizer.eos_token_id:
                        break
                tqdm.write(val_dataset.tokenizer.decode(new_tokens, skip_special_tokens=True))
            
        progress_bar.close()


        
    


