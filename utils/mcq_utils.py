import math
import re
from tqdm.auto import tqdm
import torch
import json
from utils.data_utils import prompt_w_context, clean_question
import random
from itertools import permutations


def obtain_option(generation):
    answer = generation.split("Output:")[1]
    match = re.search("\d+", answer)
    if not match:
        return 1
    return match.group()

def get_option(question, model, tokenizer, option_ids):
    question_tokens = tokenizer(question, return_tensors="pt")
    logits = model(**question_tokens.to(model.device)).logits[:,-1,option_ids].squeeze()
    pred_option = int(["1", "2", "3", "4", "5"][logits.argmax().item()])
    return pred_option, (logits, question_tokens)

def evaluate_mcqs(mcqs, model, tokenizer, rag=False):
    correct_preds = 0
    total = 0
    score = 0
    progbar = tqdm(range(len(mcqs)))
    progbar.desc = "MCQ Eval"
    question_key = "question_context" if rag else "question"
    option_ids = [tokenizer(o).input_ids[0] for o in ["1", "2", "3", "4", "5"]]
    wrong_answers = []
    model.eval()
    cache = []
    with torch.inference_mode():
        for mcq in mcqs:
            total += 1
            question = mcq[question_key]
            pred_option, logits = get_option(question, model, tokenizer, option_ids)
            # correct_option = mcq["answer"]
            # question_tokens = tokenizer(question, return_tensors="pt")
            # generation_tokens = model.generate(**question_tokens.to(model.device), max_new_tokens=30).squeeze().tolist()
            # generation = tokenizer.decode(generation_tokens)\
            # # breakpoint()
            # pred_option = int(obtain_option(generation))
            correct_option = mcq["answer_option"]
            # breakpoint()
            correct_preds += (correct_option == pred_option)
            score = correct_preds/total
            progbar.set_postfix({"score":f"{score:.4f}"})
            if correct_option != pred_option:
                wrong_answers += [(mcq, pred_option)]
            progbar.update(1)
            # cache += [(pred_option, logits[0].item())]
            # json.dump(logits[-1].input_ids.tolist(), open("txt","w"))
            # json.dump([question], open("qtxt","w"))
        progbar.close()
    return score, wrong_answers

def evaluate_mcqs2(mcqs, model, tokenizer, rag=False):
    chunk_idxs = [i for i in range(4)]
    topk = 2
    num_correct = 0
    total = 0
    option_ids = [tokenizer(o).input_ids[0] for o in ["1", "2", "3", "4", "5"]]
    pbar = tqdm(range(len(mcqs)), ncols=100)
    tokenizer.padding_side = "left"
    model.eval()
    bs = 16

    examples = []
    for example in mcqs:
        total += 1
        options = example["options"].copy()
        # correct_option_idx = example["correct_option_idx"]
        # correct_option_txt = options.pop(correct_option_idx)
        opt_poses = []
        qs = []

        perms = list(permutations(range(len(options))))
        random.shuffle(perms)
        for opt_pos in perms[:bs]:
            _options = [options[o] for o in opt_pos]
            opt_poses += [opt_pos]
            
            random.shuffle(chunk_idxs)
            options_txt = "\n".join(
                [
                    f"{i+1}) {val[:-1] if val[-1] == '.' else val}" for i, val in enumerate(_options)
                ]
            )
            context = "\n".join([example["chunks"][chunk_idxs[i]] for i in range(topk)])
            prompt = prompt_w_context.format(
                choices=f"({','.join([f"{i+1}" for i in range(len(options))])})",
                question=clean_question(example["question"]),
                options=options_txt,
                context=context
            )
            question_context = f"{prompt}"
            qs += [question_context]

        examples += [{"questions": qs, "answer_mappings":opt_poses, "correct_option_idx": example["correct_option_idx"]}]
            
    for example in examples:
        chosen_options = {}
        correct_option = example["correct_option_idx"]
        preds = []
        samples = math.ceil(len(qs)/bs)
        opt_poses = example["answer_mappings"]
        qs = example["questions"]
        for i in range(samples):
            s_idx = i*bs
            e_idx = s_idx+bs
            x = qs[s_idx:e_idx]
            tokens = tokenizer(x, return_tensors="pt", padding="longest").to(model.device)
            with torch.inference_mode():
                preds += model(**tokens).logits[:,-1,option_ids].argmax(dim=1).tolist()

        for pred_op, map_ops in zip(preds, opt_poses):
            chosen_options[map_ops[pred_op]] = chosen_options.get(map_ops[pred_op], 0) + 1

        sorted_choices = sorted([(k,v) for k,v in chosen_options.items()], key=lambda x: x[1])
        most_pred_option = sorted_choices[-1][0]
        num_correct += most_pred_option == correct_option
        score = num_correct/total
        pbar.set_description(f"Score: {score}")
        pbar.update(1)

    pbar.close()

        

def answer_mcqs(mcqs, model, tokenizer, rag=False):
    progbar = tqdm(range(len(mcqs)))
    progbar.desc = "MCQ Eval"
    question_key = "question_context" if rag else "question"
    option_ids = [tokenizer(o).input_ids[0] for o in ["1", "2", "3", "4", "5"]]
    for mcq in mcqs:
        question = mcq[question_key]
        # question_tokens = tokenizer(question, return_tensors="pt")
        # generation_tokens = model.generate(**question_tokens.to(model.device), max_new_tokens=30).squeeze().tolist()
        # generation = tokenizer.decode(generation_tokens)
        # pred_option = int(obtain_option(generation))
        pred_option, logits = get_option(question, model, tokenizer, option_ids)
        mcq["pred_option"] = pred_option
        progbar.update(1)
    progbar.close()


