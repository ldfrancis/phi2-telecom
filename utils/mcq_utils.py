import re
from tqdm.auto import tqdm


def obtain_option(generation):
    answer = generation.split("Answer:")[1]
    match = re.search("\d+", answer)
    if not match:
        return 1
    return match.group()


def evaluate_mcqs(mcqs, model, tokenizer, rag=False):
    correct_preds = 0
    total = 0
    score = 0
    progbar = tqdm(range(len(mcqs)))
    progbar.desc = "MCQ Eval"
    question_key = "question_context" if rag else "question"
    wrong_answers = []
    for mcq in mcqs:
        total += 1
        question = mcq[question_key]
        question_tokens = tokenizer(question, return_tensors="pt")
        generation_tokens = model.generate(**question_tokens.to(model.device), max_new_tokens=30).squeeze().tolist()
        generation = tokenizer.decode(generation_tokens)
        pred_option = int(obtain_option(generation))
        correct_option = mcq["answer_option"]
        correct_preds += (correct_option == pred_option)
        score = correct_preds/total
        progbar.set_postfix({"score":f"{score:.4f}"})
        if correct_option != pred_option:
            wrong_answers += [(mcq, pred_option, generation)]
        progbar.update(1)
    progbar.close()
    return score, wrong_answers

def answer_mcqs(mcqs, model, tokenizer, rag=False):
    progbar = tqdm(range(len(mcqs)))
    progbar.desc = "MCQ Eval"
    question_key = "question_context" if rag else "question"
    for mcq in mcqs:
        question = mcq[question_key]
        question_tokens = tokenizer(question, return_tensors="pt")
        generation_tokens = model.generate(**question_tokens.to(model.device), max_new_tokens=30).squeeze().tolist()
        generation = tokenizer.decode(generation_tokens)
        pred_option = int(obtain_option(generation))
        mcq["pred_option"] = pred_option
        progbar.update(1)
    progbar.close()


