import random
import re

PREFIX_TOKENS = 8192
LETTERS = "ABCD"
HEADER = ("<|im_start|>user\nRead the document below, then answer the multiple-choice question that follows."
          "\n\n<document>\n")
OPTIONS = "\n</document>\n\nQuestion: {question}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n\n"
RATIONALE = ("Explain your reasoning in at most three sentences, then give your final answer on its own line "
             "in the form \"Answer: <letter>\".<|im_end|>\n<|im_start|>assistant\n")
LETTER_ONLY = "Respond with ONLY the answer letter (A, B, C, or D). No explanation.<|im_end|>\n<|im_start|>assistant\n"
ANSWER = re.compile(r"Answer:\s*\**\s*\(?([ABCD])\b")


def encode(tokenizer, text):
    return tokenizer(text, add_special_tokens=False).input_ids


def header_ids(tokenizer):
    return encode(tokenizer, HEADER)


def document_budget(tokenizer):
    return PREFIX_TOKENS - len(header_ids(tokenizer))


def prefix_ids(tokenizer, document_ids):
    header = header_ids(tokenizer)
    budget = PREFIX_TOKENS - len(header)
    if len(document_ids) < budget:
        raise ValueError(f"DOCUMENT_TOO_SHORT tokens={len(document_ids)} budget={budget}")
    return header + list(document_ids[:budget])


def question_text(item, letter_only=False):
    a, b, c, d = item["options"]
    return OPTIONS.format(question=item["question"], a=a, b=b, c=c, d=d) + (LETTER_ONLY if letter_only else RATIONALE)


def arrange_options(correct, distractors, index, seed):
    if len(distractors) != 3:
        raise ValueError(f"DISTRACTOR_COUNT {len(distractors)}")
    rng = random.Random(f"{seed}:{index}")
    shuffled = list(distractors)
    rng.shuffle(shuffled)
    position = index % 4
    options = shuffled[:position] + [correct] + shuffled[position:]
    return options, LETTERS[position]


def parse_letter(text):
    matches = ANSWER.findall(text)
    return matches[-1] if matches else None
