import re

def normalize(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def check_output(out1, out2):
    if normalize(out1) == normalize(out2):
        return "exact_match"

    if normalize(out1) in normalize(out2) or normalize(out2) in normalize(out1):
        return "partial_match"

    return "different"

def token_match_rate(tokenizer, out1, out2):

    t1 = tokenizer.encode(out1)
    t2 = tokenizer.encode(out2)

    min_len = min(len(t1), len(t2))

    match = 0
    for i in range(min_len):
        if t1[i] == t2[i]:
            match += 1

    return match / max(len(t1), len(t2))

def debug_token_diff(tokenizer, out1, out2):

    t1 = tokenizer.encode(out1)
    t2 = tokenizer.encode(out2)

    for i,(a,b) in enumerate(zip(t1,t2)):
        if a != b:
            print("diff at position", i)
            print("token1:", a, tokenizer.decode([a]))
            print("token2:", b, tokenizer.decode([b]))
            break

def compare_tokens(tokenizer, out1, out2):

    t1 = tokenizer.encode(out1)
    t2 = tokenizer.encode(out2)

    min_len = min(len(t1), len(t2))

    for i in range(min_len):
        if t1[i] != t2[i]:
            return {
                "match": False,
                "diff_position": i,
                "token1": t1[i],
                "token2": t2[i]
            }

    if len(t1) != len(t2):
        return {
            "match": False,
            "diff_position": min_len,
            "reason": "length_different"
        }

    return {"match": True}