import ast
import re

def convert_indices(data, model_w2v):
    result = []
    for words, label in data:
        lst = ast.literal_eval(words)
        word = [w for w in lst if re.match(r'^[A-Za-z]+$', w)]
        indices = []
        for w in word:
            if w in model_w2v.wv.key_to_index:
                # Shift indices up by one since the padding token is at index 0
                indices.append(model_w2v.wv.key_to_index.get(w) + 1) 
            else:
                indices.append(0)
        result.append((indices, label),)
    return result
