import re
from nltk.stem import WordNetLemmatizer


# Tokenizer that will also lemmatize
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        tokens = re.findall(r'(\b\w\w+\b)', articles)
        return [self.wnl.lemmatize(t) for t in tokens]
