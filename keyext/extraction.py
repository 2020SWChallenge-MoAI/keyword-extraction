import pickle
import os

from konlpy.tag import Komoran
from sklearn.feature_extraction.text import TfidfVectorizer


class KeywordExtractor(object):
    def __init__(self, model_path):
        super().__init__()

        self.tokenizer = Tokenizer()

        if model_path is not None:
            with open(model_path, 'rb') as f:
                self.documents = pickle.load(f)
                self.vocab = pickle.load(f)

    def build(self, documents):
        pass

    def recommend(self, document_id, num=2, exclude_keywords=[]):
        pass

    def listall(self):
        pass

    def export(self, model_path):
        pass


class Tokenizer(object):
    def __init__(self, tagger=Komoran(), noun_only=True):
        self.tagger = tagger

    def __call__(self, sent):
        pos = self.tagger.pos(sent)
        pos = ['{}/{}'.format(word, tag) for word, tag in pos if not noun_only or tag.startswith('NN')]
        return pos
