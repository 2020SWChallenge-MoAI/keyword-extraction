import pickle

from gensim.models import FastText
from typing import *

from . import Context
from ..model import *


class Word2VecContext(Context):
    def __init__(self):
        self._initialized = False

    def import_model(self, model: bytes):
        self.model = pickle.loads(model)
        self._initialized = True

    def export_model(self) -> bytes:
        return pickle.dumps(self.model)

    def build(self, documents: List[Document]):
        sentences = sum([doc.sentences for doc in documents], [])
        sentences = [sent.text_tokens() for sent in sentences]
        self.model = FastText(sentences, window=3, min_count=3, sg=1, iter=20, workers=10, word_ngrams=1)

        self._initialized = True

    def get_similarity(self, word1: str, word2: str) -> float:
        if not self._initialized:
            raise Exception('word2vec context is not initialized.')
        return self.model.similarity(word1, word2)

    def get_related_keywords(self, keywords, num=10):
        if not self._initialized:
            raise Exception('word2vec context is not initialized.')

        return self.model.most_similar(positive=keywords, topn=num)
