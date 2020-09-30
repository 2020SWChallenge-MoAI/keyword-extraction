from abc import ABCMeta, abstractmethod


class Context(metaclass=ABCMeta):
    @abstractmethod
    def import_model(self, model: bytes) -> None:
        pass

    @abstractmethod
    def export_model(self) -> bytes:
        pass


from .tfidf import TfidfContext
from .ner import NerContext
from .word2vec import Word2VecContext