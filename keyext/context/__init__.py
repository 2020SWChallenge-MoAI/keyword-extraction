from abc import ABCMeta, abstractmethod


class Context(metaclass=ABCMeta):
    @abstractmethod
    def import_model(self, f) -> None:
        pass

    @abstractmethod
    def export_model(self, f) -> None:
        pass


from .tfidf import TfidfContext
from .ner import NerContext
from .word2vec import Word2VecContext