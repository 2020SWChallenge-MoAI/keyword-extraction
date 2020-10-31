import re
import requests
from typing import *
from collections import defaultdict

from . import Context
from ..util import token_preprocess, preprocess
from ..model import Document


API_SERVER_URL = 'http://127.0.0.1:9002'

class NerContext(Context):
    def __init__(self):
        super().__init__()
        self._initialized = False

    def import_model(self, model: bytes) -> None:
        pass

    def export_model(self) -> bytes:
        pass

    def get_keywords(self, document: Document) -> List[Tuple[str,str]]:
        keywords = []

        res = requests.post(API_SERVER_URL, json={ 'text': document.text() })
        if res.ok:
            keywords = res.json()['ners']

        return [(tag, word) for tag, word in keywords]

    def get_related_keywords(self, document: Document, queries: List[str]) -> List[Tuple[str,str]]:
        related_sentences = []

        # find all sentences containing all queries
        for sentence in document.sentences:
            if all(x in re.sub('[^\w]','',sentence.text()) for x in queries):
                related_sentences.append(sentence.text())

        # find all sentences containing last queries if there are no sentences
        if len(related_sentences)==0 or (len(queries)>=3 and len(related_sentences)<=5): 
            for sentence in document.sentences:
                if any(x in re.sub('[^\w]','',sentence.text()) for x in queries[-1]):
                    related_sentences.append(sentence.text())

        related_sentences = "\n".join(related_sentences)

        keywords = []
        res = requests.post(API_SERVER_URL, json={ 'text': related_sentences })
        if res.ok:
            keywords = res.json()['ners']

        return [(tag, word) for tag, word in keywords if word not in queries]
        

