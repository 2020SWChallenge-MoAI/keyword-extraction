from pickle import TRUE
import re
import requests
import pickle
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

    def import_model(self, f) -> None:
        self.contexts = pickle.load(f)
        self._initialized = True

    def export_model(self, f) -> None:
        if not self._initialized:
            raise Exception('model is not initialized. nothing to export.')

        pickle.dump(self.contexts, f)
    
    @staticmethod
    def request_api(document: str):
        keywords = []
        res = requests.post(API_SERVER_URL, json={ 'text': document})
        if res.ok:
            keywords = res.json()['ners']

        return keywords

    def get_keywords(self, document: Document) -> List[Tuple[str,str]]:
        if self._initialized and document.id in self.contexts:
            return self.contexts[document.id]['all']
        
        else: # API request
            return [(tag, word) for (tag, word) in self.request_api(document.text())]

    def get_related_keywords(self, document: Document, queries: List[str]) -> List[Tuple[str,str]]:
        queries = [re.sub('[^\w]', '', x) for x in queries]
        sentences = [re.sub('[^\w]', '', sentence.text()) for sentence in document.sentences]

        related_sentence_ids = []

        # find all sentences containing all queries
        for sentence_id, sentence in enumerate(sentences): 
            if all(x in sentence for x in queries):
                related_sentence_ids.append(sentence_id)
        
        # find all sentences containing last queries if there are no sentences
        if len(related_sentence_ids)==0 or (len(queries)>=3 and len(related_sentence_ids)<=5):
            for sentence_id, sentence in enumerate(sentences):
                if queries[-1] in sentence:
                    related_sentence_ids.append(sentence_id)
        
        if self._initialized and document.id in self.contexts:
            ners = sum([self.contexts[document.id]['by_sentence'][sentence_id] for sentence_id in related_sentence_ids],[])
        else:
            all_sentences = "\n".join([document.sentences[sentence_id].text() for sentence_id in related_sentence_ids])
            ners = self.request_api(all_sentences)

        return [(tag, word) for tag, word in ners if re.sub('[^\w]','', word) not in queries]

