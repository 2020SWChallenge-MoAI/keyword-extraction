import requests
from collections import defaultdict

from . import Context
from ..util import token_preprocess
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

    def get_keywords(self, document: Document):
        keywords = []

        res = requests.post(API_SERVER_URL, json={ 'text': document.text() })
        if res.ok:
            keywords = res.json()['ners']
            keywords = [(token_preprocess(word), 1) for tag, word in keywords]

        return keywords