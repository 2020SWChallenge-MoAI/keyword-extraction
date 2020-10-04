from typing import *

class Sentence(object):
    def __init__(self, id=None, raw: str = None):
        self.id = id
        self._raw = raw
        self._tokens = []
        self._analyzed = False

    def set_tokens(self, tokens):
        self._tokens = tokens
        self._analyzed = True

    def text(self):
        return self._raw

    def tokens(self):
        if not self._analyzed:
            raise Exception(f'sentence is not analyzed.')

        return self._tokens


class Document(object):
    TEMP_ID = -1

    def __init__(self, id=None, sentences: List[Sentence] = None):
        self.id = id
        self.sentences: List[Sentence] = sentences

    def text(self):
        return ' '.join([sent.text() for sent in self.sentences])

    def tokens(self):
        return sum([sent.tokens() for sent in self.sentences], [])
