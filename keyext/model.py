from typing import *

class Sentence(object):
    def __init__(self, id=None, raw: str = None):
        self.id = id
        self.raw = raw
        
        self._pos_tokens = []
        self._text_tokens = []
        self._analyzed = False

    def set_tokens(self, pos_tokens, text_tokens):
        self._pos_tokens = pos_tokens
        self._text_tokens = text_tokens

        self._analyzed = True

    def text(self):
        return self.raw

    def pos_tokens(self):
        if not self._analyzed:
            raise Exception(f'sentence is not analyzed.')

        return self._pos_tokens

    def text_tokens(self):
        if not self._analyzed:
            raise Exception(f'sentence is not analyzed.')

        return self._text_tokens


class Document(object):
    def __init__(self, id=None, sentences: List[Sentence] = None):
        self.id = id
        self.sentences: List[Sentence] = sentences

    def text(self):
        return ' '.join([sent.text() for sent in self.sentences])

    def pos_tokens(self):
        return sum([sent.pos_tokens() for sent in self.sentences], [])

    def text_tokens(self):
        return sum([sent.text_tokens() for sent in self.sentences], [])