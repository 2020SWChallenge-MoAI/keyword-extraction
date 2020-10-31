import re

from collections import defaultdict
from konlpy.tag import Komoran

from ..model import *
from . import simple_preprocess


class PosTokenizer(object):

    def __init__(self, tagger=Komoran()):
        self.tagger = tagger

    def tokenize(self, sent):
        tokens = self.tagger.pos(simple_preprocess(sent))
        tokens = [f'{word}/{tag or "NA"}' for word, tag in tokens]

        return tokens

    @staticmethod
    def subtokens(token: str) -> Tuple[str]:
        tokens = re.findall(r'(?:[ㄱ-ㅎ가-힣A-Za-z0-9]+(?:\s[ㄱ-ㅎ가-힣A-Za-z0-9]+)*|_)\/\w+', token)
        return tuple([tuple(t.split('/')) for t in tokens])

    @staticmethod
    def word(token: str) -> str:
        return ''.join([word for word, tag in PosTokenizer.subtokens(token) if tag != 'NA'])

    @staticmethod
    def contains(token1: str, token2: str) -> bool:
        token1_subtokens = PosTokenizer.subtokens(token1)
        token2_subtokens = PosTokenizer.subtokens(token2)

        for word1, _ in token1_subtokens:
            for word2, _ in token2_subtokens:
                if word1 == word2:
                    return True

        return False

class PosValidator(object):
    VALID_TAGS = ('NNP', 'NNG', 'XR', 'S', 'NF')
    VALID_INDEPENDENT_TAGS = ('NNP', 'NNG', 'XR', 'S', 'NF')
    INVALID_TOKEN = '_/NA'

    @staticmethod
    def is_valid(token: str):
        subtokens = PosTokenizer.subtokens(token)
        if len(subtokens) == 1:  # single
            return subtokens[0][1].startswith(PosValidator.VALID_INDEPENDENT_TAGS)
        else:
            if not all([tag.startswith(PosValidator.VALID_TAGS) for _, tag in subtokens]):
                return False
            if len(set(subtokens)) <= 1:
                return False
        return True

    @staticmethod
    def filter(tokens: List[str]):
        return [token if PosValidator.is_valid(token) else PosValidator.INVALID_TOKEN for token in tokens]

class NgramTokenizer(object):
    @staticmethod
    def build_ngram_context(document: Document, max_ngram=4, delta=5, num=5):
        doc_tokens = [sent.tokens() for sent in document.sentences]

        ngram_counter = defaultdict(int)
        for sent_tokens in doc_tokens:
            for n in range(1, max_ngram + 1):
                for ngram in NgramTokenizer._ngrams(sent_tokens, n):
                    ngram_counter[ngram] += 1

        ngrams_ = {}
        for ngram, count in ngram_counter.items():
            if len(ngram) == 1:
                continue
            first = ngram_counter[ngram[:-1]]
            second = ngram_counter[ngram[1:]]
            score = (count - delta) / (first * second)
            if score > 0:
                ngrams_[ngram] = (count, score)

        ngrams__ = {}
        for ngram, ngram_context in sorted(ngrams_.items(), key=lambda x: -x[1][1])[:num]:
            ngrams__[ngram] = ngram_context

        return ngrams__

    @staticmethod
    def tokenize(sentence: Sentence, ngram_context, max_ngram=4):
        tokens = sentence.tokens()
        tokens_ = []

        i = 0
        while i < len(tokens):
            ngrams = []
            for n in range(1, max_ngram+1):
                ngrams.append(tuple(tokens[i:i+n]))
            ngrams = list(filter(lambda x: x in ngram_context, ngrams))

            if len(ngrams) == 0:
                tokens_.append(tokens[i])
                i += 1
            else:
                ngram_size = len(sorted(ngrams, key=lambda x: -len(x))[0])
                tokens_.append('-'.join(tokens[i:i+ngram_size]))
                i += ngram_size

        return tokens_

    @staticmethod
    def _ngrams(tokens, n):
        def filter_ngrams(ngrams):
            ngrams_ = []
            for ngram in ngrams:
                if any([not PosValidator.is_valid(token) for token in ngram]):
                    continue
                ngrams_.append(ngram)
            return ngrams_

        ngrams = []
        for b in range(0, len(tokens) - n + 1):
            ngrams.append(tuple(tokens[b:b+n]))

        return filter_ngrams(ngrams)
