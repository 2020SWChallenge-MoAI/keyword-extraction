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
        pos_tokens = [f'{word}/{tag or "NA"}' for word, tag in tokens]
        text_tokens = [word for word, _ in tokens]

        return {
            'pos': pos_tokens,
            'text': text_tokens
        }

    @staticmethod
    def subtokens(pos_token: str) -> Tuple[str]:
        tokens = re.findall(r'[ㄱ-ㅎ가-힣A-Za-z0-9]+(?:\s[ㄱ-ㅎ가-힣A-Za-z0-9]+)*\/\w+', pos_token)
        if not tokens:
            print(f'{pos_token}->{tokens}')
        return tuple([tuple(t.split('/')) for t in tokens])

    @staticmethod
    def text(pos_token: str) -> str:
        return ' '.join([word for word, _ in PosTokenizer.subtokens(pos_token)])

    @staticmethod
    def joinedtext(pos_token: str) -> str:
        return ''.join([word for word, _ in PosTokenizer.subtokens(pos_token)])


class PosValidator(object):
    VALID_TAGS = ('NNP', 'NNG', 'XR', 'XP', 'XSN', 'S', 'NF')
    VALID_INDEPENDENT_TAGS = ('NNP', 'NNG', 'XR', 'S', 'NF')

    @staticmethod
    def is_valid(pos_token: str):
        subtokens = PosTokenizer.subtokens(pos_token)
        if len(subtokens) == 1:  # single
            return subtokens[0][1].startswith(PosValidator.VALID_INDEPENDENT_TAGS)
        else:
            if not all([tag.startswith(PosValidator.VALID_TAGS) for _, tag in subtokens]):
                return False
            if len(set(subtokens)) <= 1:
                return False
            
            return True


class NgramMerger(object):
    @staticmethod
    def build_ngram_context(document: Document, max_ngram=4, delta=3, num=20):
        doc_tokens = [sent.pos_tokens() for sent in document.sentences]

        ngram_counter = defaultdict(int)
        for sent_tokens in doc_tokens:
            for n in range(1, max_ngram + 1):
                for ngram in NgramMerger._ngrams(sent_tokens, n):
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
    def merge_ngram(sentence: Sentence, ngram_context, max_ngram=4):
        pos_tokens = sentence.pos_tokens()
        text_tokens = sentence.text_tokens()

        pos_tokens_ = []
        text_tokens_ = []

        i = 0
        while i < len(pos_tokens):
            ngrams = []
            for n in range(1, max_ngram+1):
                ngrams.append(tuple(pos_tokens[i:i+n]))
            ngrams = list(filter(lambda x: x in ngram_context, ngrams))

            if len(ngrams) == 0:
                pos_tokens_.append(pos_tokens[i])
                text_tokens_.append(text_tokens[i])
                i += 1
            else:
                ngram_size = len(sorted(ngrams, key=lambda x: -len(x))[0])
                pos_tokens_.append('-'.join(pos_tokens[i:i+ngram_size]))
                text_tokens_.append('-'.join(text_tokens[i:i+ngram_size]))
                i += ngram_size

        return {
            'pos': pos_tokens_,
            'text': text_tokens_
        }

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
