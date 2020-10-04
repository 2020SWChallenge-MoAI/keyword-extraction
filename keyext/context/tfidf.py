import pickle
import logging
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from typing import *
from collections import defaultdict

from ..util import PosTokenizer, PosValidator
from ..model import *
from . import Context

logger = logging.getLogger(__name__)


def _identical_tokenizer(text):
    return text


class TfidfContext(Context):
    def __init__(self):
        super().__init__()

        self.tokenizer = _identical_tokenizer
        self.count_vectorizer = CountVectorizer(
            ngram_range=(1, 3),
            tokenizer=self.tokenizer,
            lowercase=False,
            max_df=0.5
            #stop_words=[PosValidator.INVALID_TOKEN]
        )
        self.tfidf_transformer = TfidfTransformer()

        self._initialized = False

    def import_model(self, model: bytes):
        data = pickle.loads(model)

        self.count_vectorizer = data['count_vectorizer']
        self.tfidf_transformer = data['tfidf_transformer']
        self.vocab2idx = data['vocab2idx']
        self.idx2vocab = data['idx2vocab']
        self.contexts = data['contexts']

        self._initialized = True

    def export_model(self) -> bytes:
        if not self._initialized:
            raise Exception('model is not initialized. nothing to export.')

        data = pickle.dumps({
            'count_vectorizer': self.count_vectorizer,
            'tfidf_transformer': self.tfidf_transformer,
            'vocab2idx': self.vocab2idx,
            'idx2vocab': self.idx2vocab,
            'contexts': self.contexts
        })

        return data

    def build(self, documents: List[Document]):
        # build document-wide vectors
        logger.info('start count vectorization and tfidf transformation. this may take a while...')

        count_vectors = self.count_vectorizer.fit_transform([doc.tokens() for doc in documents])
        self.tfidf_transformer.fit(count_vectors)
        logger.info('count vectorization and tfidf transformation complete')

        self.idx2vocab = [v for v, _ in sorted(
            self.count_vectorizer.vocabulary_.items(), key=lambda x:x[1])]
        self.vocab2idx = self.count_vectorizer.vocabulary_
        logger.info('vocab generation complete')

        logger.info(f'start build context [0/{len(documents)}]')
        self.contexts = {}
        for i, (doc, count_vector) in enumerate(zip(documents, count_vectors)):
            context = self._build_document_context(doc, count_vector)
            self.contexts[doc.id] = context
            logger.info(f'done [{i+1}/{len(documents)}]')

        self._initialized = True

    def _build_document_context(self, document: Document, count_vector=None):
        if count_vector is None:
            count_vector = self.count_vectorizer.transform([document.tokens()])[0]

        tfidf_vector = self.tfidf_transformer.transform(count_vector)

        sentence_matrix = self.count_vectorizer.transform([sent.tokens() for sent in document.sentences])
        sentence_matrix = self.tfidf_transformer.transform(sentence_matrix)
        cooccurence_matrix = (sentence_matrix.T * sentence_matrix)
        cooccurence_matrix.setdiag(0)

        keywords = sorted([
            (self.idx2vocab[idx], weight)
            for idx, weight
            in enumerate(tfidf_vector.toarray().squeeze())
            if weight > 0
        ], key=lambda k: -k[1])

        return {
            'count_vector': count_vector,
            'tfidf_vector': tfidf_vector,
            'sentence_matrix': sentence_matrix,
            'cooccurence_matrix': cooccurence_matrix,
            'keywords': keywords
        }

    def _get_document_context(self, document:Document):
        if not self._initialized:
            raise Exception('Tfidf context is not initialized.')

        if not document.id or not document.id in self.contexts:
            return self._build_document_context(document)
        else:
            return self.contexts[document.id]

    def get_keywords(self, document: Document):
        document_context = self._get_document_context(document)

        keywords = defaultdict(float)
        for keyword, weight in document_context['keywords']:
            if PosValidator.is_valid(keyword):
                word = PosTokenizer.word(keyword)
                keywords[word] = max(keywords[word], weight)

        return list(keywords.items())

    def get_related_keywords(self, document, tokens: List[str]):
        document_context = self._get_document_context(document)
        
        query_token = tokens[-1]
        cooccurence_vector = document_context['cooccurence_matrix'][self.vocab2idx[query_token]]

        keywords = sorted([
            (self.idx2vocab[idx], weight)
            for idx, weight
            in enumerate(cooccurence_vector.toarray().squeeze())
            if weight > 0
        ], key=lambda k: -k[1])

        filtered_keywords = defaultdict(float)
        for keyword, weight in keywords:
            if PosValidator.is_valid(keyword) and not PosTokenizer.contains(keyword, query_token):
                word = PosTokenizer.word(keyword)
                filtered_keywords[word] = max(filtered_keywords[word], weight)
        
        return list(filtered_keywords.items())