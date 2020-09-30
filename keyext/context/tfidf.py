import pickle
import logging
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack

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
            lowercase=False
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
        logger.info(
            'start count vectorization and tfidf transformation. this may take a while...')

        count_vectors = self.count_vectorizer.fit_transform(
            [doc.pos_tokens() for doc in documents]
        )
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
            count_vector = self.count_vectorizer.transform(
                [document.pos_tokens()])[0]
        tfidf_vector = self.tfidf_transformer.transform(count_vector)

        sentence_vectors = [self.count_vectorizer.transform(
            [sent.pos_tokens()])[0] for sent in document.sentences]

        keywords = sorted([
            (weight, self.idx2vocab[idx])
            for idx, weight
            in enumerate(tfidf_vector.toarray().squeeze())
            if weight > 0
        ], reverse=True)

        return {
            'count_vector': count_vector,
            'tfidf_vector': tfidf_vector,
            'sentence_vectors': sentence_vectors,
            'keywords': keywords
        }

    def get_keywords(self, document: Document):
        if not self._initialized:
            raise Exception('Tfidf context is not initialized.')

        # no contextual information
        if not document.id or not document.id in self.contexts:
            document_context = self._build_document_context(document)
        else:
            document_context = self.contexts[document.id]

        keywords = document_context['keywords']

        filtered_keywords = defaultdict(float)
        for weight, keyword in keywords:
            if PosValidator.is_valid(keyword):
                word = PosTokenizer.text(keyword)
                filtered_keywords[word] = max(filtered_keywords[word], weight)

        return list(filtered_keywords.items())

    def get_related_keywords(self, document, pos_tokens: List[str]):
        if not self._initialized:
            raise Exception('Tfidf context is not initialized.')

        if not document.id or not document.id in self.contexts:
            document_context = self._build_document_context(document)
        else:
            document_context = self.contexts[document.id]

        result = defaultdict(float)

        # build search vector (count based bag-of-word vector)
        search_vector = np.zeros((1, len(self.idx2vocab)), dtype=np.int64)
        for token in pos_tokens:
            if token in self.vocab2idx:
                search_vector[0][self.vocab2idx[token]] = 1

        # search by cosine similarity
        similarity = cosine_similarity(search_vector, vstack(
            [v for v in document_context['sentence_vectors']]))

        similar_sentences = [(sim, document.sentences[idx])
                             for idx, sim
                             in enumerate(similarity.squeeze())
                             if sim > 0]
        similar_sentences = sorted(similar_sentences, key=lambda x: -x[0])

        # add results
        for sim, sentence in similar_sentences:
            for pos_token in sentence.pos_tokens():
                if PosValidator.is_valid(pos_token):
                    result[pos_token] += sim

        return sorted([(PosTokenizer.text(k), v) for k, v in result.items()], key=lambda x: -x[1])
