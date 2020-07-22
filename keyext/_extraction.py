import pickle
import numpy as np
from konlpy.tag import Komoran
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack

from ._model import *
from ._util import remove_duplicate, ordered_combination


class KeywordExtractor(object):
    """
    Analyze documents and extracts keywords.

    ### Arguments

    - options (dict): keyword analyze options.
        - ngram_range (tuple): (min phrase length, max phrase length)
        - max_df (float): set maximum document frequency. Every term that over `max_df` will be excluded.
    - model_path: path of pre-analyzed document model (optional)
    """

    def __init__(self, options={'ngram_range': (1, 2), 'max_df': 0.6}, model_path=None):
        super().__init__()

        self._ngram_range = options['ngram_range']
        self._max_df = options['max_df']

        self._tagger = Komoran()
        self._document_vectorizer = TfidfVectorizer(
            ngram_range=self._ngram_range,
            max_df=self._max_df,
            tokenizer=Tokenizer(noun_only=True))
        self._sentence_vectorizer = CountVectorizer(
            ngram_range=self._ngram_range,
            tokenizer=Tokenizer(noun_only=True))

        self._documents = []
        self._idx2vocab = []
        self._vocab2idx = {}

        if model_path is not None:
            self.load(model_path)

    def load(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                self._documents = pickle.load(f)
                self._idx2vocab = pickle.load(f)
                self._vocab2idx = pickle.load(f)
        except:
            raise ValueError("model is not a valid file.")

    def save(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self._documents, f)
            pickle.dump(self._idx2vocab, f)
            pickle.dump(self._vocab2idx, f)

    def build(self, documents):
        # build document vectors and vocab
        document_vectors = self._document_vectorizer.fit_transform(
            [' '.join(doc.sentences) for doc in documents])

        idx2vocab = [v for v, idx in sorted(
            self._document_vectorizer.vocabulary_.items(), key=lambda x:x[1])]
        vocab2idx = self._document_vectorizer.vocabulary_

        self._idx2vocab = idx2vocab
        self._vocab2idx = vocab2idx

        # pos tagging and build sentence vector
        for document, document_vector in zip(documents, document_vectors):
            doc = AnalyzedDocument(document)

            # build sentences vectors and vocab
            sentence_vectors = self._sentence_vectorizer.fit_transform(doc.sentences)

            # build document-wide keyword set
            document_keywords = [(weight, idx2vocab[idx])
                                 for idx, weight
                                 in enumerate(document_vector.toarray().squeeze())
                                 if weight > 0]
            document_keywords = sorted(document_keywords, reverse=True)
            document_idx2vocab = [v for v, idx in sorted(
                self._sentence_vectorizer.vocabulary_.items(), key=lambda x:x[1])]

            # fill informations
            doc.vector = document_vector
            doc.keywords = document_keywords
            doc.idx2vocab = document_idx2vocab
            doc.analyzed_sentences = []
            for sentence, sentence_vector in zip(doc.sentences, sentence_vectors):
                sent = AnalyzedSentence()

                sent.vector = sentence_vector
                sent.tags = self._tagger.pos(sentence)

                keyword_tfidfs = document_vector.toarray().squeeze()
                sentence_keywords = [document_idx2vocab[idx]
                                     for idx, weight
                                     in enumerate(sentence_vector.toarray().squeeze()) if weight > 0]

                sent.keywords = []
                for keyword in sentence_keywords:
                    if keyword in vocab2idx:
                        sent.keywords.append((keyword_tfidfs[vocab2idx[keyword]], keyword))

                doc.analyzed_sentences.append(sent)

            self._documents.append(doc)

    def recommend(self, document_id, keyword_history=[], num=2):
        keywords = []
        candidates = []
        included = set()

        # mark keyword history as already included
        for keyword in keyword_history:
            included.add(keyword)
            included.update(self._tagger.morphs(keyword))

        if False and len(keyword_history) > 0:
            candidates = self.__search_related_keywords(document_id, keyword_history)
        else:
            candidates = self._documents[document_id].keywords

        # get keywords
        for keyword in candidates:
            word_pieces = [k.split('/')[0] for k in keyword[1].split(' ')]
            word = ''.join(word_pieces)

            duplicated = False
            for word_piece in word_pieces:
                if word_piece in included:
                    duplicated = True
                    break
            if word in included:
                duplicated = True

            if not duplicated:
                keywords.append(word)
                included.add(word)
                for word_piece in word_pieces:
                    included.add(word_piece)

        return keywords[:num]

    def __search_related_keywords(self, document_id, keyword_history):
        document = self._documents[document_id]
        result = []

        # build search vector (count based bag-of-word vector)
        search_vector = np.zeros((1, len(document.idx2vocab)), dtype=np.int64)
        for keyword in keyword_history:
            tagged_keyword = ['{}/{}'.format(word, tag) for word, tag in self._tagger.pos(keyword)]

            ngram_range = range(self._ngram_range[0], self._ngram_range[1] + 1)
            predicates = [' '.join(ngram) for ngram in ordered_combination(tagged_keyword, ngram_range)]

            for predicate in predicates:
                try:
                    pred_idx = document.idx2vocab.index(predicate)
                    search_vector[0][pred_idx] = 1
                except ValueError:
                    continue

        # search by cosine similarity
        similarity = cosine_similarity(search_vector, vstack([s.vector for s in document.analyzed_sentences]))

        similar_sentences = [(sim, document.analyzed_sentences[idx]) for idx, sim in enumerate(similarity.squeeze()) if sim > 0]
        similar_sentences = sorted(similar_sentences, key=lambda tuple: tuple[0], reverse=True)
        print([sent.keywords for sim, sent in similar_sentences])

        # add results
        for sim, sentence in similar_sentences:
            result.extend([word for vector, word in sentence.keywords])

        return remove_duplicate(result)


class Tokenizer(object):
    def __init__(self, tagger=Komoran(), noun_only=True):
        self._tagger = tagger
        self._noun_only = noun_only

    def __call__(self, sent):
        pos = self._tagger.pos(sent)
        pos = ['{}/{}'.format(word, tag) for word, tag in pos if not self._noun_only or tag.startswith('NN')]
        return pos
