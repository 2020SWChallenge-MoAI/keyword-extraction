import pickle
import os

from konlpy.tag import Komoran
from sklearn.feature_extraction.text import TfidfVectorizer

from . import Document, AnalyzedDocument


class KeywordExtractor(object):
    def __init__(self, options={'ngram_range': (1, 2), 'max_df': 0.6}, model_path=None):
        super().__init__()

        self._tokenizer = Tokenizer()
        self._vectorizer = TfidfVectorizer(
            ngram_range=options['ngram_range'],
            max_df=options['max_df'],
            tokenizer=_tokenizer)

        self._documents = []
        self._vocab = []

        if model_path is not None:
            load_model(model_path)

    def load(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                self._documents = pickle.load(f)
                self._vocab = pickle.load(f)
        except:
            raise ValueError("model is not a valid file.")

    def save(self, model_path):
        with open(os.path.join(model_path, 'model.pkl'), 'wb') as f:
            pickle.dump(self._documents, f)
            pickle.dump(self._vocab, f)

    def build(self, documents):
        # pos tagging
        for document in documents:
            assert type(document) is Document

            doc = AnalyzedDocument(document)
            doc.tags = [tagger.pos(sent) for sent in document.sentences]

            self._documents.append(doc)

        # build document vector
        vectors = vectorizer.fit_transform([' '.join(doc.sentences) for doc in documents])

        # build vocabulary
        self._vocab = [vocab for vocab, idx in sorted(vectorizer.vocabulary_.items(), key=lambda x:x[1])]

        # add additional informations to documents
        for doc, vector in zip(self._documents, vectors):
            # add document vector
            doc.vector = vector

            # add sorted keywords (only if term weight > 0.01)
            keywords = [(weight, self._vocab[idx])
                        for idx, weight in enumerate(vector.toarray().squeeze()) if weight > 0.01]
            doc.keywords = sorted(keywords, reversed=True)

    def recommend(self, document_id, num=2, exclude_keywords=[]):
        keywords = []
        included = {}

        for ex_keyword in exclude_keywords:
            included[ex_keyword] = True

        for keyword in self._documents[document_id].keywords:
            word_pieces = [k.split('/')[0] for k in keyword[1].split(' ')]

            duplicated = False
            for word_piece in word_pieces:
                if word_piece in included:
                    duplicated = True
                    break

            if not duplicated:
                keywords.append(merge_word(word_pieces))
                for word_piece in word_pieces:
                    included[word_piece] = True

        # TODO add smarter recommendation method

        return keywords[:num]


class Tokenizer(object):
    def __init__(self, tagger=Komoran(), noun_only=True):
        self.tagger = tagger

    def __call__(self, sent):
        pos = self.tagger.pos(sent)
        pos = ['{}/{}'.format(word, tag) for word, tag in pos if not noun_only or tag.startswith('NN')]
        return pos
