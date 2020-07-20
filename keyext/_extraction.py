import pickle
from konlpy.tag import Komoran
from sklearn.feature_extraction.text import TfidfVectorizer

from . import Document, AnalyzedDocument


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

        self._tagger = Komoran()
        self._vectorizer = TfidfVectorizer(
            ngram_range=options['ngram_range'],
            max_df=options['max_df'],
            tokenizer=Tokenizer(tagger=self._tagger))

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
        with open(model_path, 'wb') as f:
            pickle.dump(self._documents, f)
            pickle.dump(self._vocab, f)

    def build(self, documents):
        # pos tagging
        for document in documents:
            assert type(document) is Document

            doc = AnalyzedDocument(document)
            doc.tags = [self._tagger.pos(sent) for sent in document.sentences]

            self._documents.append(doc)

        # build document vector
        vectors = self._vectorizer.fit_transform([' '.join(doc.sentences) for doc in documents])

        # build vocabulary
        self._vocab = [vocab for vocab, idx in sorted(self._vectorizer.vocabulary_.items(), key=lambda x:x[1])]

        # add additional informations to documents
        for doc, vector in zip(self._documents, vectors):
            # add document vector
            doc.vector = vector

            # add sorted keywords (only if term weight > 0.01)
            keywords = [(weight, self._vocab[idx])
                        for idx, weight in enumerate(vector.toarray().squeeze()) if weight > 0.01]
            doc.keywords = sorted(keywords, reverse=True)

    def recommend(self, document_id, num=2, exclude_keywords=[]):
        keywords = []
        included = {}

        # mark exclude keywords as already included
        for ex_keyword in exclude_keywords:
            included[ex_keyword] = True

        # get keywords
        for keyword in self._documents[document_id].keywords:
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
                for word_piece in word_pieces:
                    included[word_piece] = True

        # TODO add smarter recommendation method

        return keywords[:num]


class Tokenizer(object):
    def __init__(self, tagger=Komoran(), noun_only=True):
        self._tagger = tagger
        self._noun_only = noun_only

    def __call__(self, sent):
        pos = self._tagger.pos(sent)
        pos = ['{}/{}'.format(word, tag) for word, tag in pos if not self._noun_only or tag.startswith('NN')]
        return pos
