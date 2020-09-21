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

    - options (dict): keyword analysis options.
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
        self._docid2idx = {}

        if model_path is not None:
            self.load(model_path)

    def load(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                self._documents = pickle.load(f)
                self._idx2vocab = pickle.load(f)
                self._vocab2idx = pickle.load(f)
                self._docid2idx = pickle.load(f)

                # load and reinit tokenizer
                self._document_vectorizer = pickle.load(f)
                self._document_vectorizer.tokenizer = Tokenizer(noun_only=True)
        except:
            raise ValueError("model is not a valid file.")

    def save(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self._documents, f)
            pickle.dump(self._idx2vocab, f)
            pickle.dump(self._vocab2idx, f)
            pickle.dump(self._docid2idx, f)

            # remove tokenizer (resolving dump issue)
            self._document_vectorizer.tokenizer = None
            pickle.dump(self._document_vectorizer, f)

    def build(self, documents):
        """
        Build keyword extraction model from documents.

        ### Arguments

        - documents: preprocessed documents. must be list like below:
        ```python
        [
            (id1, [sentence1, sentence2, ...]),
            (id2, [sentence1, sentence2, ...]),
            ...
        ]
        ```
        """

        # build id to idx mapping
        for idx, doc in enumerate(documents):
            self._docid2idx[doc[0]] = idx

        print("document id mapping complete")

        documents = [sents for id, sents in documents]

        # build document vectors and vocab
        document_vectors = self._document_vectorizer.fit_transform(
            [' '.join(sents) for sents in documents])

        print("document vectorizer initialization complete")

        idx2vocab = [v for v, idx in sorted(
            self._document_vectorizer.vocabulary_.items(), key=lambda x:x[1])]
        vocab2idx = self._document_vectorizer.vocabulary_

        self._idx2vocab = idx2vocab
        self._vocab2idx = vocab2idx

        print(f"start document vectorizing... ({len(documents)} documents)")

        self._documents = []
        for i, (sentences, document_vector) in enumerate(zip(documents, document_vectors)):
            self._documents.append(self.__build_document(sentences, document_vector))
            print(f"document {i + 1}/{len(documents)} complete")

        print("finished")
            
    def recommend_from_sentences(self, sentences, keyword_history=[], num=2):
        if self._document_vectorizer.vocabulary_ is None:
            raise ValueError("document vectorizer is not initialized. call build() before this method.")

        document_vector = self._document_vectorizer.transform([' '.join(sentences)])[0]
        document = self.__build_document(sentences, document_vector)

        return self.__get_keywords(document, keyword_history)[:num]

    def recommend(self, document_id, keyword_history=[], num=2):
        document = self._documents[self._docid2idx[document_id]]
        return self.__get_keywords(document, keyword_history)[:num]

    def __build_document(self, sentences, document_vector):
        document = Document()

        # build sentences vectors and vocab
        sentence_vectors = self._sentence_vectorizer.fit_transform(sentences)

        # build document-wide keyword set
        document.keywords = [(weight, self._idx2vocab[idx])
                             for idx, weight
                             in enumerate(document_vector.toarray().squeeze())
                             if weight > 0]
        document.keywords = sorted(document.keywords, reverse=True)
        document.vocab2idx = self._sentence_vectorizer.vocabulary_
        document.idx2vocab = [v for v, idx in sorted(
            self._sentence_vectorizer.vocabulary_.items(), key=lambda x:x[1])]
        document.sentences = []

        # fill informations
        for sentence, sentence_vector in zip(sentences, sentence_vectors):
            sent = Sentence(sentence)

            sent.vector = sentence_vector
            sent.tags = self._tagger.pos(sentence)

            keyword_tfidfs = document_vector.toarray().squeeze()
            sentence_keywords = [document.idx2vocab[idx]
                                 for idx, weight
                                 in enumerate(sentence_vector.toarray().squeeze())
                                 if weight > 0]

            sent.keywords = [(keyword_tfidfs[self._vocab2idx[keyword]], keyword)
                             for keyword
                             in sentence_keywords
                             if keyword in self._vocab2idx]

            document.sentences.append(sent)

        return document

    def __get_keywords(self, document, keyword_history):
        keywords = []
        candidates = []
        included = set()

        # mark keyword history as already included
        for keyword in keyword_history:
            included.add(keyword)
            included.update(self._tagger.morphs(keyword))

        if len(keyword_history) > 0:
            candidates = self.__search_related_keywords(document, keyword_history)
        else:
            candidates = document.keywords

        # get keywords
        for weight, keyword in candidates:
            word_pieces = [k.split('/')[0] for k in keyword.split(' ')]
            word = ''.join(word_pieces)

            duplicated = False
            for word_piece in word_pieces:
                if word_piece in included:
                    duplicated = True
                    break
            if word in included:
                duplicated = True

            if not duplicated:
                keywords.append((weight, word))
                included.add(word)
                for word_piece in word_pieces:
                    included.add(word_piece)

        return keywords

    def __search_related_keywords(self, document, keyword_history):
        result = []

        # build search vector (count based bag-of-word vector)
        search_vector = np.zeros((1, len(document.idx2vocab)), dtype=np.int64)
        for keyword in keyword_history:
            tagged_keyword = ['{}/{}'.format(word, tag) for word, tag in self._tagger.pos(keyword)]

            ngram_range = range(self._ngram_range[0], self._ngram_range[1] + 1)
            predicates = [' '.join(ngram) for ngram in ordered_combination(tagged_keyword, ngram_range)]

            for predicate in predicates:
                if predicate in document.vocab2idx:
                    search_vector[0][document.vocab2idx[predicate]] = 1

        # search by cosine similarity
        similarity = cosine_similarity(search_vector, vstack([s.vector for s in document.sentences]))

        similar_sentences = [(sim, document.sentences[idx])
                             for idx, sim
                             in enumerate(similarity.squeeze())
                             if sim > 0]
        similar_sentences = sorted(similar_sentences, key=lambda tuple: tuple[0], reverse=True)

        # add results
        for sim, sentence in similar_sentences:
            result.extend(sentence.keywords)

        return remove_duplicate(result)


class DummyExtractor(object):
    def __init__(self):
        super().__init__()

    def recommend(self, document_id, keyword_history=[], num=2, scenario={}):
        """
        Returns recommend keywords based on dummy scinario

        ```python
        {
            'keyword1,keyword2': ['recommend-1', 'recommend-2'],
            'keyword1,keyword2,keyword3': ['recommend-a', 'recommend-b'],
            ...
        }
        ```
        """
        key = ','.join(keyword_history)
        
        if key in scenario:
            return scenario[key][:num]
        
        return []


class Tokenizer(object):
    def __init__(self, tagger=Komoran(), noun_only=True):
        self._tagger = tagger
        self._noun_only = noun_only

    def __call__(self, sent):
        pos = self._tagger.pos(sent)
        pos = ['{}/{}'.format(word, tag) for word, tag in pos if not self._noun_only or tag.startswith('NN')]
        return pos
