import re
import pickle
import logging
import random
import os

from typing import *
from collections import defaultdict, Counter

from .context import *
from .model import *
from .util import NgramTokenizer, PosTokenizer, PosValidator
from .util import token_preprocess


logger = logging.getLogger(__name__)


class KeywordExtractor(object):
    def __init__(self, model_path: str = None):
        self.tfidf_context = TfidfContext()
        self.word2vec_context = Word2VecContext()
        self.ner_context = NerContext()

        if model_path is not None:
            self.load(model_path)

    def load(self, model_path: str):
        """
            model_path should exclude extension
        """
        try:
            with open(f'{model_path}.metadata.pkl', 'rb') as f:
                self.documents = pickle.load(f)
                self.word2tokens = pickle.load(f)

            for (context_name, context) in [('tfidf', self.tfidf_context), ('word2vec', self.word2vec_context), ('ner', self.ner_context)]:
                context_path = f'{model_path}.{context_name}.pkl'

                if not os.path.isfile(context_path):
                    continue

                with open(context_path, 'rb') as f:
                    context.import_model(f)
        except:
            raise ValueError('model is not a valid file.')

    def save(self, model_path: str):
        """
            model_path should exclude extension
        """
        try:
            with open(f'{model_path}.metadata.pkl', 'wb') as f:
                pickle.dump(self.documents, f)
                pickle.dump(self.word2tokens, f)

            if self.tfidf_context._initialized:
                with open(f'{model_path}.tfidf.pkl', 'wb') as f:
                    self.tfidf_context.export_model(f)
            if self.word2vec_context._initialized:
                with open(f'{model_path}.word2vec.pkl', 'wb') as f:
                    self.word2vec_context.export_model(f)
            if self.ner_context._initialized:
                with open(f'{model_path}.ner.pkl', 'wb') as f:
                    self.ner_context.export_model(f)

        except:
            raise ValueError('error occured when saving model. check your model path.')

    def build(self, raw_documents: List[Tuple[int, List[str]]]):
        # convert to document format
        logger.info('start document converting...')
        self.documents = dict()
        self.word2tokens = defaultdict(set)

        for doc_id, raw_sentences in raw_documents:
            self.documents[doc_id] = self._build_document(doc_id, raw_sentences)
        logger.info('document converting complete')

        self.tfidf_context.build(list(self.documents.values()))
        logger.info('tf-idf context build complete')

        #self.word2vec_context.build(list(self.documents.values()))
        #logger.info('word2vec context build complete')

    def _build_document(self, doc_id, raw_sentences, ngram=True):
        # initialize sentences
        sentences = []
        for sent_id, raw_sentence in enumerate(raw_sentences):
            sentence = Sentence(f'{doc_id}.{sent_id}', raw_sentence.strip())

            tokens = PosTokenizer.tokenize(sentence.text())
            sentence.set_tokens(tokens)

            if not ngram:
                for token in tokens:
                    word = PosTokenizer.word(token)
                    self.word2tokens[word].add(token)

                    subtokens = PosTokenizer.subtokens(token)
                    for word, _ in subtokens:
                        if ' ' in word:
                            for subword in word.split():
                                self.word2tokens[subword].add(token)
                        self.word2tokens[word].add(token)
            
            sentences.append(sentence)

        # initialize document
        document = Document(f'{doc_id}', sentences)

        # generate ngram tokens
        if ngram:
            ngram_context = NgramTokenizer.build_ngram_context(document)
            for sent in sentences:
                ngram_tokens = NgramTokenizer.tokenize(sent, ngram_context)
                sent.set_tokens(PosValidator.filter(ngram_tokens))

                for token in ngram_tokens:
                    if not PosValidator.is_valid(token):
                        continue

                    word = PosTokenizer.word(token)
                    self.word2tokens[word].add(token)

                    subtokens = PosTokenizer.subtokens(token)
                    for word, _ in subtokens:
                        if ' ' in word:
                            for subword in word.split():
                                self.word2tokens[subword].add(token)
                        self.word2tokens[word].add(token)

        logger.info(f'document {doc_id} converting complete')
        return document

    def recommend(self, document_id, queries: List[str] = [], tags: List[str]= [], num: int = 3, use_ner=True) -> List[Dict]:
        document = self.documents[document_id]
        return self._recommend(document, queries, tags, num, use_ner)

    def recommend_from_sentences(self, sentences: List[str], queries: List[str] = [], tags: List[str] = [], num: int = 3, use_ner=True) -> List[Dict]:
        document = self._build_document(Document.TEMP_ID, sentences)
        return self._recommend(document, queries, tags, num)

    def _recommend(self, document: Document, queries: List[str] = [], tags: List[str] = [], num: int = 3, use_ner=True) -> List[Dict]:
        def filter_subwords(keywords):
            valid = [True for _ in range(len(keywords))]
            for i in range(len(keywords)):
                for j in range(max(0,i-3), min(i+3,len(keywords))): # compare within certain range
                    if i!=j and keywords[i][0] in keywords[j][0]:
                        keywords[j] = (keywords[j][0], max(keywords[j][1], keywords[i][1]))
                        valid[i] = False

            return sorted([k for k, v in zip(keywords, valid) if v], key=lambda x: -x[1])


        def combine_ner_and_keywords(keywords, counter):
            base_weight = 0 if len(keywords)==0 else keywords[:num - 1][-1][1]
            max_weight = 1 if len(keywords)==0 else keywords[0][1]
            ner_keywords = [(ner[1], count*random.uniform(0.1, max_weight) + base_weight) for ner, count in counter.items()]

            d = dict(keywords)
            for k, w in ner_keywords:
                d[k] = max(d[k], w) if k in d else w
                
            return sorted(list(d.items()), key=lambda x:-x[1])


        # preprocess and convert to tokens (using predefined token dictionary)
        preprocessed_queries = [token_preprocess(k) for k in queries]
        query_tokens = sum([list(self.word2tokens[k]) for k in preprocessed_queries], [])

        if len(query_tokens) > 0: # get related keywords
            keywords = self.tfidf_context.get_related_keywords(document, query_tokens)
            # if related keywords are empty, get non-contextual keywords
            if not keywords:
                keywords = self.tfidf_context.get_keywords(document)
        else: # non-contextual keywords extraction
            keywords = self.tfidf_context.get_keywords(document)
        
        if use_ner:
            if len(preprocessed_queries) > 0: # get related named entities
                ner_keywords = self.ner_context.get_related_keywords(document, preprocessed_queries)[:int(num*0.6)]
                keywords = combine_ner_and_keywords(keywords, Counter(ner_keywords))
            
            if len(tags) > 0: # get keywords related to ner tags
                ner_keywords = self.ner_context.get_keywords(document)
                counter = Counter(filter(lambda x: x[0] in tags, ner_keywords)) # filter by tags
                keywords = combine_ner_and_keywords(keywords, counter)

        # convert to dictionary format and return
        keywords = filter_subwords(keywords)
        query_string = " ".join(preprocessed_queries)
        return [{'word': k, 'weight': w} for k, w in keywords if token_preprocess(k) not in query_string][:num]


class DummyExtractor(object):
    def __init__(self):
        super().__init__()

    def recommend(self, document_id, queries: List[str] = [], num: int = 3):
        return [{'word': 'sample', 'weight': 0.7}]

    def recommend_from_sentences(self, sentences: List[str], queries: List[str] = [], num: int = 3) -> List[Dict]:
        return [{'word': 'sample', 'weight': 0.7}]
