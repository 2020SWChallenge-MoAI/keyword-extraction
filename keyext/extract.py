import pickle
import logging

from typing import *
from collections import defaultdict

from .context import *
from .model import *
from .util import NgramTokenizer, PosTokenizer, PosValidator


logger = logging.getLogger(__name__)


class KeywordExtractor(object):
    def __init__(self, model_path: str = None):
        self.tfidf_context = TfidfContext()
        self.word2vec_context = Word2VecContext()
        self.ner_context = NerContext()

        self.tokenizer: PosTokenizer = PosTokenizer()

        if model_path is not None:
            self.load(model_path)

    def load(self, model_path: str):
        try:
            with open(model_path, 'rb') as f:
                models = pickle.load(f)
                documents = pickle.load(f)
                word2tokens = pickle.load(f)

            if 'tfidf' in models:
                self.tfidf_context.import_model(models['tfidf'])
            if 'word2vec' in models:
                self.word2vec_context.import_model(models['word2vec'])
            if 'ner' in models:
                self.ner_context.import_model(models['ner'])

            self.documents = documents
            self.word2tokens = word2tokens
        except:
            raise ValueError('model is not a valid file.')

    def save(self, model_path: str):
        try:
            models = {}
            if self.tfidf_context._initialized:
                models['tfidf'] = self.tfidf_context.export_model()
            if self.word2vec_context._initialized:
                models['word2vec'] = self.word2vec_context.export_model()
            if self.ner_context._initialized:
                models['ner'] = self.ner_context.export_model()

            with open(model_path, 'wb') as f:
                pickle.dump(models, f)
                pickle.dump(self.documents, f)
                pickle.dump(self.word2tokens, f)
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

    def _build_document(self, doc_id, raw_sentences):
        # initialize sentences
        sentences = []
        for sent_id, raw_sentence in enumerate(raw_sentences):
            sentence = Sentence(f'{doc_id}.{sent_id}', raw_sentence.strip())

            tokens = self.tokenizer.tokenize(sentence.text())
            sentence.set_tokens(tokens)

            sentences.append(sentence)

        # initialize document
        document = Document(f'{doc_id}', sentences)

        # generate ngram tokens
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
                    self.word2tokens[word].add(token)


        logger.info(f'document {doc_id} converting complete')
        return document

    def recommend(self, document_id, queries: List[str] = [], num: int = 3) -> List[Dict]:
        document = self.documents[document_id]
        return self._recommend(document, queries, num)

    def recommend_from_sentences(self, sentences: List[str], queries: List[str] = [], num: int = 3) -> List[Dict]:
        document = self._build_document(Document.TEMP_ID, sentences)
        return self._recommend(document, queries, num)

    def _recommend(self, document: Document, queries: List[str] = [], num: int = 3) -> List[Dict]:
        # preprocess queries
        queries = [k.replace(' ', '') for k in queries]

        if len(queries) > 0: # contextual recommendation
            # query extension (using word2vec)
            extended_queries = list(queries)
            #for queryword in queries:
            #    w2v_related_keywords = self.word2vec_context.get_related_keywords([queryword], num=2)
            #    extended_queries.extend([k for k, _ in w2v_related_keywords])

            # convert to tokens (using predefined token dictionary)
            query_tokens = sum([list(self.word2tokens[k]) for k in extended_queries], [])

            # get related keywords
            if len(query_tokens) > 0:
                keywords = self.tfidf_context.get_related_keywords(document, query_tokens)
            else:
                keywords = self.tfidf_context.get_keywords(document)
        else: # non-contextual keywords extraction
            keywords = self.tfidf_context.get_keywords(document)

        # convert to dictionary format
        keywords = [{'word': k, 'weight': w} for k, w in keywords if k not in queries]

        return keywords[:num]


class DummyExtractor(object):
    def __init__(self):
        super().__init__()

    def recommend(self, document_id, queries: List[str] = [], num: int = 3):
        return [{'word': 'sample', 'weight': 0.7}]

    def recommend_from_sentences(self, sentences: List[str], queries: List[str] = [], num: int = 3) -> List[Dict]:
        return [{'word': 'sample', 'weight': 0.7}]
