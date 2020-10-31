from contextlib import suppress
import pickle
import logging

from typing import *
from collections import defaultdict, Counter

from .context import *
from .model import *
from .util import NgramTokenizer, PosTokenizer, PosValidator
from .util import token_preprocess
from .util import same_cheon


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

    def recommend(self, document_id, queries: List[str] = [], tags: List[str]= [], num: int = 3) -> List[Dict]:
        document = self.documents[document_id]
        return self._recommend(document, queries, tags, num)

    def recommend_from_sentences(self, sentences: List[str], queries: List[str] = [], tags: List[str] = [], num: int = 3) -> List[Dict]:
        document = self._build_document(Document.TEMP_ID, sentences)
        return self._recommend(document, queries, tags, num)

    def _recommend(self, document: Document, queries: List[str] = [], tags: List[str] = [], num: int = 3) -> List[Dict]:
        def filter_subwords(keywords):
            valid = [True for _ in range(len(keywords))]
            for i, (k, w) in enumerate(keywords):
                for j , (kk, ww) in enumerate(keywords[i+1:i+6]):
                    if k in kk:
                        keywords[j+i+1] = (kk, ww)
                        valid[i] = False

                for j, (kk, ww) in enumerate(keywords[max(0,i-5):i]):
                    if k in kk:
                        keywords[j+max(0,i-5)] = (kk, ww)
                        valid[i] = False

            return sorted([k for k, v in zip(keywords, valid) if v], key=lambda x:-x[1])

        def correct_weights(keywords):
            ner_keywords = self.ner_context.get_keywords(document)
            for i, (k, w) in enumerate(keywords):
                ner_matched = 0
                for kk, ww in ner_keywords:
                    if same_cheon(k, kk):
                        ner_matched += ww
                
                if not ner_matched:
                    keywords[i] = (k, w * 0.5)

            return sorted(keywords, key=lambda k: -k[1])

        def combine_ner_and_keywords(keywords, counter, lower_bound):
            lower_w = 0 if len(keywords)==0 else keywords[min(len(keywords)-1,lower_bound)][1]
            ner_keywords = sorted([(ner[1], count) for ner, count in counter.items()], key=lambda x: -x[1]) # sort by count
            for i, (k, c) in enumerate(ner_keywords):
                ner_keywords[i] = (k, c*0.01+lower_w)

            d = dict(keywords)
            d.update(ner_keywords)

            return sorted(list(d.items()), key=lambda x:-x[1])


        # preprocess and convert to tokens (using predefined token dictionary)
        query_tokens = sum([list(self.word2tokens[token_preprocess(k)]) for k in queries], [])

        if len(query_tokens) > 0: # get related keywords
            keywords = self.tfidf_context.get_related_keywords(document, query_tokens)
            ner_keywords = self.ner_context.get_related_keywords(document, [token_preprocess(k) for k in queries])
            keywords = combine_ner_and_keywords(keywords, Counter(ner_keywords), lower_bound=int(num*0.4))

        else: # non-contextual keywords extraction
            keywords = self.tfidf_context.get_keywords(document)

        if len(tags) > 0: # get keywords related to ner tags
            ner_keywords = self.ner_context.get_keywords(document)
            counter = Counter(filter(lambda x: x[0] in tags, ner_keywords)) # filter by tags
            keywords = combine_ner_and_keywords(keywords, counter, lower_bound=int(num*0.2))

        # convert to dictionary format and return
        return [{'word': k, 'weight': w} for k, w in filter_subwords(keywords) if k not in queries][:num]


class DummyExtractor(object):
    def __init__(self):
        super().__init__()

    def recommend(self, document_id, queries: List[str] = [], num: int = 3):
        return [{'word': 'sample', 'weight': 0.7}]

    def recommend_from_sentences(self, sentences: List[str], queries: List[str] = [], num: int = 3) -> List[Dict]:
        return [{'word': 'sample', 'weight': 0.7}]
