import pickle
import logging

from typing import *
from collections import defaultdict

from .context import *
from .model import *
from .util import NgramMerger, PosTokenizer, simple_preprocess


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

            if models['tfidf']:
                self.tfidf_context.import_model(models['tfidf'])
            if models['word2vec']:
                self.word2vec_context.import_model(models['word2vec'])
            # if models['ner']:
            #    self.ner_context.import_model(models['ner'])

            self.documents = documents
        except:
            raise ValueError('model is not a valid file.')

    def save(self, model_path: str):
        try:
            models = {}
            if self.tfidf_context._initialized:
                models['tfidf'] = self.tfidf_context.export_model()
            if self.word2vec_context._initialized:
                models['word2vec'] = self.word2vec_context.export_model()
            #models['ner'] = self.ner_context.export_model()

            with open(model_path, 'wb') as f:
                pickle.dump(models, f)
                pickle.dump(self.documents, f)
        except:
            raise ValueError(
                'error occured when saving model. check your model path.')

    def build(self, raw_documents: List[Tuple[int, List[str]]]):
        # convert to document format
        logger.info('start document converting...')
        self.documents = dict()
        self.word2postoken = defaultdict(set)
        self.word2texttoken = defaultdict(set)

        for doc_id, raw_sentences in raw_documents:
            self.documents[doc_id] = self._build_document(doc_id, raw_sentences)
        logger.info('document converting complete')

        self.tfidf_context.build(list(self.documents.values()))
        logger.info('tf-idf context build complete')

        self.word2vec_context.build(list(self.documents.values()))
        logger.info('word2vec context build complete')

    def _build_document(self, doc_id, raw_sentences):
        sentences = []
        for sent_id, raw_sentence in enumerate(raw_sentences):
            sentence = Sentence(f'{doc_id}.{sent_id}', raw_sentence.strip())

            tokens = self.tokenizer.tokenize(sentence.text())
            sentence.set_tokens(tokens['pos'], tokens['text'])

            sentences.append(sentence)

        logger.info(f'document {doc_id} converting complete')

        document = Document(f'{doc_id}', sentences)

        ngram_context = NgramMerger.build_ngram_context(document)
        for sent in sentences:
            merged_tokens = NgramMerger.merge_ngram(sent, ngram_context)
            sent.set_tokens(merged_tokens['pos'], merged_tokens['text'])

            for pos_token, text_token in zip(merged_tokens['pos'], merged_tokens['text']):
                subtokens = PosTokenizer.subtokens(pos_token)
                for word, _ in subtokens:
                    self.word2postoken[word].add(pos_token)
                    self.word2texttoken[word].add(text_token)

                word = PosTokenizer.joinedtext(pos_token)
                self.word2postoken[word].add(pos_token)
                self.word2texttoken[word].add(text_token)

        return document

    def recommend(self, document_id, keyword_history=[], num: int = 3) -> List[Dict]:
        document = self.documents[document_id]
        return self._recommend(document, keyword_history, num)

    def recommend_from_sentences(self, sentences: List[str], keyword_history=[], num: int = 3) -> List[Dict]:
        document = self._build_document(-1, sentences)
        return self._recommend(document, keyword_history, num)

    def _recommend(self, document, keyword_history, num) -> List[Dict]:
        # contextual recommendation
        if len(keyword_history) > 0:
            extended_keyword_history = []
            extended_keyword_history.extend(keyword_history)
            for keyword in keyword_history:
                w2v_related_keywords = self.word2vec_context.get_related_keywords([keyword], num=2)
                extended_keyword_history.extend([w for w, v in w2v_related_keywords])

            search_pos_tokens = []
            for keyword in extended_keyword_history:
                keyword = simple_preprocess(keyword).replace(' ', '')
                search_pos_tokens.extend(list(self.word2postoken[keyword]))

            keywords = self.tfidf_context.get_related_keywords(document, search_pos_tokens)
        else:
            keywords = self.tfidf_context.get_keywords(document)

        keywords = [{'word': keyword, 'weight': weight} for keyword, weight in keywords if keyword not in keyword_history]

        return keywords[:num]


class DummyExtractor(object):
    def __init__(self):
        super().__init__()

    def recommend(self, document_id, keyword_history=[], num: int = 3, scenario={}):
        """
        Returns recommend keywords based on dummy scenario

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

    def recommend_from_sentences(self, sentences: List[str], keyword_history=[], num: int = 3) -> List[Dict]:
        return []
