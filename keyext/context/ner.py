import re
import os
import requests
import pickle
from typing import *
from tqdm import tqdm
from collections import defaultdict

from . import Context
from ..util import token_preprocess, preprocess
from ..model import Document

# from tagger import KhaiiiTagger
from konlpy.tag import Komoran
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig


class NerContext(Context):
    def __init__(self):
        super().__init__()
        self._initialized = False

    def import_model(self, f, model_dir) -> None:
        self.contexts = pickle.load(f)

        # Load model
        args = torch.load(os.path.join(model_dir, 'training_args.bin'))
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir, return_dict=True)
        self.config = AutoConfig.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.label_list = [x[1] for x in sorted(self.config.id2label.items(), key=lambda x:x[0])]

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.to(self.device)
        self.model.eval()

        self.tagger = Komoran()

        self._initialized = True

    def export_model(self, f) -> None:
        if not self._initialized:
            raise Exception('model is not initialized. nothing to export.')

        pickle.dump(self.contexts, f)
    
    def prediction(self, document: str):
        sentences = [" ".join(self.tagger.morphs(sentence)) for sentence in document.split('\n')]

        NEs = []
        with torch.no_grad():
            for sentence in tqdm(sentences, desc='Predicting NER'):
                words = sentence.split()
                inputs = self.tokenizer.encode(sentence, return_tensors="pt").to(self.device)

                outputs = self.model(inputs).logits
                predictions = torch.argmax(outputs, dim=2)

                tokens = []
                slot_label_mask = []
                for word in words:
                    word_tokens = self.tokenizer.tokenize(word)
                    if not word_tokens:
                        word_tokens = [self.tokenizer.unk_token]
                    tokens.extend(word_tokens)
                    slot_label_mask.extend([0] + [-100] * (len(word_tokens) - 1))
                slot_label_mask = [-100] + slot_label_mask + [-100]

                preds_list = predictions[0].detach().cpu().numpy()
                preds_list_ = []

                for i in range(len(preds_list)):
                    if slot_label_mask[i] != -100:
                        preds_list_.append(self.label_list[preds_list[i]])


                # BIO concatenation
                NE = []

                for i, (pred, word) in enumerate(zip(preds_list_, words)):
                    if pred == 'O':
                        continue
                    if pred.endswith('I') and (i > 0 and preds_list_[i-1][:-2] == pred[:-2]):
                        NE[-1] = [NE[-1][0], NE[-1][1]+' '+word]
                    else:
                        NE.append([pred, word]) 

                # TODO: B,I 아니어도 붙이는거 고려하기 - Tokenizer Issue로 인해
                # ex. 패(PS-B), 러(PS-B), 데이(PS-B)
                # TODO: NER subword 제거
                # TODO: PS 태그에서 은,는,이,가 붙은 태그 많으면 제거하는 방식 고려

                if NE:
                    NEs.extend([(tag[:-2], word) for (tag, word) in NE])

        return NEs

    def get_keywords(self, document: Document) -> List[Tuple[str,str]]:
        if self._initialized and document.id in self.contexts:
            return self.contexts[document.id]['all']
        else:
            return [(tag, word) for (tag, word) in self.prediction(document.text())]

    def get_related_keywords(self, document: Document, queries: List[str]) -> List[Tuple[str,str]]:
        sentences = [sentence.text() for sentence in document.sentences]
        related_sentence_ids = []

        # find all sentences containing all queries
        related_sentence_ids = [sentence_id for sentence_id, sentence in enumerate(sentences) if all(x in sentence for x in queries)]
        
        # find all sentences containing last queries if there are no sentences
        if len(related_sentence_ids)==0 or (len(queries)>=2 and len(related_sentence_ids)<=5):
            related_sentence_ids = [sentence_id for sentence_id, sentence in enumerate(sentences) if queries[-1] in sentence]
        
        if self._initialized and document.id in self.contexts:
            ners = sum([self.contexts[document.id]['by_sentence'][sentence_id] for sentence_id in related_sentence_ids],[])
        else:
            all_sentences = "\n".join([document.sentences[sentence_id].text() for sentence_id in related_sentence_ids])
            ners = self.prediction(all_sentences)

        return [(tag, word) for tag, word in ners if tag != 'NUM']

