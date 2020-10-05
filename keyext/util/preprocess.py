import re
from krwordrank.hangle import normalize

from ..model import *

def simple_preprocess(raw_document):
    return normalize(raw_document, english=True, number=True, remove_repeat=3)

def preprocess(raw_document, english=True, number=True):
    """
    Processes raw document to preprocessed sentences.

    - sentences are splitted by period(.), question mark(?), and exclamation mark(!).
    - all non-word characters (non hangle) are eliminated.
    - parenthesis are eliminated.
    - quotes are included in parent sentence.

    ### Arguments

    - raw_document: document contents
    - author: document author. (optional)
    - english: if `True`, english words are included in preprocessed sentences. default: `True`
    - number: if `True`, numbers are included in preprocessed sentences. default: `True`

    ### Convert Example
    점순이는 "너 봄春 감자가 맛있단다. 느(너희) 집엔 이거 없지?" 하며 나를 놀렸습니다.
    => 점순이는 너 봄 감자가 맛있단다 느 집엔 이거 없지 하며 나를 놀렸습니다
    """

    # join all lines
    document = raw_document.replace('\n', '').replace('\r', '').strip()

    # regularize quotation mark
    document = re.sub('[‘’]', "'", document)
    document = re.sub('[“”]', '"', document)

    # remove period in quote
    quote = False
    parenthesis = False
    parenthesis_open = '〈<⟪[(『'
    parenthesis_close = '〉>⟫])』'
    cur_parenthesis = ''
    
    new_document = ''
    for c in document:
        if re.match('[\"\']', c):
            quote = not quote
        if re.match('[〈\<⟪\[\(『]', c) and cur_parenthesis == '':
            parenthesis = True
            cur_parenthesis = c
            continue
        if re.match('[〉\>⟫\]\)』]', c) and parenthesis_open.find(cur_parenthesis) == parenthesis_close.find(c):
            parenthesis = False
            cur_parenthesis = ''
            continue
        if not parenthesis and (not quote or (quote and not re.match('[\.\?\!]', c))):
            new_document += c
    document = new_document

    # split and normalize sentences by period
    sents = [normalize(sent, english=english, number=number, remove_repeat=3) for sent in document.split('.')]
    sents = [sent for sent in sents if sent != '']

    return sents
