import re

from ..model import *

pattern = re.compile(r'[^ㄱ-ㅎ가-힣\w]')
spaces_pattern = re.compile(r'\s+')
repeatchars_pattern = re.compile(r'([ㄱ-ㅎ가-힣\w])\1{3,}')

def token_preprocess(raw_token):
    return simple_preprocess(raw_token).replace(' ', '')

def simple_preprocess(raw_document):
    doc = repeatchars_pattern.sub(r'\1' * 3, raw_document)
    doc = pattern.sub(' ', doc)
    doc = spaces_pattern.sub(' ', doc).strip()

    return doc

def preprocess(raw_document):
    """
    Processes raw document to preprocessed sentences.

    - sentences are splitted by period(.), question mark(?), and exclamation mark(!).
    - all non-word characters (non hangle) are eliminated.
    - parenthesis are eliminated.
    - quotes are included in parent sentence.

    ### Arguments

    - raw_document: document contents

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
    sents = [simple_preprocess(sent) for sent in document.split('.')]
    sents = [sent for sent in sents if sent != '']

    return sents
