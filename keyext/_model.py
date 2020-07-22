class Document(object):
    """
    Document analyzed by `keyext.KeywordExtractor`.

    ### Attributes
    inherited attributes of `keyext.Document`

    - vector: document term vector
    - keywords: extracted keywords based on TF-IDF
    - analyzed_sentences: sentences of documents, contains pos tag and count vector
    - idx2vocab: document-wide vocabulary (not database-wide)
    """

    def __init__(self, vector=None, keywords=None,
                 sentences=None, idx2vocab=None, vocab2idx=None):
        super().__init__()

        self.vector = vector
        self.keywords = keywords
        self.sentences = sentences
        self.idx2vocab = idx2vocab
        self.vocab2idx = vocab2idx


class Sentence(object):
    """
    Analyzed sentence.

    ### Attributes

    - original: non-tagged sentence string
    - vector: sentence term vector (document-wide, count vector)
    - keywords: sentence keywords (sorted by keyword TF-IDF (not count))
    - tags: pos tagged sentence
    """

    def __init__(self, original, vector=None, keywords=None, tags=None):
        super().__init__()

        self.original = original
        self.vector = vector
        self.keywords = keywords
        self.tags = tags
