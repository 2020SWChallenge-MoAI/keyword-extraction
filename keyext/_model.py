class Document(object):
    """
    Preprocessed document.

    Strongly recommends to create with `keyext.convert_to_document`.

    ---

    ### Attributes
    - title: document title
    - sentences: document sentences (preprocessed)
    """

    def __init__(self, title, sentences):
        super().__init__()
        self.title = title
        self.sentences = sentences


class AnalyzedDocument(Document):
    """
    Document analyzed by `keyext.KeywordExtractor`.

    ---

    ### Attributes
    inherited attributes of `keyext.Document`

    - vector: document term vector
    - keywords: extracted keywords based on TF-IDF
    - analyzed_sentences: sentences of documents, contains pos tag and count vector
    - idx2vocab: document-wide vocabulary (not database-wide)
    """

    def __init__(self, document, vector=None, keywords=None,
                 analyzed_sentences=None, idx2vocab=None):
        super().__init__(document.title, document.sentences)

        self.vector = vector
        self.keywords = keywords
        self.analyzed_sentences = analyzed_sentences
        self.idx2vocab = idx2vocab


class AnalyzedSentence(object):
    """
    Analyzed sentence.

    ---

    ### Attributes

    - vector: sentence term vector (document-wide, count vector)
    - keywords: sentence keywords (sorted by keyword TF-IDF (not count))
    - tags: pos tagged sentence
    """

    def __init__(self, vector=None, keywords=None, tags=None):
        super().__init__()

        self.vector = vector
        self.keywords = keywords
        self.tags = tags
