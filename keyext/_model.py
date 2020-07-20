class Document(object):
    """
    Raw document.
    
    Strongly recommends to create with `keyext.convert_to_document`.
    
    ### Attributes
    - title: document title
    - author: document author
    - sentences: document sentences (preprocessed)
    """

    def __init__(self, title, sentences):
        super().__init__()
        self.title = title
        self.sentences = sentences


class AnalyzedDocument(Document):
    """
    Document analyzed by `keyext.KeywordExtractor`.
    
    ### Attributes
    inherited attributes of `keyext.Document`

    - vector: document term vector
    - keywords: extracted keywords based on TF-IDF
    - tags: part-of-speeches of sentences
    """
    def __init__(self, document, vector=None, keywords=None, tags=None):
        super().__init__(document.title, document.sentences)

        self.vector = vector
        self.keywords = keywords
        self.tags = tags
