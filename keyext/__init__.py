class Document(object):
    def __init__(self, title, sentences):
        super().__init__()
        self.title = title
        self.sentences = sentences


class AnalyzedDocument(Document):
    def __init__(self, document, vector, keywords, tags):
        super().__init__(document.title, document.contents)

        self.vector = vector
        self.keywords = keywords
        self.tags = tags

from .preprocess import convert_to_document
from .extraction import KeywordExtractor