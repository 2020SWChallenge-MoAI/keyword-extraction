from . import Context


class NerContext(Context):
    def __init__(self):
        super().__init__()

    def import_model(self, model: bytes) -> None:
        pass

    def export_model(self) -> bytes:
        pass
