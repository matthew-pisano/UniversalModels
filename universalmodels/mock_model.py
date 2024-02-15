from universalmodels.wrappers.wrapper_model import WrapperModel


class MockModel(WrapperModel):

    def __init__(self, model_name: str, model_task: str, **kwargs):

        super().__init__(model_name, **kwargs)
        self.model_task = model_task
