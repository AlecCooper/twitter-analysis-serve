from abc import ABC
import logging

import torch
from torch.nn import Softmax

from transformers import AutoModelForSequenceClassification, AutoTokenizer, \
                         AutoConfig

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class TransformersClassifierHandler(BaseHandler, ABC):
    """
    Transformers text classifier handler class.
    This handler takes a text (string) and
    as input and returns the classification text.
    """
    def __init__(self):
        super(TransformersClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() else "cpu"
            )

        self.model = AutoModelForSequenceClassification. \
            from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.config = AutoConfig.from_pretrained(model_dir)

        self.model.to(self.device)
        self.model.eval()

        # Used to extract probabilities
        self.softmax = Softmax(dim=0)

        logger.debug(f'Transformer model from path'
                     f' {model_dir} loaded successfully')

        if self.config.id2label:
            self.mapping = self.config.id2label
        else:
            logger.warning('Missing the mapping json in the config file.'
                           ' Inference output will not include class name.')

        self.initialized = True

    def preprocess(self, data):
        """ Preprocessing handled on the app server at the moment
            just tokenizes right now
        """
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        sentences = text.decode('utf-8')
        logger.info("Received text: '%s'", sentences)

        inputs = self.tokenizer.encode_plus(
            sentences,
            add_special_tokens=True,
            return_tensors="pt"
        )
        return inputs

    def inference(self, inputs):
        """
        Predict the class of a text using a trained transformer model.
        """
        # NOTE: This makes the assumption that your model expects
        # text to be tokenized with "input_ids" and
        # "token_type_ids" - which is true for some
        # popular transformer models, e.g. bert.
        # If your transformer model expects different tokenization,
        # adapt this code to suit its expected input format.
        outputs = self.model(
            inputs['input_ids'].to(self.device),
        )
        prediction = outputs[0].argmax().item()
        prob = max(self.softmax(outputs[0][0]).tolist())

        logger.info(f"Model predicted: '{prediction}' "
                    f"with softmaxed probability {prob}")

        if self.mapping:
            prediction = self.mapping[prediction]

        payload = {
            "prediction": prediction,
            "probability": prob
        }

        return [payload]

    def postprocess(self, inference_output):
        # TODO: Add any needed post-processing of the model predictions here
        return inference_output


_service = TransformersClassifierHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e
