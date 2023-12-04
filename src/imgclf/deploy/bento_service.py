import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

BENTO_MODEL_TAG = "keras_model:hcvtykmsxsyxgogv"

classifier_runner = bentoml.keras.get(BENTO_MODEL_TAG).to_runner()

mnist_service = bentoml.Service("mnist_classifier", runners=[classifier_runner])


@mnist_service.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_data: np.ndarray) -> np.ndarray:
    raw_output = classifier_runner.predict.run(input_data)
    return np.argmax(np.array(raw_output))  # Convert to NumPy array
