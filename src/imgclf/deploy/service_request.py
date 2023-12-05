import json
import numpy as np
import requests

from imgclf.data import load_data

SERVICE_URL = "http://localhost:3000/classify"


def make_request_to_bento_service(service_url: str, input_array: np.ndarray) -> str:
    serialized_input_data = json.dumps(input_array.tolist())
    response = requests.post(
        service_url,
        data=serialized_input_data,
        headers={"content-type": "application/json"}
    )
    return response.text


def main():
    input_data, expected_output = load_data.sample_random_mnist_data_point()
    prediction = make_request_to_bento_service(SERVICE_URL, input_data)
    print(f"Prediction: {prediction}")
    print(f"Expected output: {expected_output}")


if __name__ == "__main__":
    main()
