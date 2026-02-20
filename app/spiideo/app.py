from pathlib import Path
from pydantic import ValidationError
import yaml

import requests
import cv2
from PIL import Image
import numpy as np
import streamlit as st

from apis.payload import NumpyPayload

with Path(__file__).parent.joinpath('config.yaml').open('r', encoding='utf-8') as f: 
    config = yaml.safe_load(f)
    print(config)

st.title("Spiideo Dataset Visualiser")
idx = st.number_input('ID: ', min_value=0, step=1)
payload = idx
response = requests.post(url=config['endpoint'], 
                            json=payload,
                            timeout=5000)

print(response.status_code)
try:
    response_payload = NumpyPayload(**response.json())
    img = response_payload.to_numpy()
    st.image(img, width=960)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    payload_2 = NumpyPayload.from_numpy(img_gray)
    response_2 = requests.post(
        url=config['endpoint_segmentation'], json=payload_2.model_dump(), timeout=5000)

    print(response_2)

    response_payload_2 = NumpyPayload(**response_2.json())
    img_2 = response_payload_2.to_numpy()
    st.image(img_2, width=960)


except ValidationError as e:
    st.write('No images are returned.') 