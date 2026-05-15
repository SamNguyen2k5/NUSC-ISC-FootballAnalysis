from pathlib import Path
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

st.title("Segmentation Model")
img = st.file_uploader("Upload a screenshot of a football match", type=["jpg", "jpeg", "png"])
if img is not None:
    img_obj = Image.open(img)
    img_arr = np.array(img_obj)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    st.image(Image.fromarray(img_arr))

    payload = NumpyPayload.from_numpy(img_arr)
    response = requests.post(url=config['endpoint'], 
                             json=payload.model_dump(), 
                             timeout=5000)

    print(response.status_code)
    response_payload = NumpyPayload(**response.json())

    mask = response_payload.to_numpy()
    st.image(mask)