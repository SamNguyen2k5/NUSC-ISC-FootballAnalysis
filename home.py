import streamlit as st

segmentation_page = st.Page(
    'app/segmentation/app.py', 
    title='Segmentation',
    url_path='segmentation'
)
spiideo_page = st.Page(
    'app/spiideo/app.py', 
    title='Spiideo',
    url_path='spiideo'
)

pg = st.navigation([
    segmentation_page,
    spiideo_page,
])

pg.run()