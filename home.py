import streamlit as st

segmentation_page = st.Page('app/segmentation/app.py', title='Segmentation')

pg = st.navigation([
    segmentation_page,
])

pg.run()