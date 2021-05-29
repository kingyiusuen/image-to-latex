import requests
from PIL import Image

import streamlit as st


st.set_page_config(page_title="Image To Latex Converter")


st.title("Image to Latex Converter")


uploaded_file = st.file_uploader(
    "Upload an image of latex math equation",
    type=["png", "jpg"],
)


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width="auto")


beam_width = st.sidebar.slider(
    "Beam width",
    min_value=1,
    max_value=10,
    value=5,
    help=(
        "The number of best alternatives to consider at each decoding step. "
        "A larger beam width may result in a better prediction but will lead "
        "to a slower inference speed."
    ),
)


if st.button("Convert"):
    if uploaded_file is not None and image is not None:
        files = {"file": uploaded_file.getvalue()}
        with st.spinner("Wait for it..."):
            response = requests.post(
                "http://0.0.0.0:8000/predict", files=files
            )
        st.header("Output")
        latex_code = response.json()["data"]["pred"]
        st.code(latex_code)
        st.markdown(f"${latex_code}$")
    else:
        st.error("Please upload an image.")
