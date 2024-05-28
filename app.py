import streamlit as st
from PIL import Image
import humanpose
import os

st.title('Human Pose Estimation')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img_path = os.path.join("temp", uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(Image.open(img_path), caption='Uploaded Image.', use_column_width=True)
    st.write("Processing...")

    keypoints_img, skeleton_img, output_folder = humanpose.process_image(img_path)

    st.image(keypoints_img[:, :, ::-1], caption='Keypoints Image', use_column_width=True)
    st.image(skeleton_img[:, :, ::-1], caption='Skeleton Image', use_column_width=True)
