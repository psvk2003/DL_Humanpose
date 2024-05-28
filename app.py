import streamlit as st
from PIL import Image
import humanpose

def main():
    st.title('Human Pose Estimation')

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform human pose estimation
        pose_img = humanpose.process_image(image)

        # Display the result
        st.image(pose_img, caption='Human Pose Estimation', use_column_width=True)

if __name__ == '__main__':
    main()
