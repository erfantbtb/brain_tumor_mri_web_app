import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os


class WebApp:
    def __init__(self, model):
        self.model = model
        self.header = st.container()
        self.information = st.container()
        self.results = st.container()

    def heading(self):
        with self.header:
            st.title("Brain Tumor analyzer!")
            st.markdown("---")

    def patient_information(self):
        with self.information:
            st.header("Patient Information")
            info, mri_image = self.information.columns(2)

            with info:
                name = st.text_input('First name: ', '')
                last_name = st.text_input('Last name: ', '')
                age = st.text_input('Age: ', '')
                gender = st.text_input('Gender: ', '')
                ID = st.text_input('ID: ', '')
                background = st.text_input('Background: ', '')

            with mri_image:
                self.uploaded_file = st.file_uploader("Choose your image")

                if self.uploaded_file != None:
                    string_file = self.uploaded_file.name
                    main_path = "images"
                    image_path = os.path.join(main_path, string_file)

                    self.image = Image.open(image_path)
                    self.image = self.image.resize((250, 250))
                    st.image(
                        self.image, caption=f'{name} {last_name} brain tumor mri')

            st.markdown("---")

    def result(self):
        with self.results:
            result_col, button_column = self.results.columns(2)

            with button_column:
                but_res = st.button("Show Results!")

                if self.uploaded_file != None and but_res:
                    image_arr = self.image.resize((150, 150))
                    image_arr = np.array(image_arr).astype(
                        "float32").reshape(1, 150, 150, 3)
                    image_arr = image_arr/255.

                    idx = self.model.predict(image_arr)[0].argmax(axis=0)
                    max_prob = self.model.predict(image_arr)[0].max()
                    labels = ["glioma", "meningioma", "no tumor", "pituitary"]
                    predicted_label = labels[idx]

            with result_col:
                if self.uploaded_file != None and but_res:
                    st.markdown(
                        f"<p> The patient has {predicted_label} with probability of {np.round(max_prob, 3)} </p>",
                        unsafe_allow_html=True)


if __name__ == "__maine__":
    model = tf.keras.models.load_model(
        "H:/computer_vision/machine_learning_projects/brain_tumor_mri_project/deployment_model")
    print(model.summary())
    wa = WebApp(model)
    wa.heading()
    wa.patient_information()
    wa.result()
