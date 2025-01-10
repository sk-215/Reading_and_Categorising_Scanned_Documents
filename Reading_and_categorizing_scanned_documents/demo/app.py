import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import sys
sys.path.append(r'C:\Users\Santha Kumaran\PycharmProjects\doctr-main')
from doctr.file_utils import is_tf_available
from doctr.io import DocumentFile
from doctr.utils.visualization import visualize_page
import shutil
import os
import joblib
# Load the saved model
loaded_model = joblib.load('text_classification_model.joblib')
def classify_text(input_text):
    predicted_label = loaded_model.predict([input_text])[0]
    return predicted_label

if is_tf_available():
    import tensorflow as tf
    from backend.tensorflow import DET_ARCHS, RECO_ARCHS, forward_image, load_predictor

    if any(tf.config.experimental.list_physical_devices("gpu")):
        forward_device = tf.device("/gpu:0")
    else:
        forward_device = tf.device("/cpu:0")

else:
    import torch
    from backend.pytorch import DET_ARCHS, RECO_ARCHS, forward_image, load_predictor

    forward_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(det_archs, reco_archs):
    """Build a streamlit layout"""
    flag = 0
    # Wide mode
    st.set_page_config(layout="wide")

    # Designing the interface
    st.title("Document Text Recognition and Categorization")
    # For newline
    st.write("\n")
    # Instructions
    st.markdown("*Hint: click on the top-right corner of an image to enlarge it!*")
    # Set the columns
    cols = st.columns((1, 1, 1))
    cols[0].subheader("Input page")
    cols[1].subheader("Segmentation heatmap")
    cols[2].subheader("OCR output")
    # cols[3].subheader("Page reconstitution")

    # Sidebar
    # File selection
    st.sidebar.title("Document selection")
    # Choose your own image
    uploaded_file = st.sidebar.file_uploader("Upload files", type=["pdf", "png", "jpeg", "jpg"])
    if uploaded_file is not None:
        file_name = uploaded_file.name
        if uploaded_file.name.endswith(".pdf"):
            flag=1
            doc = DocumentFile.from_pdf(uploaded_file.read())
        else:
            doc = DocumentFile.from_images(uploaded_file.read())
        page_idx = st.sidebar.selectbox("Page selection", [idx + 1 for idx in range(len(doc))]) - 1
        page = doc[page_idx]
        cols[0].image(page)

    # For newline
    st.sidebar.write("\n")

    if st.sidebar.button("Analyze page"):
        if uploaded_file is None:
            st.sidebar.write("Please upload a document")

        else:
            with st.spinner("Loading model..."):
                predictor = load_predictor(
                    "db_resnet50", "crnn_vgg16_bn", True, False, 0.3, 0.1, forward_device
                )

            with st.spinner("Analyzing..."):
                # Forward the image to the model
                seg_map = forward_image(predictor, page, forward_device)
                seg_map = np.squeeze(seg_map)
                seg_map = cv2.resize(seg_map, (page.shape[1], page.shape[0]), interpolation=cv2.INTER_LINEAR)

                # Plot the raw heatmap
                fig, ax = plt.subplots()
                ax.imshow(seg_map)
                ax.axis("off")
                cols[1].pyplot(fig)

                # Plot OCR output
                out = predictor([page])
                fig = visualize_page(out.pages[0].export(), out.pages[0].page, interactive=False, add_labels=False)
                cols[2].pyplot(fig)
                words_string = ""
                for page in out.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            for word in line.words:
                                words_string += word.value + " "
                            words_string += "\n"
                # st.sidebar.write(out)
                # st.sidebar.write(words_string)
                if flag == 1:
                    file_name = file_name[:-4] + "_" + str(page_idx+1) + ".txt"
                else:
                    file_name = file_name[:-4] + ".txt"
                topic_name = ["email", "questionnaire"]
                folder_paths = ['/mnt/d/image_to_text - Copy/email/', '/mnt/d/image_to_text - Copy/questionnaire/']
                labels = [0, 1]
                predicted_label = classify_text(words_string)
                destination_folder = folder_paths[predicted_label]
                new_path = os.path.join(destination_folder, file_name)
                with open(new_path, 'w') as file:
                  file.write(words_string)

                st.markdown("\nExtracted Text:")
                st.write(words_string)

                st.markdown(f"\nTopic name : {topic_name[predicted_label]}")


if __name__ == "__main__":
    main(DET_ARCHS, RECO_ARCHS)
