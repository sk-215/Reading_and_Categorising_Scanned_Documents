import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from doctr.file_utils import is_tf_available
from doctr.io import DocumentFile
from doctr.utils.visualization import visualize_page

# os.add_dll_directory(r"C:\Program Files\GTK3-Runtime Win64\bin")
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

# Folder containing topic folders (i.e., "News", "Letters", etc.)
img_folder = r'/mnt/c/Users/Santha Kumaran/PycharmProjects/doctr-main/image - Copy/'

# Iterate through each topic folder
for subfol in os.listdir(img_folder):
    sfpath = os.path.join(img_folder, subfol)

    # Get all PNG files in the topic folder
    png_files = [f for f in os.listdir(sfpath) if f.lower().endswith('.png')]

    # Iterate through each PNG file in the topic folder
    for pngfile in png_files:
        pngpath = os.path.join(sfpath, pngfile)

        # Read in the PNG image with Pillow
        img = Image.open(pngpath)

        # Split file name and extension
        filename, ext = os.path.splitext(pngfile)

        # New PNG file path (no change in file path)
        newpngpath = os.path.join(sfpath, filename + ".png")
        print(newpngpath)
        print("\n")

        # Load the document file from the image file
        doc = DocumentFile.from_images([newpngpath])
        # page_idx = st.sidebar.selectbox("Page selection", [idx + 1 for idx in range(len(doc))]) - 1
        page = doc[0]

        #Loading model
        predictor = load_predictor(
            "db_resnet50", "crnn_vgg16_bn", True, False, 0.3, 0.1, forward_device
        )

        #ocr output
        out = predictor([page])
        words_string = ""
        for page in out.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        words_string += word.value + " "

        output_text_file = os.path.splitext(pngpath)[0] + ".txt"

        # Write the extracted text to a text file
        with open(output_text_file, "w") as text_file:
            text_file.write(words_string)
        os.remove(pngpath)