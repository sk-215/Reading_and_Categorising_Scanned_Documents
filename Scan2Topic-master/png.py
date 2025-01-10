import os
from PIL import Image

img_folder = r'D:/image - Copy'  # Folder containing topic folders (i.e., "News", "Letters", etc.)

for subfol in os.listdir(img_folder):  # For each of the topic folders
    sfpath = os.path.join(img_folder, subfol)

    # Get all TIFF files in the topic folder
    tiff_files = [f for f in os.listdir(sfpath) if f.lower().endswith(('.tif', '.tiff'))]

    for tifffile in tiff_files:
        tiffpath = os.path.join(sfpath, tifffile)
        img = Image.open(tiffpath)  # Read in the TIFF image with Pillow
        filename, ext = os.path.splitext(tifffile)  # Split file name and extension

        # New PNG file path
        newpngpath = os.path.join(sfpath, filename + ".png")

        # Save TIFF image as PNG
        img.save(newpngpath)

        # Remove the original TIFF file
        os.remove(tiffpath)


