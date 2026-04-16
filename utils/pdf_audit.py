import os
import sys
import fitz  # PyMuPDF
from PIL import Image


def render_pdf_to_images(pdf_path, output_folder="pdf_pages"):
    """
    Converts PDF pages to images using PyMuPDF (fitz).
    Self-contained, does not require poppler.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"Rendering {pdf_path} with PyMuPDF...")

    try:
        doc = fitz.open(pdf_path)
        image_paths = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Higher zoom for better readability (300 DPI approx)
            zoom = 2.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            path = os.path.join(output_folder, f"page_{page_num + 1}.png")
            pix.save(path)
            image_paths.append(path)

        doc.close()
        print(f"Successfully rendered {len(image_paths)} pages.")
        return image_paths

    except Exception as e:
        print(f"PDF Rendering Error: {e}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--out", default="pdf_pages", help="Output folder for images")
    args = parser.parse_args()

    render_pdf_to_images(args.pdf_path, args.out)
