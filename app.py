import streamlit as st
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from pdf2image import convert_from_path
from PIL import Image

# Load model and processor
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

st.title("Document Classification with LayoutLMv3")

# File uploader for PDFs, JPGs, and PNGs
uploaded_files = st.file_uploader(
    "Upload Document", type=["pdf", "jpg", "png"], accept_multiple_files=False
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            images = convert_from_path(uploaded_file)
        else:
            images = [Image.open(uploaded_file)]

        # Process each image for classification
        for i, image in enumerate(images):
            st.image(image, caption=f'Uploaded Image {i}', use_column_width=True)
            # Prepare image for model input
            encoding = processor(image, return_tensors="pt")
            outputs = model(**encoding)
            predictions = outputs.logits.argmax(-1)

            # Display predictions (you may want to map indices to labels)
            st.write(f"Predictions: {predictions}")

            # User feedback section
            feedback = st.radio(
                "Is the classification correct?", ("Yes", "No")
            )
            if feedback == "No":
                correct_label = st.text_input(
                    "Please provide the correct label:"
                )
                # Here you can implement logic to store or process feedback
