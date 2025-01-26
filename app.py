import streamlit as st
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
from pdf2image import convert_from_bytes
from PIL import Image

labels = [
    'budget',
    'email',
    'form',
    'handwritten',
    'invoice',
    'language',
    'letter',
    'memo',
    'news article',
    'questionnaire',
    'resume',
    'scientific publication',
    'specification',
]
id2label = {i: label for i, label in enumerate(labels)}
label2id = {v: k for k, v in id2label.items()}

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3ForSequenceClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

st.title("Document Classification with LayoutLMv3")

# File uploader for PDFs, JPGs, and PNGs
uploaded_file = st.file_uploader(
    "Upload Document", type=["pdf", "jpg", "png"], accept_multiple_files=False
)

if uploaded_file:
    # for uploaded_file in uploaded_files:
    if uploaded_file.type == "application/pdf":
        images = convert_from_bytes(uploaded_file.getvalue())
    else:
        images = [Image.open(uploaded_file)]

    # Process each image for classification
    for i, image in enumerate(images):
        st.image(image, caption=f'Uploaded Image {i}', use_container_width=True)
        # Prepare image for model input
        encoding = processor(
            image,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        outputs = model(**encoding)
        prediction = outputs.logits.argmax(-1)[0].item()

        # Display predictions (you may want to map indices to labels)
        st.write(f"Prediction: {id2label[prediction]}")

        # User feedback section
        feedback = st.radio(
            "Is the classification correct?", ("Yes", "No"),
            key=f'prediction-{i}'
        )
        if feedback == "No":
            correct_label = st.text_input(
                "Please provide the correct label:"
            )
            # Here you can implement logic to store or process feedback
