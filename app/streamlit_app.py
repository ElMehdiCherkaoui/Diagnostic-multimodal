import requests
import streamlit as st
from PIL import Image, ImageDraw


API_URL = "http://api:8000"

st.title("Brain Tumor Analysis")

image_file = st.file_uploader(
    "Upload a brain image",
    type=["jpg", "jpeg", "png"],
)

option = st.selectbox(
    "Choose an analysis",
    ["Classification", "Detection"],
)

if image_file is not None:
    st.image(image_file, caption="Uploaded image", use_container_width=True)

    if st.button("Analyze"):
        endpoint = "/classify" if option == "Classification" else "/detect"
        files = {
            "file": (
                image_file.name,
                image_file.getvalue(),
                image_file.type,
            )
        }

        response = requests.post(
            API_URL + endpoint,
            files=files,
            timeout=120,
        )

        if response.ok:
            result = response.json()
            st.success("Analysis complete")

            if option == "Classification":
                st.write("Predicted class:", result["class"])
                st.write("Confidence:", result["confidence"])
                st.json(result["probabilities"])
            else:
                image = Image.open(image_file).convert("RGB")
                draw = ImageDraw.Draw(image)
                for detection in result["detections"]:
                    box = detection["box"]
                    draw.rectangle(box, outline="red", width=3)
                    draw.text(
                        (box[0], box[1]),
                        f'{detection["class"]} {detection["confidence"]:.2f}',
                        fill="red",
                    )
                st.image(image, caption="Detection result")
                st.json(result["detections"])
        else:
            try:
                error = response.json().get("detail", response.text)
            except ValueError:
                error = response.text
            st.error(f"The API could not analyze this image: {error}")
