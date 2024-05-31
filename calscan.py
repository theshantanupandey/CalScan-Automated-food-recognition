import streamlit as st
from PIL import Image

from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

import requests

preprocess_functions = {
    "MobileNetV3Large": preprocess_input,
}

models = {
    "MobileNetV3Large": MobileNetV3Large,
}

def preprocess_image(img, model_name):
    # img_array = np.resize(img, input_size[model_name])
    img = img.resize((224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocess_fn = preprocess_functions[model_name]
    img_array = preprocess_fn(img_array)
    return img_array

def predict_food_item(img, model_name):
    model = models[model_name](weights='imagenet')
    preprocessed_img = preprocess_image(img, model_name)
    predictions = model.predict(preprocessed_img)
    decoded_predictions = decode_predictions(predictions, top=3)[0]  # Get top 3 predictions
    return decoded_predictions

# Streamlit app
def main():
    st.title("CalScan: Food Image Analyzer")

    # Upload image
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True, width=300)

    # Analyze button
    if st.button("Analyze"):
        if uploaded_image is not None:

            prediction = predict_food_item(image, 'MobileNetV3Large')

            im, lab, sc = prediction[0]

            url = "https://api.nal.usda.gov/fdc/v1/foods/search"
            api_key = "oRarskrCVUx3h2wP76Two47medu4Gn9bZrd6Fe4m"
            params = {
                "query": lab,
                "api_key": api_key
            }

            try:
                response = requests.get(url, params=params)
                response.raise_for_status()  # Raise error for bad responses

                data = response.json()

                if "foods" in data and data["foods"]:
                    food_item = data["foods"][0]

                    # Extracting relevant information
                    description = food_item["description"]
                    nutrients = food_item["foodNutrients"]
                    st.subheader(f"Prediction: {lab}")
                    st.subheader("\nNutritional Information:")
                    for nutrient in nutrients:
                        st.write(nutrient["nutrientName"], "-", nutrient["value"], nutrient["unitName"])
                else:
                    print("No information found for the specified food.")
            except requests.exceptions.HTTPError as err:
                print("HTTP Error:", err)
            except Exception as err:
                print("Error:", err)

if __name__ == "__main__":
    main()