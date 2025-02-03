import os
import cv2
import imghdr
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import image_dataset_from_directory
import shutil

app = FastAPI()

# Global Variables
model = None
class_names = None

# Directory containing the dataset
data_dir = 'data'
image_exts = ['jpeg', 'jpg', 'png', 'bmp']






@app.post("/train")
async def train_model():
    """
    Endpoint to train the model.
    """
    global model, class_names

    # Clean up invalid image files
    for img_class in os.listdir(data_dir):
        for image in os.listdir(os.path.join(data_dir, img_class)):
            image_path = os.path.join(data_dir, img_class, image)
            try:
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts:
                    os.remove(image_path)
            except Exception as e:
                print(f"Issue with image: {image_path} - {str(e)}")

    # Load dataset
    data = image_dataset_from_directory(data_dir, image_size=(256, 256), batch_size=32)
    class_names = data.class_names

    # Preprocess dataset
    def preprocess_data(image, label):
        image = tf.cast(image, tf.float32) / 255.0  # Normalize the images
        return image, label

    data = data.map(preprocess_data)

    # Split data into train, validation, and test sets
    data = data.unbatch()
    images = []
    labels = []

    for image, label in data:
        images.append(image.numpy())
        labels.append(label.numpy())

    images = np.array(images)
    labels = np.array(labels)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Build the model
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        MaxPooling2D(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Dropout to prevent overfitting
        Dense(len(class_names), activation='softmax')  # Output layer for multi-class classification
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32
    )

    # Save the trained model
    model.save('food_identification_model.keras')
    return {"message": "Model training completed and saved successfully."}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Endpoint to predict the class of an uploaded image.
    """
    global model, class_names

    if model is None or class_names is None:
        raise HTTPException(status_code=400, detail="Model is not loaded. Train the model first using the /train endpoint.")

    # Save the uploaded file temporarily
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)

    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Read and preprocess the image
        img = cv2.imread(temp_file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = tf.image.resize(img, (256, 256)) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_resized, axis=0)  # Expand dimensions for batch size

        # Predict the class
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        return {"predicted_class": predicted_class, "confidence": round(float(confidence), 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
    finally:
        # Clean up temporary file
        os.remove(temp_file_path)


# Health Check Endpoint
@app.get("/")
async def root():
    return {"message": "FastAPI Food Identification App is running!"}
