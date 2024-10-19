import tensorflow as tf
from jsonargparse import CLI

from src.config_class import PredictConfig
from src.predict_hairstyle import (
    load_image_to_predict,
    preprocess_image_for_model,
    make_prediction,
)


if __name__ == "__main__":
    config = CLI(PredictConfig)

    model_path = config.predict_params.model_path
    image_dir_path = config.predict_params.image_dir_path
    image_name = config.predict_params.image_name

    image = load_image_to_predict(image_dir_path, image_name)
    preprocessed_image = preprocess_image_for_model(image)

    model = tf.keras.models.load_model(model_path)

    prediction = make_prediction(model, preprocessed_image)
    print(f"Predicted label: {prediction}")
