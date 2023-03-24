import json
from flask import Blueprint, render_template, flash, request
import base64
from .digit_recognition_ai import recognize
from PIL import Image
import io
import os
import numpy as np

views = Blueprint("views", __name__)


@views.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")


@views.route("/recognize", methods=["POST"])
def upload_image():
    data = request.get_json()
    image_string = data["image"]
    image_data = base64.b64decode(image_string.split(",")[1])
    image = Image.open(io.BytesIO(image_data)).convert("LA")
    img_data = list(image.resize((28, 28)).getdata())
    img_data = np.array([p[1] / 255 for p in img_data])
    img_data = img_data.reshape((1, 28, 28))
    result = recognize(img_data)

    return json.dumps({"message": "message received", "prediction": result.tolist()[0]})
