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
    image = Image.open(io.BytesIO(image_data))
    result = recognize(image)

    return json.dumps({"message": "message received", "prediction": result.tolist()})
