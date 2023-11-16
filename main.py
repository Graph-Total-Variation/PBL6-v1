import shutil
from io import BytesIO
from fastapi import File, UploadFile, FastAPI, Form, HTTPException
from pathlib import Path
import hashlib
import requests
import numpy as np
import os
from keras.utils import load_img, img_to_array
from keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import csv
import cv2

from deepgtv.runcc import *
# from deepgtv import *
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các origin
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức
    allow_headers=["*"]  # Cho phép tất cả các header
)

upload_folder = Path("uploads")
upload_folder.mkdir(parents=True, exist_ok=True)

gtv_model = load_gtv_model("deepgtv/model/GTV_13g5.pkl")

def generate_hash(file_content):
    hash_object = hashlib.sha256()
    hash_object.update(file_content)
    return hash_object.hexdigest()


# resize ảnh
def resize_image(image_path, target_size=(512, 512)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    cv2.imwrite(image_path, img)  # Lưu ảnh sau khi thay đổi kích thước


# implement model denoise
def denoise_image(image_path):
    # Load the image
    # img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Convert the image to the format expected by the denoise model (e.g., scaling to [0, 1])
    # img = img.astype(np.float32) / 255.0

    
    # Apply denoising using the model
    denoised_img = denoise_image(image_path, gtv_model)

    # Convert the denoised image back to [0, 255] range
    # denoised_img = (denoised_img * 255.0).astype(np.uint8)

    return denoised_img

# Example usage:
# denoised_image = denoise_image("input_image.jpg", denoise_model)
# cv2.imwrite("denoised_image.jpg", denoised_image)

# implement model


def preprocess_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    x /= 255
    return x


def predict_image(image_data, model_path):
    # Load model
    model = load_model(model_path)
    # Make prediction
    prediction = model.predict(image_data)
    # Get predicted class
    predicted_class = np.argmax(prediction)

    if predicted_class == 0:
        predicted_label = "Human"
    else:
        predicted_label = "AI"
    return predicted_label


def save_image(response, filename):
    with open(filename, "wb") as f:
        f.write(response.content)

# face detection


def contains_human_face(image_path):
    try:
        img = cv2.imread(image_path)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        faces = face_classifier.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        return len(faces) > 0
    except Exception as e:
        print("Error during face detection:", str(e))
        return False


@app.get("/predict_url")
async def say_hello(url: str):
    try:
        print(f"URL: {url}")
        response = requests.get(url)

        # get filename
        h = hashlib.sha256()
        h.update(response.content)
        checksum = h.hexdigest()
        file_path = os.path.join(upload_folder, f"{checksum}.jpg")

        save_image(response, file_path)
        with open(file_path, "rb") as f:
            content = f.read()

        # Process the uploaded image
        processed_image = preprocess_image(str(file_path))

        # Load model from file
        model_path = 'models/densenet.h5'

        # Predict image
        predicted_class = predict_image(processed_image, model_path)

        # Print image path and predicted class
        print("Image path:", file_path)
        print("Predicted class:", predicted_class)

        save = 0
        if predicted_class == "AI":
            save = 1

        file_path_str = str(file_path)
        # Save to CSV
        if not os.path.exists('predict.csv') or file_path_str not in open('predict.csv').read():
            with open('predict.csv', mode='a', newline='') as file:
                writer = csv.writer(file, delimiter=',',
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([file_path, save])

        return {"result": predicted_class, "url": f"{url}", "filepath": f"{str(file_path)}", "error": "none"}

    except Exception as e:
        return {"result": "", "url": "", "filepath": '', "error": f"Error: {str(e)}"}


@app.post("/upload_and_process")
async def upload_and_process(file: UploadFile = File(...)):
    try:
        # Lưu file được gửi từ website
        file_content = await file.read()
        hash_filename = generate_hash(file_content)
        file_path = upload_folder / f"{hash_filename}.jpg"

        with open(file_path, "wb") as buffer:
            buffer.write(file_content)

        if contains_human_face(file_path):
            # Nếu ảnh chứa khuôn mặt, tiếp tục xử lý và dự đoán
            processed_image = preprocess_image(str(file_path))
            model_path = 'models/densenet.h5'
            predicted_class = predict_image(processed_image, model_path)

            save = 0
            if predicted_class == "AI":
                save = 1

            file_path_str = str(file_path)
            # Lưu vào CSV nếu file_path chưa tồn tại trong predict.csv
            if not os.path.exists('predict.csv') or file_path_str not in open('predict.csv').read():
                with open('predict.csv', mode='a', newline='') as file:
                    writer = csv.writer(
                        file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([file_path, save])

            return {"result": predicted_class, "filepath": f"{file_path_str}", "error": 'none'}
        else:
            return {"result": '', "filepath": "", "error": "Uploaded image is not a face!"}

    except Exception as e:
        return {"result": '', "filepath": "", "error": f"Error: {str(e)}"}

# button correct


@app.post("/upload_and_denoise")
async def upload_and_denoise(file: UploadFile = File(...)):
    try:
        # Kiểm tra xem tệp gửi lên phải là ảnh PNG
        if not file.filename.endswith(".png"):
            return {"error": "Only PNG files are supported."}

        # Lưu file được gửi từ website
        file_content = await file.read()
        hash_filename = generate_hash(file_content)
        # Lưu lại với định dạng PNG
        file_path = upload_folder / f"{hash_filename}.png"

        with open(file_path, "wb") as buffer:
            buffer.write(file_content)

        # Denoise ảnh
        # model_path = 'model/densenet.h5'
        # denoised_image = denoise_image(file_path, model_path)

        ref_image_path = Path("images") / "ref.png"

        # Lưu ảnh đã được denoise
        denoised_file_path = upload_folder / f"{hash_filename}_denoised.png"
        # cv2.imwrite(str(denoised_file_path), denoised_image)

        # Trả về kết quả dưới dạng tệp ảnh
        # return FileResponse(str(denoised_file_path))
        return FileResponse(ref_image_path)

    except Exception as e:
        # return {"error": f"Error: {str(e)}"}
        return {"result": '', "filepath": "", "error": f"Error: {str(e)}"}


def save_feedback_to_csv(image_path: str, feedback: int):
    rows = []

    with open('predict.csv', mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            while len(row) < 3:
                row.append('')
            rows.append(row)

    found = False
    for row in rows:
        if row[0] == image_path:
            row[2] = feedback
            found = True
            break

    if not found:
        new_row = [image_path, '', feedback]
        rows.append(new_row)

    with open('predict.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)


@app.post("/submit_feedback")
async def submit_feedback(feedback_data: dict):
    feedback = feedback_data.get("feedback")
    image_path = feedback_data.get("image_path")
    print(feedback)
    print(image_path)
    try:
        feedback = int(feedback)
        if feedback not in [0, 1]:
            return {"message": "ok"}
    except ValueError:
        return {"message": "err"}

    try:
        save_feedback_to_csv(image_path, feedback)
    except Exception as e:
        return {"message": f"err: {str(e)}"}

    return {"message": "done"}


# @app.post("/uploadfile/")
# async def upload_file(file: UploadFile):
#     # Lưu tệp tải lên tạm thời
#     with open(f"temp_{file.filename}", "wb") as temp_file:
#         shutil.copyfileobj(file.file, temp_file)

#     # Thực hiện xử lý ảnh ở đây
#     # Đảm bảo bạn có hàm hoặc module xử lý ảnh để thực hiện công việc này

#     # Trả kết quả (ví dụ: nội dung ảnh đã xử lý)
#     return {"filename": file.filename, "size": file.file.read()}
