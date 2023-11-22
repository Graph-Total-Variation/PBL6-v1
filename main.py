import requests
from requests.auth import HTTPBasicAuth
import io
from fastapi import File, UploadFile, FastAPI, Form, HTTPException
from pathlib import Path
import hashlib
import requests
import numpy as np
import os
from keras.utils import load_img, img_to_array
from keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
import csv
import cv2
import tempfile
import base64
import pyrebase


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
# gtv_model99 = load_gtv_model("deepgtv/model/GTV_99.pkl")

Config = {
  "apiKey": "AIzaSyAk7msp7PhRI0Tx8twH4XoLAw8_ITo_sqQ",
  "authDomain": "pbl6-a6a2b.firebaseapp.com",
  "projectId": "pbl6-a6a2b",
  'databaseURL':"https://pbl6-a6a2b-default-rtdb.firebaseio.com",
  "storageBucket": "pbl6-a6a2b.appspot.com",
  "messagingSenderId": "1034015281754",
  "appId": "1:1034015281754:web:046c04e0d4bcc89204fe63",
  "measurementId": "G-P76MGJZH33",
  "serviceAccount": "uploads/pbl6-a6a2b-firebase-adminsdk-5kepy-144470f96a.json"
};

firebase = pyrebase.initialize_app(Config)
# database = firebase.database()
storage = firebase.storage()

def generate_hash(file_content):
    hash_object = hashlib.sha256()
    hash_object.update(file_content)
    return hash_object.hexdigest()


# resize ảnh
def resize_image(image_path, target_size=(512, 512)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    cv2.imwrite(image_path, img)  # Lưu ảnh sau khi thay đổi kích thước

#up anh len firebase storage 
def upload_save(image_data, image_name):
    pth = "result/" + image_name
    storage.child(image_name).put(image_data, content_type="image/png")
    image_url = storage.child(image_name).get_url(image_name)
    return image_url



def preprocess_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    # x /= 255
    return x




def save_image(response, filename):
    with open(filename, "wb") as f:
        f.write(response.content)

# face detection

def image_to_base64(image_np):
    # Chuyển đổi mảng NumPy thành đối tượng PIL
    image_pil = Image.fromarray(np.uint8(image_np))

    # Tạo đối tượng BytesIO để lưu trữ dữ liệu base64
    buffer = io.BytesIO()

    # Lưu ảnh dưới dạng base64 vào đối tượng BytesIO
    image_pil.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return base64_image



@app.get("/predict_url")
async def predict_url(url: str):
    try:
        print(f"URL: {url}")
        response = requests.get(url)

        # get filename
        h = hashlib.sha256()
        h.update(response.content)
        checksum = h.hexdigest()
        file_path = os.path.join(upload_folder, f"temp.png")

        save_image(response, file_path)
        with open(file_path, "rb") as f:
            content = f.read()

        # Process the uploaded image
        processed_image = preprocess_image(str(file_path))

        img = denoise_image(str(file_path),gtv_model)

        link = upload_save(img,checksum)

        return JSONResponse(content={ "filepath": link, "error": None})
    except Exception as e:
        return JSONResponse(content={ "filepath": "error", "error": f"Error: {str(e)}"})




@app.post("/upload_and_denoise")
async def upload_and_denoise(file: UploadFile = File(...)):
    try:

        # Lưu file được gửi từ website
        file_content = await file.read()

        # hash_filename = generate_hash(file_content)
        hash_filename = generate_hash(file_content)
        temp_file_path = upload_folder / f"temp.png"
        # temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        with open(temp_file_path, "wb") as buffer:
            buffer.write(file_content)

        img_patch = str(temp_file_path)
        img1 = denoise_image(img_patch, gtv_model)
        # img2 = denoise_image2(file_content, gtv_model)
        link = upload_save(img1,hash_filename)

        return JSONResponse(content={ "filepath": link, "error": None})
    

    except Exception as e:
        return JSONResponse(content={ "filepath": "error", "error": f"Error: {str(e)}"})
    
@app.post("/get_image_base64")
async def get_image_base64(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        hash_filename = generate_hash(file_content)
        file_path = upload_folder / f"temp.png"

        with open(file_path, "wb") as buffer:
            buffer.write(file_content) 

        img_pth = upload_save(file_content,hash_filename)
        
        return JSONResponse(content={"filepath": img_pth, "error": None})

    except Exception as e:
        # Xử lý nếu có lỗi
        return JSONResponse(content={"filepath":None, "error": f"Error: {str(e)}"})


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

