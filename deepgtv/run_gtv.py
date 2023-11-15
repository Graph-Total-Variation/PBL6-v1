import sys
import pickle
import torch
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import argparse
from train_gtv import *
import logging
from skimage.metrics import structural_similarity as compare_ssim
from test_gtv import *

def load_gtv_model(model_path):
    # Tạo đối tượng OPT để cấu hình
    opt = OPT(
        batch_size=32,
        channels=1,
        lr=1e-4,
        momentum=0.9,
        u_max=1000,
        u_min=0.0001,
        cuda=True if torch.cuda.is_available() else False
    )
    
    # Thiết lập logger
    import logging
    logger = logging.getLogger("root")
    logger.addHandler(logging.StreamHandler(sys.stdout))
    opt.logger = logger
    opt.legacy = True
    
    # Chuẩn bị ma trận hỗ trợ
    supporting_matrix(opt)
    
    # Tạo đối tượng GTV
    gtv = GTV(width=36, cuda=opt.cuda, opt=opt)
    device = torch.device("cuda") if cuda else torch.device("cpu")
    # Load trọng số đã được đào tạo
    gtv.load_state_dict(torch.load(model_path, map_location=device))
    gtv.cuda()

    return gtv

modelgtv = load_gtv_model("model/GTV_13g5.pkl")

def denoise_image(image_path, model = modelgtv, width=512, stride=9):
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    # Đọc ảnh đầu vào
    sample = cv2.imread(image_path)
    sample.cuda()
    # Resize ảnh đầu vào nếu cần
    if width is not None:
        sample = cv2.resize(sample, (width, width))
    
    # Chuyển đổi sang định dạng phù hợp (RGB -> Grayscale)
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    sample = np.expand_dims(sample, axis=2)
    sample = sample.transpose((2, 0, 1))
    
    # Chuyển sang tensor và thêm chiều batch
    sample = torch.from_numpy(sample)
    sample = sample.unsqueeze(0).float()
    
    # Chuyển sang GPU nếu sử dụng CUDA
    if cuda:
        sample = sample.cuda()

    # Sử dụng mô hình để dự đoán
    with torch.no_grad():
        prediction = model(sample)

    # Chuyển kết quả về CPU và chuyển đổi sang numpy array
    prediction = prediction.cpu().numpy()

    # Giải mã hình ảnh từ tensor về định dạng hình ảnh gốc
    prediction = np.squeeze(prediction)
    prediction = np.clip(prediction, 0, 1)  # Đảm bảo giá trị pixel nằm trong khoảng [0, 1]

    # Đánh giá chất lượng của ảnh đã giải nhiễm so với ảnh gốc
    # Tùy thuộc vào yêu cầu của bạn, bạn có thể lưu lại các thông số này hoặc không
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    _, ssim = compare_ssim(original_image, prediction, full=True)

    # Tạo đường dẫn đến thư mục kết quả
    if not os.path.exists("result"):
        os.makedirs("result")

    # Lưu ảnh đã giải nhiễm vào thư mục kết quả
    result_path = os.path.join("result", "denoised_image.png")
    plt.imsave(result_path, prediction, cmap='gray')

    return prediction, ssim
