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


cuda = True if torch.cuda.is_available() else False
if cuda:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

resroot = "result"

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-w",
        "--width",
        help="Resize image to a square image with given width",
        type=int,
    )
    parser.add_argument("-m", "--model")
    parser.add_argument("--stride", default=18, type=int)
    parser.add_argument(
        "--multi", default=30, type=int, help="# of patches evaluation in parallel"
    )
    parser.add_argument("--opt", default="opt")
    parser.add_argument("--image_path_train")
    parser.add_argument("--image_path_test")
    parser.add_argument("--image_path")
    parser.add_argument("--layers", default=1, type=int)
    parser.add_argument("--row", default=100, type=int)
    args = parser.parse_args()
    return args
    
def denoise(
    inp,
    gtv,
    argref,
    normalize=False,
    stride=36,
    width=324,
    prefix="_",
    verbose=0,
    opt=None,
    approx=False,
    args=None,
    logger=None,
):
    try:
        from skimage.metrics import structural_similarity as compare_ssim
    except Exception:
        from skimage.measure import compare_ssim

    sample = cv2.imread(inp)
    if width is None:
        width = sample.shape[0]
    else:
        sample = cv2.resize(sample, (width, width))
    sample = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
    sample = np.expand_dims(sample, axis=2)
    sample = sample.transpose((2, 0, 1))
    shape = sample.shape

    # if normalize:
    #     sample = _norm(sample, newmin=0, newmax=1)
    sample = torch.from_numpy(sample)

    cuda = True if torch.cuda.is_available() else False

    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    if argref:
        ref = cv2.imread(argref)
        if ref.shape[0] != width or ref.shape[1] != width:
            ref = cv2.resize(ref, (width, width))
        ref = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
        #ref = np.expand_dims(ref, axis=2)
        ref_p = resroot + "/ref_" + argref.split("/")[-1]
        plt.imsave(ref_p, ref,cmap='gray')
        ref = np.expand_dims(ref, axis=2)
        logger.info(ref_p)
        tref = ref.copy()
        ref = ref.transpose((2, 0, 1))
        ref = torch.from_numpy(ref)
        # if normalize:
        #     ref = _norm(ref, newmin=0, newmax=1)

    tstart = time.time()
    T1 = sample
    if argref:
        T1r = ref

    T1 = torch.nn.functional.pad(T1, (0, stride, 0, stride), mode="constant", value=0)
    shapex = T1.shape
    T2 = (
        torch.from_numpy(T1.detach().numpy().transpose(1, 2, 0))
        .unfold(0, opt.width, stride)
        .unfold(1, opt.width, stride)
    ).type(dtype)
    T2 = T2.contiguous()
    if argref:
        T1r = torch.nn.functional.pad(
            T1r, (0, stride, 0, stride), mode="constant", value=0
        )
    # MAX_PATCH = args.multi
    MAX_PATCH = 36 
    oT2s0 = T2.shape[0]
    T2 = T2.view(-1, opt.channels, opt.width, opt.width)
    dummy = torch.zeros(T2.shape).type(dtype)
    logger.info("{0}".format(T2.shape))
    with torch.no_grad():
        for ii, i in enumerate(range(0, T2.shape[0], MAX_PATCH)):
            P = gtv.predict(
                T2[i : (i + MAX_PATCH), :, :, :].float().contiguous().type(dtype),
            )
            dummy[i : (i + MAX_PATCH)] = P
    dummy = dummy.view(oT2s0, -1, opt.channels, opt.width, opt.width)
    dummy = dummy.cpu()
    if verbose:
        logger.info("Prediction time: {0}".format(time.time() - tstart))
    else:
        logger.info("Prediction time: {0}".format(time.time() - tstart))

    dummy = (
        patch_merge(dummy, stride=stride, shape=shapex, shapeorg=shape).detach().numpy()
    )

    ds = np.array(dummy).copy()
    d = np.minimum(np.maximum(ds, 0), 255)
    logger.info("RANGE: {0} - {1}".format(d.min(), d.max()))
    d = d.transpose(1, 2, 0) / 255
    d = d[:,:,0]
    if 0:
        opath = args.output
    else:
        filename = inp.split("/")[-1]
        opath = resroot + "/{0}_{1}".format(prefix, filename)
        opath = opath[:-3] + "png"
    d = np.minimum(np.maximum(d, 0), 1)
    plt.imsave(opath, d, cmap='gray')
    if argref:
        mse = ((d - (tref / 255.0)) ** 2).mean() * 255
        logger.info("MSE: {:.5f}".format(mse))
        d = cv2.imread(opath)
        d = cv2.cvtColor(d, cv2.COLOR_RGB2GRAY)
        d = np.expand_dims(d, axis=2)
        psnr2 = cv2.PSNR(tref, d)
        logger.info("PSNR: {:.5f}".format(psnr2))
        #(score, diff) = compare_ssim(tref, d, full=True, channel_axis=True)
        (score, diff) = compare_ssim(tref[:,:,0], d[:,:,0], full=True)
        logger.info("SSIM: {:.5f}".format(score))
    logger.info("Saved {0}".format(opath))
    if argref:

        return (0, score, 0, psnr2, mse, d)  # psnr, ssim, denoised image
    return d


def patch_merge(P, stride=36, shape=None, shapeorg=None):
    S1, S2 = P.shape[0], P.shape[1]
    m = P.shape[-1]

    R = torch.zeros(shape)
    Rc = torch.zeros(shape)

    ri, rj = 0, 0
    c = 1

    for i in range(S1):
        for j in range(S2):

            R[:, ri : (ri + m), rj : (rj + m)] += P[i, j, :, :, :].cpu()
            Rc[:, ri : (ri + m), rj : (rj + m)] += 1
            rj += stride
            c += 1
        ri += stride
        rj = 0

    return (R / Rc)[:, : shapeorg[-1], : shapeorg[-1]]


def main_eva(
    seed,
    model_name,
    trainset,
    testset,
    imgw=None,
    verbose=0,
    image_path_train=None,
    image_path_test=None,
    image_path = None,
    noise_type="gauss",
    opt=None,
    args=None,
    logger=None,
):
    cuda = True if torch.cuda.is_available() else False
    torch.autograd.set_detect_anomaly(True)
    opt.logger.info("CUDA: {0}".format(cuda))
    if cuda:
        dtype = torch.cuda.FloatTensor
        opt.logger.info(torch.cuda.get_device_name(0))
    else:
        dtype = torch.FloatTensor
    gtv = GTV(width=36, cuda=cuda, opt=opt)  # just initialize to load the trained model, no need to change
    PATH = model_name
    device = torch.device("cuda") if cuda else torch.device("cpu")
    gtv.load_state_dict(torch.load(PATH, map_location=device))
    gtv.cuda()
    width = gtv.opt.width
    opt.width = width
    opt = gtv.opt
    # if not image_path_train:
    #     image_path_train = "..\\all\\all\\"
    # if noise_type == "gauss":
    #     npref = "_g"
    # else:
    #     npref = "_n"
    if image_path:
        logger.info("EVALUATING IMAGE")
        traineva = {
            "psnr": float(),
            "ssim": float(),
            "ssim2": float(),
            "psnr2": float(),
            "mse": float(),
        }
        testeva = {
            "psnr": float(),
            "ssim": float(),
            "ssim2": float(),
            "psnr2": float(),
            "mse": float(),
        }
        inp = "{0}/noisy.png".format(image_path)
        argref = "{0}/ref.png".format(image_path)
        _, _ssim, _, _psnr2, _mse, _ = denoise(
                inp,
                gtv,
                argref,
                stride=args.stride,
                width=imgw,
                prefix=seed,
                opt=opt,
                args=args,
                logger=logger,
        )
        # traineva["psnr"].append(_psnr)
        traineva["ssim"] = _ssim
        # traineva["ssim2"].append(_ssim2)
        traineva["psnr2"] = _psnr2
        traineva["mse"] = _mse
        try:
            from skimage.metrics import structural_similarity as compare_ssim
        except Exception:
            from skimage.measure import compare_ssim

        img1 = cv2.imread(inp)[:, :, : opt.channels]
        img2 = cv2.imread(argref)[:, :, : opt.channels]
        #(score, diff) = compare_ssim(img1, img2, full=True, channel_axis=True)
        (score, diff) = compare_ssim(img1[:,:,0], img2[:,:,0], full=True)
        logger.info("Original {0:.2f} {1:.2f}".format(cv2.PSNR(img1, img2), score))
        img_noise_p =  "result/gauss_noisy.png"
        img_ref_p =   "result/ref_ref.png"
        img3 = np.array(cv2.imread(img_noise_p)[:, :, : opt.channels])
        img4 = np.array(cv2.imread(img_ref_p)[:, :, : opt.channels])
        (score1, diff) = compare_ssim(img3[:,:,0], img2[:,:,0], full=True)
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
        import matplotlib
        matplotlib.use('Agg')
        intensity_original = img4[args.row, :]
        intensity_noisy = img1[args.row, :]
        intensity_denoised = img3[args.row, :]
        # Tạo trục X (chỉ số cột)
        x_direction = range(intensity_original.shape[0])
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        axes[0,0].imshow(img4,cmap='gray')
        axes[0,0].set_title('grouth truth')
        axes[0,0].axis('off')
        
        axes[0,1].imshow(img1,cmap='gray')
        axes[0,1].set_title('noisy\nPSNR:{:.2f}\nSSIM:{:.2f}'.format(cv2.PSNR(img1, img2), score))
        axes[0,1].axis('off')
        
        axes[0,2].imshow(img3,cmap='gray')
        axes[0,2].set_title('denoise\nPSNR:{:.2f}\nSSIM:{:.2f}'.format(cv2.PSNR(img3, img2), score1))
        axes[0,2].axis('off')
        
        axes[1,0].plot(x_direction, intensity_original, color='blue', label='Original Image', linestyle='--', linewidth=2)
        axes[1,0].set_xlabel('X Direction')
        axes[1,0].set_ylabel('Intensity')
        axes[1,0].legend()
        axes[1,0].grid(True)

        axes[1,1].plot(x_direction, intensity_original, color='blue', label='Original Image', linestyle='--', linewidth=1)
        axes[1,1].plot(x_direction, intensity_noisy, color='black', label='Noisy Image', linestyle='--', linewidth=2)
        axes[1,1].set_xlabel('X Direction')
        axes[1,1].set_ylabel('Intensity')
        axes[1,1].legend()
        axes[1,1].grid(True)

        axes[1,2].plot(x_direction, intensity_original, color='blue', label='Original Image', linestyle='--', linewidth=1)
        axes[1,2].plot(x_direction, intensity_denoised, color='black', label='Denoise Image', linestyle='--', linewidth=2)
        axes[1,2].set_xlabel('X Direction')
        axes[1,2].set_ylabel('Intensity')
        axes[1,2].legend()
        axes[1,2].grid(True)
        plt.tight_layout()
        plt.savefig('output_image.png')
        plt.show()
    if image_path_train:
        logger.info("EVALUATING TRAIN SET")
        # trainset = ["10", "1", "7", "8", "9"]
        traineva = {
            "psnr": list(),
            "ssim": list(),
            "ssim2": list(),
            "psnr2": list(),
            "mse": list(),
        }
        testeva = {
            "psnr": list(),
            "ssim": list(),
            "ssim2": list(),
            "psnr2": list(),
            "mse": list(),
        }
        for t in trainset:
            logger.info("image #{0}".format(t))
            inp = "{0}/noisy/{1}.png".format(image_path_train, t)
            logger.info(inp)
            argref = "{0}/ref/{1}.png".format(image_path_train, t)
            _, _ssim, _, _psnr2, _mse, _ = denoise(
                inp,
                gtv,
                argref,
                stride=args.stride,
                width=imgw,
                prefix=seed,
                opt=opt,
                args=args,
                logger=logger,
            )
            # traineva["psnr"].append(_psnr)
            traineva["ssim"].append(_ssim)
            # traineva["ssim2"].append(_ssim2)
            traineva["psnr2"].append(_psnr2)
            traineva["mse"].append(_mse)
            try:
                from skimage.metrics import structural_similarity as compare_ssim
            except Exception:
                from skimage.measure import compare_ssim

            img1 = cv2.imread(inp)[:, :, : opt.channels]
            img2 = cv2.imread(argref)[:, :, : opt.channels]
            #(score, diff) = compare_ssim(img1, img2, full=True, channel_axis=True)
            (score, diff) = compare_ssim(img1[:,:,0], img2[:,:,0], full=True)
            logger.info("Original {0:.2f} {1:.2f}".format(cv2.PSNR(img1, img2), score))
        logger.info("========================")
        # logger.info("MEAN PSNR: {:.2f}".format(np.mean(traineva["psnr"])))
        logger.info("MEAN SSIM: {:.3f}".format(np.mean(traineva["ssim"])))
        # logger.info("MEAN SSIM2 (patch-based SSIM): {:.2f}".format(np.mean(traineva["ssim2"])))
        logger.info(
            "MEAN PSNR2 (image-based PSNR): {:.2f}".format(np.mean(traineva["psnr2"]))
        )
        logger.info("MEAN MSE (image-based MSE): {:.2f}".format(np.mean(traineva["mse"])))
        logger.info("========================")

    if image_path_test:
        logger.info("EVALUATING TEST SET")
        traineva = {
            "psnr": list(),
            "ssim": list(),
            "ssim2": list(),
            "psnr2": list(),
            "mse": list(),
        }
        testeva = {
            "psnr": list(),
            "ssim": list(),
            "ssim2": list(),
            "psnr2": list(),
            "mse": list(),
        }
        # testset = ["2", "3", "4", "5", "6"]
        for t in testset:
            logger.info("image #{0}".format(t))
            inp = "{0}/noisy/{1}.png".format(image_path_test, t)
            logger.info(inp)
            argref = "{0}/ref/{1}.png".format(image_path_test, t)
            _, _ssim, _, _psnr2, _mse, _ = denoise(
                inp,
                gtv,
                argref,
                stride=args.stride,
                width=imgw,
                prefix=seed,
                opt=opt,
                args=args,
                logger=logger,
            )
            # testeva["psnr"].append(_psnr)
            testeva["ssim"].append(_ssim)
            # testeva["ssim2"].append(_ssim2)
            testeva["psnr2"].append(_psnr2)
            testeva["mse"].append(_mse)
            try:
                from skimage.metrics import structural_similarity as compare_ssim
            except Exception:
                from skimage.measure import compare_ssim

            img1 = cv2.imread(inp)[:, :, : opt.channels]
            img2 = cv2.imread(argref)[:, :, : opt.channels]
            #(score, diff) = compare_ssim(img1, img2, full=True, channel_axis=True)
            (score, diff) = compare_ssim(img1[:,:,0], img2[:,:,0], full=True)
            logger.info("Original {0:.2f} {1:.2f}".format(cv2.PSNR(img1, img2), score))
        logger.info("========================")
        # logger.info("MEAN PSNR: {:.2f}".format(np.mean(testeva["psnr"])))
        logger.info("MEAN SSIM: {:.3f}".format(np.mean(testeva["ssim"])))
        # logger.info("MEAN SSIM2 (patch-based SSIM): {:.2f}".format(np.mean(testeva["ssim2"])))
        logger.info(
            "MEAN PSNR2 (image-based PSNR): {:.2f}".format(np.mean(testeva["psnr2"]))
        )
        logger.info("MEAN MSE (image-based MSE): {:.2f}".format(np.mean(testeva["mse"])))
        logger.info("========================")
    return traineva, testeva

opt = OPT(
    batch_size=32,
    channels=1,
    lr=1e-4,
    momentum=0.9,
    u_max=1000,
    u_min=0.0001,
    cuda=True if torch.cuda.is_available() else False
    )
if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(
        filename="log/test_gtv_{0}.log".format(time.strftime("%Y-%m-%d-%H%M")),
        filemode="a",
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.NOTSET,
    )

    logger = logging.getLogger("root")
    logger.addHandler(logging.StreamHandler(sys.stdout))

    opt.logger = logger
    opt.legacy = True
    supporting_matrix(opt)
    logger.info("GTV evaluation")
    logger.info(" ".join(sys.argv))
    _, _ = main_eva(
        seed="gauss",
        model_name=args.model,
        trainset=["1", "2", "3", "4"],
        testset=["1", "2", "3", "4"],
        imgw=args.width,
        verbose=1,
        image_path_train=args.image_path_train,
        image_path_test=args.image_path_test,
        image_path = args.image_path,
        noise_type="gauss",
        opt=opt,
        args=args,
        logger=logger,
    )

#python test_gtv.py -w 512 -m model/GTV_19.pkl --stride 9 --multi 200 --image_path_test dataset/Test --image_path_train dataset/dataset_structure