import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import structural_similarity as ssim_alt

# Load face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# Open video file
cap = cv2.VideoCapture("smol_video_test.mp4")

# Define output video codec and fps
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

fps = cap.get(cv2.CAP_PROP_FPS)

# Get video dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# skewness, kurtosis, std_dev, variance
# Create video writer object
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))
m_gaussian = list()
p_gaussian = list()
s_gaussian = list()
v_gaussian = list()
sd_gaussian = list()
sk_gaussian = list()
k_gaussian = list()
var_gaussian = list()


m_median = list()
p_median = list()
s_median = list()
v_median = list()
sd_median = list()
sk_median = list()
k_median = list()
var_median = list()

m_mosaic = list()
p_mosaic = list()
s_mosaic = list()
v_mosaic = list()
sd_mosiac = list()
sk_mosiac = list()
k_mosiac = list()
var_mosiac = list()

m_invz = list()
p_invz = list()
s_invz = list()
v_invz = list()
sd_inviz = list()
sk_inviz = list()
k_inviz = list()
var_inviz = list()
count = 0


def calc(face_roi):
    # face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    face_roi_gray = face_roi
    gx = cv2.Sobel(face_roi_gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(face_roi_gray, cv2.CV_32F, 0, 1)
    gradient = cv2.magnitude(gx, gy)
    skewness = np.mean((gradient - np.mean(gradient)) ** 3) / np.power(
        np.var(gradient), 1.5
    )
    kurtosis = (
        np.mean((gradient - np.mean(gradient)) ** 4) / np.power(np.var(gradient), 2) - 3
    )
    std_dev = np.std(gradient)
    variance = np.var(gradient)
    return (skewness, kurtosis, std_dev, variance)


# Loop through frames of the video
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break
    if count == 0:
        count = 1
        height, width, layer = frame.shape
        count = 1
        original_inv = frame.copy()
        nonBlured_original = original_inv.copy()
        grid = [[1 for i in range(6)] for i in range(6)]
        a = 15
        kernel = np.ones((a, a), dtype=np.float32) / (a**2)

        gray = cv2.cvtColor(original_inv, cv2.COLOR_BGR2GRAY)
        convolved = cv2.filter2D(gray, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        convolved = cv2.cvtColor(convolved, cv2.COLOR_GRAY2BGR)
        # Crop the result to the same size as the input frame
        height, width = original_inv.shape[:2]
        original_inv = convolved[:height, :width]
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Blur the faces using different techniques
    for x, y, w, h in faces:
        face = frame[y : y + h, x : x + w]
        face_roi = frame[y : y + h, x : x + w]

        # Calculate the skewness, kurtosis, standard deviation, and variance of the face region
        skewness, kurtosis, std_dev, variance = calc(face_roi)
        # Gaussian blur
        gaussian_blur = cv2.GaussianBlur(face, (15, 15), 0)
        frame[y : y + h, x : x + w] = gaussian_blur

        # Median blur
        median_blur = cv2.medianBlur(face, 15)
        frame[y : y + h, x : x + w] = median_blur

        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        mosaic_blur = cv2.resize(face_gray, (10, 10), interpolation=cv2.INTER_LINEAR)
        mosaic_blur = cv2.resize(mosaic_blur, (w, h), interpolation=cv2.INTER_NEAREST)
        mosaic_blur = cv2.cvtColor(mosaic_blur, cv2.COLOR_GRAY2BGR)
        frame[y : y + h, x : x + w] = mosaic_blur

        # Calculate PSNR, MSE, SSIM, VQM, SD for each blurred image
        original = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gaussian = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2GRAY)
        median = cv2.cvtColor(median_blur, cv2.COLOR_BGR2GRAY)
        mosaic = cv2.cvtColor(mosaic_blur, cv2.COLOR_BGR2GRAY)

        psnr_gaussian = psnr(original, gaussian)
        mse_gaussian = mse(original, gaussian)
        ssim_gaussian = ssim(
            original, gaussian, data_range=gaussian.max() - gaussian.min()
        )
        vqm_gaussian = ssim_alt(
            original, gaussian, data_range=gaussian.max() - gaussian.min()
        )
        # skewness = np.mean((gradient - np.mean(gradient))**3) / np.power(np.var(gradient), 1.5)
        # kurtosis = np.mean((gradient - np.mean(gradient))**4) / np.power(np.var(gradient), 2) - 3

        psnr_median = psnr(original, median)
        mse_median = mse(original, median)
        ssim_median = ssim(original, median, data_range=median.max() - median.min())
        vqm_median = ssim_alt(original, median, data_range=median.max() - median.min())

        mse_mosaic = mse(original, mosaic)
        if mse_mosaic == 0:
            mse_mosaic = 1e-10

        eps = 1e-10  # add a small constant value
        psnr_mosaic = 20 * np.log10(255.0 / np.sqrt(mse_mosaic + eps))

        ssim_mosaic = ssim(original, mosaic, data_range=mosaic.max() - mosaic.min())
        vqm_mosaic = ssim_alt(original, mosaic, data_range=mosaic.max() - mosaic.min())
        s = 15
        kernel = np.ones((s, s), dtype=np.float32) / s**2

        subframe = frame[y : y + h, x : x + w]
        gray = cv2.cvtColor(subframe, cv2.COLOR_BGR2GRAY)
        convolved = cv2.filter2D(gray, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        convolved = cv2.cvtColor(convolved, cv2.COLOR_GRAY2BGR)
        original_sub_section = original_inv[y : y + h, x : x + w]
        nonBlured_original_subSection = nonBlured_original[y : y + h, x : x + w]
        diff = np.abs(convolved - original_sub_section)
        mask = diff > 20
        subframe[mask] = nonBlured_original_subSection[mask]

        inviz = cv2.cvtColor(subframe, cv2.COLOR_BGR2GRAY)
        # print(inviz.shape)
        psnr_inviz = psnr(original, inviz)
        mse_inviz = mse(original, inviz)
        ssim_inviz = ssim(original, inviz, data_range=inviz.max() - inviz.min())
        vqm_inviz = ssim_alt(original, inviz, data_range=inviz.max() - inviz.min())

        p_invz.append(psnr_inviz)
        m_invz.append(mse_inviz)
        s_invz.append(ssim_inviz)
        v_invz.append(vqm_inviz)
        a, b, c, d = calc(inviz)
        sd_inviz.append(a)
        sk_inviz.append(b)
        k_inviz.append(c)
        var_inviz.append(d)

        m_gaussian.append(mse_gaussian)
        p_gaussian.append(psnr_gaussian)
        s_gaussian.append(ssim_gaussian)
        v_gaussian.append(vqm_gaussian)
        a, b, c, d = calc(gaussian)
        sd_gaussian.append(a)
        sk_gaussian.append(b)
        k_gaussian.append(c)
        var_gaussian.append(d)

        m_median.append(mse_median)
        p_median.append(psnr_median)
        s_median.append(ssim_median)
        v_median.append(vqm_median)
        a, b, c, d = calc(median)
        sd_median.append(a)
        sk_median.append(b)
        k_median.append(c)
        var_median.append(d)

        m_mosaic.append(mse_mosaic)
        p_mosaic.append(psnr_mosaic)
        s_mosaic.append(ssim_mosaic)
        v_mosaic.append(vqm_mosaic)
        a, b, c, d = calc(mosaic)
        sd_mosiac.append(a)
        sk_mosiac.append(b)
        k_mosiac.append(c)
        var_mosiac.append(d)


# print('MSE Gaussian Blur=',sum(m_gaussian)/len(m_gaussian), 'MSE Median Blur=',sum(m_median)/len(m_gaussian), 'MSE Mosaic Blur=',sum(m_mosaic)/len(m_mosaic))
print(
    "PSNR Gaussian Blur=",
    sum(p_gaussian) / len(p_gaussian),
    " PSNR Median Blur=",
    sum(p_median) / len(p_gaussian),
    "PSNR Mosaic Blur=",
    sum(p_mosaic) / len(p_mosaic),
    "PSNR Invisiblur Blur=",
    sum(p_invz) / len(p_invz),
)
print(
    "SSIM Gaussian Blur=",
    sum(s_gaussian) / len(s_gaussian),
    "SSIM Median Blur=",
    sum(s_median) / len(s_gaussian),
    "SSIM Mosaic Blur=",
    sum(s_mosaic) / len(s_mosaic),
    "SSIM Invisiblur Blur=",
    sum(s_invz) / len(s_invz),
)
print(
    "VQM Gaussian Blur=",
    sum(v_gaussian) / len(v_gaussian),
    "VQM Median Blur=",
    sum(v_median) / len(v_gaussian),
    "VQM Mosaic Blur=",
    sum(v_mosaic) / len(v_mosaic),
    "VQM Invisiblur Blur=",
    sum(v_invz) / len(v_invz),
)
print(
    "Std D Gaussian Blur=",
    sum(sd_gaussian) / len(sd_gaussian),
    "Std D Median Blur=",
    sum(sd_median) / len(sd_gaussian),
    "Std D Mosaic Blur=",
    sum(sd_mosiac) / len(sd_mosiac),
    "Std D Invisiblur Blur=",
    sum(sd_inviz) / len(sd_inviz),
)
print(
    "Skew Gaussian Blur=",
    sum(sk_gaussian) / len(sk_gaussian),
    "Skew Median Blur=",
    sum(sk_median) / len(sk_gaussian),
    "Skew Mosaic Blur=",
    sum(sk_mosiac) / len(sk_mosiac),
    "Skew Invisiblur Blur=",
    sum(sk_inviz) / len(sk_inviz),
)
print(
    "Kurt Gaussian Blur=",
    sum(k_gaussian) / len(k_gaussian),
    "Kurt Median Blur=",
    sum(k_median) / len(k_gaussian),
    "Kurt Mosaic Blur=",
    sum(k_mosiac) / len(k_mosiac),
    "Kurt Invisiblur Blur=",
    sum(k_inviz) / len(k_inviz),
)
print(
    "Var Gaussian Blur=",
    sum(var_gaussian) / len(var_gaussian),
    "Var Median Blur=",
    sum(var_median) / len(var_gaussian),
    "Var Mosaic Blur=",
    sum(var_mosiac) / len(var_mosiac),
    "Var Invisiblur Blur=",
    sum(var_inviz) / len(var_inviz),
)


# inside detected part take variance and std deviation
# curtosis
# quinus
# heat map of image
# histogram plot


# values = [sum(m_gaussian)/len(m_gaussian), sum(m_median)/len(m_gaussian) ,sum(m_mosaic)/len(m_mosaic) ]
# labels = ['Gaussian Blur', 'Median Blur', 'Mosaic Blur']


values1 = [
    sum(p_gaussian) / len(p_gaussian),
    sum(p_median) / len(p_gaussian),
    sum(p_mosaic) / len(p_mosaic),
    sum(p_invz) / len(p_invz),
]
labels1 = ["Gaussian Blur", "Median Blur", "Mosaic Blur", "InvisiBlur"]


values2 = [
    sum(s_gaussian) / len(s_gaussian),
    sum(s_median) / len(s_gaussian),
    sum(s_mosaic) / len(s_mosaic),
    sum(s_invz) / len(s_invz),
]
labels2 = ["Gaussian Blur", "Median Blur", "Mosaic Blur", "InvisiBlur"]


values3 = [
    sum(v_gaussian) / len(v_gaussian),
    sum(v_median) / len(v_gaussian),
    sum(v_mosaic) / len(v_mosaic),
    sum(v_invz) / len(v_invz),
]
labels3 = ["Gaussian Blur", "Median Blur", "Mosaic Blur", "InvisiBlur"]

values4 = [
    sum(sd_gaussian) / len(sd_gaussian),
    sum(sd_median) / len(sd_gaussian),
    sum(sd_mosiac) / len(sd_mosiac),
    sum(sd_inviz) / len(sd_inviz),
]
labels4 = ["Gaussian Blur", "Median Blur", "Mosaic Blur", "InvisiBlur"]

values5 = [
    sum(sk_gaussian) / len(sk_gaussian),
    sum(sk_median) / len(sk_gaussian),
    sum(sk_mosiac) / len(sk_mosiac),
    sum(sk_inviz) / len(sk_inviz),
]
labels5 = ["Gaussian Blur", "Median Blur", "Mosaic Blur", "InvisiBlur"]

values6 = [
    sum(k_gaussian) / len(k_gaussian),
    sum(k_median) / len(k_gaussian),
    sum(k_mosiac) / len(k_mosiac),
    sum(k_inviz) / len(k_inviz),
]
labels6 = ["Gaussian Blur", "Median Blur", "Mosaic Blur", "InvisiBlur"]

values7 = [
    sum(var_gaussian) / len(var_gaussian),
    sum(var_median) / len(var_gaussian),
    sum(var_mosiac) / len(var_mosiac),
    sum(var_inviz) / len(var_inviz),
]
labels7 = ["Gaussian Blur", "Median Blur", "Mosaic Blur", "InvisiBlur"]


fig, axs = plt.subplots(4, 2, figsize=(10, 8))

# Plot the first bar graph in the top-left subplot
# axs[0, 0].bar(labels, values)
# axs[0, 0].set_title('MSE Values')

# Plot the second bar graph in the top-right subplot
axs[0, 0].bar(labels1, values1)
axs[0, 0].set_title("PSNR Values")

# Plot the third bar graph in the bottom-left subplot
axs[0, 1].bar(labels2, values2)
axs[0, 1].set_title("SSIM Values")

# Plot the fourth bar graph in the bottom-right subplot
axs[1, 0].bar(labels3, values3)
axs[1, 0].set_title("VQM Values")

axs[1, 1].bar(labels4, values4)
axs[1, 1].set_title("STD Values")

axs[2, 0].bar(labels3, values5)
axs[2, 0].set_title("skew Values")

axs[2, 1].bar(labels3, values6)
axs[2, 1].set_title("kurt Values")

axs[3, 0].bar(labels3, values7)
axs[3, 0].set_title("var Values")


# Adjust the layout of the subplots
plt.tight_layout()
plt.show()
