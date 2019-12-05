from skimage.measure import compare_ssim as ssim
import cv2


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def compare_image(imageA, imageB):
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    fig = plt.figure("test")
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")
    # show second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")

    plt.show()


def image_similarity(imageA, imageB):
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    rslt = [m, s]
    return rslt


def load_images(paths, color="gray"):
    image_files = []

    for f in paths[213:214]:
        image_files.append(sorted(glob.glob(f + "/*")))

    images = []
    if color == "gray":
        for ff in image_files:
            curr = []
            for f in ff:
                im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                curr.append(im)
            images.append(curr)
    else:
        for ff in image_files:
            curr = []
            for f in ff:
                im = cv2.imread(f, cv2.IMREAD_COLOR)
                curr.append(im)
            images.append(curr)
    return images


image_folders = sorted(glob.glob("../01_data/timelapse_images_fast/*"))
images_gray = load_images(image_folders)
images_col = load_images(image_folders, color="col")

summary = []
for i in np.arange(316):
    a = image_similarity(images_gray[0][i], images_gray[0][i + 1])
    summary.append()

summary = np.array(summary)
summary[:, 2] = np.arange(316)

plt.plot(summary[:, 0])
plt.scatter(x=np.arange(316), y=summary[:, 1])
plt.show()
