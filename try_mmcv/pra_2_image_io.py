import mmcv
import numpy as np

flag_1 = False  # 读取图像，保存图像
flag_2 = False  # 以byte形式读取图像
flag_3 = False  # 显示图像
flag_4 = False  # 图像颜色空间转换
flag_5 = False  # Resize
flag_6 = False  # Rotate
flag_7 = False  # Flip
flag_8 = False  # Crop
flag_9 = False  # Padding

if flag_1:
    # To read or write images files, use imread or imwrite.
    img = mmcv.imread("asset/a.jpg")
    img = mmcv.imread("asset/a.jpg", flag='grayscale')
    img_ = mmcv.imread(img)  # nothing will happen, img_ = img
    mmcv.imwrite(img, 'out.jpg')

if flag_2:
    # To read images from bytes
    with open("asset/a.jpg", 'rb') as f:
        data = f.read()
    img = mmcv.imfrombytes(data)
    print(img)

if flag_3:
    # To show an image file or a loaded image
    mmcv.imshow("asset/a.jpg")
    # this is equivalent to

    for i in range(10):
        img = np.random.randint(256, size=(100, 100, 3), dtype=np.uint8)
        mmcv.imshow(img, win_name='test image', wait_time=200)

if flag_4:
    # Color space conversion
    """
    Supported conversion methods:
        bgr2gray
        gray2bgr
        bgr2rgb
        rgb2bgr
        bgr2hsv
        hsv2bgr
    """
    img = mmcv.imread("asset/a.jpg")
    img1 = mmcv.bgr2rgb(img)
    img2 = mmcv.rgb2gray(img1)
    img3 = mmcv.bgr2hsv(img)
    mmcv.imshow(img1)
    mmcv.imshow(img2)
    mmcv.imshow(img3)

if flag_5:
    # Resize
    """
    There are three resize methods. All imresize_* methods have an argument return_scale, if this argument is False, then the return value is merely the resized image, otherwise is a tuple (resized_img, scale).
    """
    img = mmcv.imread("asset/a.jpg")
    dst_img = mmcv.imread("asset/b.jpg")

    # resize to a given size
    resized_img = mmcv.imresize(img, (1000, 600), return_scale=True)
    mmcv.imshow(resized_img[0])
    print(resized_img[1])  # `w_scale`
    print(resized_img[2])  # `h_scale`

    # resize to the same size of another image
    resized_img_1 = mmcv.imresize_like(img, dst_img, return_scale=False)
    mmcv.imshow(resized_img_1)
    # print(resized_img_1[1])  # `w_scale`
    # print(resized_img_1[2])  # `h_scale`

    # resize by a ratio
    resized_img_2 = mmcv.imrescale(img, 0.5)
    mmcv.imshow(resized_img_2)

    # resize so that the max edge no longer than 1000, short edge no longer than 800
    # without changing the aspect ratio
    resized_img_3 = mmcv.imrescale(img, (1000, 800))
    mmcv.imshow(resized_img_3)

    # mmcv.imshow(img_1)
    # mmcv.imshow(img_2)
    # mmcv.imshow(img_3)
    # mmcv.imshow(img_4)

if flag_6:
    # Rotate
    """
    To rotate an image by some angle, use imrotate. 
    The center can be specified, which is the center of original image by default. 
    There are two modes of rotating, one is to keep the image size unchanged so that some parts of the image will be cropped after rotating, the other is to extend the image size to fit the rotated image.
    """
    img = mmcv.imread('asset/a.jpg')

    # rotate the image clockwise by 30 degrees.
    img_ = mmcv.imrotate(img, 30)
    mmcv.imshow(img_)

    # rotate the image counterclockwise by 90 degrees.
    img_ = mmcv.imrotate(img, -90)
    mmcv.imshow(img_)

    # rotate the image clockwise by 30 degrees, and rescale it by 1.5x at the same time.
    img_ = mmcv.imrotate(img, 30, scale=1.5)
    mmcv.imshow(img_)

    # rotate the image clockwise by 30 degrees, with (100, 100) as the center.
    img_ = mmcv.imrotate(img, 30, center=(100, 100))
    mmcv.imshow(img_)

    # rotate the image clockwise by 30 degrees, and extend the image size.
    img_ = mmcv.imrotate(img, 30, auto_bound=True)
    mmcv.imshow(img_)

if flag_7:
    # Flip
    """
    To flip an image, use imflip.
    """
    img = mmcv.imread("asset/a.jpg")

    # flip the image horizontally
    img_ = mmcv.imflip(img)
    mmcv.imshow(img_)

    # flip the image vertically
    img_ = mmcv.imflip(img, direction='vertical')
    mmcv.imshow(img_)

if flag_8:
    # Crop
    """
    imcrop can crop the image with one or some regions, represented as (x1, y1, x2, y2).
    """
    img = mmcv.imread("asset/a.jpg")

    # crop the region (10, 10, 100, 120)
    bboxes = np.array([10, 10, 100, 120])
    patch = mmcv.imcrop(img, bboxes)
    mmcv.imshow(patch)
    # crop two regions (10, 10, 100, 120) and (0, 0, 50, 50)
    bboxes = np.array([[10, 10, 100, 120], [0, 0, 50, 50]])
    patches = mmcv.imcrop(img, bboxes)
    mmcv.imshow(patches[0])
    mmcv.imshow(patches[1])

    # crop two regions, and rescale the patches by 1.2x
    patches = mmcv.imcrop(img, bboxes, scale=1.2)
    mmcv.imshow(patches[0])
    mmcv.imshow(patches[1])

if flag_9:
    # Padding
    """
    There are two methods impad and impad_to_multiple to pad an image to the specific size with given values.
    """
    img = mmcv.imread("asset/a.jpg")

    # pad the image to (1000, 1200) with all zeros
    img_ = mmcv.impad(img, shape=(1000, 1200), pad_val=0)
    mmcv.imshow(img_)

    # pad the image to (1000, 1200) with different values for three channels.
    img_ = mmcv.impad(img, shape=(1000, 1200), pad_val=(100, 50, 200))
    mmcv.imshow(img_)

    # pad the image on left, right, top, bottom borders with all zeros
    img_ = mmcv.impad(img, padding=(10, 20, 30, 40), pad_val=0)
    mmcv.imshow(img_)

    # pad the image on left, right, top, bottom borders with different values
    # for three channels.
    img_ = mmcv.impad(img, padding=(10, 20, 30, 40), pad_val=(100, 50, 200))
    mmcv.imshow(img_)

    # pad an image so that each edge is a multiple of some value.padding到宽高能被某个数整除
    img_ = mmcv.impad_to_multiple(img, 29)
    mmcv.imshow(img_)
