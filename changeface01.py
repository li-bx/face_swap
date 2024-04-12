import os
import dlib
import cv2 
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

pwd_path = os.path.abspath(os.path.dirname(__file__))
source_img_path = os.path.join(pwd_path, '1.jpg')
target_img_path = os.path.join(pwd_path, '2.jpg')

plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

def detect_face_landmarks(image, predictor):
    # 使用dlib的face detector（如HOG或CNN）检测人脸位置
    face_detector = dlib.get_frontal_face_detector()
    faces = face_detector(image)

    if len(faces) < 1:
        raise ValueError("Exactly one face should be present in the image.")

    # 使用shape_predictor预测面部特征点
    face_shape = predictor(image, faces[0])
    return  face_shape
def show_img_landmarks(image, landmarks):
    # 显示图片和特征点
    for idx, point in enumerate(landmarks.parts()):
            # 68点的坐标
            pos = (point.x, point.y)
            print(idx+1,pos)

            # 利用cv2.circle给每个特征点画一个圈，共68个
            cv2.circle(image, pos, 5, color=(0, 255, 0))
            # 利用cv2.putText输出1-68
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, str(idx+1), pos, font, 0.5, (0, 0, 255), 1,cv2.LINE_AA)
    face_rect=landmarks.rect
    cv2.rectangle(image,(face_rect.left(),face_rect.top()) ,(face_rect.right(),face_rect.bottom()), (0, 255, 0), 2)
    return image
def get_face_mask(image_size, face_landmarks):
    """
    获取人脸掩模，包含轮廓
    :param image_size: 图片大小
    :param face_landmarks: 68个特征点
    :return: image_mask, 掩模图片
    """
    mask = np.zeros(image_size, dtype=np.uint8)
    points = np.concatenate([face_landmarks[0:17], face_landmarks[26:16:-1]])
    cv2.fillPoly(img=mask, pts=[points], color=1)
    return mask
def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)
def get_face_mask2(sz, landmarks):
    LEFT_EYE_POINTS = list(range(42, 48))
    RIGHT_EYE_POINTS = list(range(36, 42))
    LEFT_BROW_POINTS = list(range(22, 27))
    RIGHT_BROW_POINTS = list(range(17, 22))
    NOSE_POINTS = list(range(27, 35))
    MOUTH_POINTS = list(range(48, 61))
    OVERLAY_POINTS = [
        LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
        NOSE_POINTS + MOUTH_POINTS,
    ]
    FEATHER_AMOUNT = 11
    im = np.zeros(sz, dtype=np.float64)
	#双眼的外接多边形、鼻和嘴的多边形，作为掩膜
    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    # 三维掩码
    # im = np.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im.astype('uint8')
def transformation_from_points(target_points, source_points):
    # 计算两个点集之间的仿射变换矩阵
    # 输入：points1, points2
    target_points = target_points.copy()
    source_points = source_points.copy()
    target_points = target_points.astype(np.float64)
    source_points = source_points.astype(np.float64)

    c1 = np.mean(target_points, axis=0)
    c2 = np.mean(source_points, axis=0)
    target_points -= c1
    source_points -= c2

    s1 = np.std(target_points)
    s2 = np.std(source_points)
    target_points /= s1
    source_points /= s2
    
    U, S, Vt = np.linalg.svd(np.dot(target_points.T , source_points))
    R = (np.dot(U , Vt)).T 
    return np.vstack([np.hstack(((s2 / s1) * R,
                                       np.array([c2.T - np.dot((s2 / s1) * R , c1.T)]).T )),
                         np.array([0., 0., 1.])])
def wrap_im(im,M,dshape):
    '''
    对齐图像到目标图上，返回对齐后的targe
    '''
    output_im = np.zeros(dshape,dtype=im.dtype)
    cv2.warpAffine(im,M[:2],(dshape[1],dshape[0]),dst=output_im,borderMode=cv2.BORDER_TRANSPARENT,flags=cv2.WARP_INVERSE_MAP)
    return output_im
def correct_colors(im1, im2, landmarks1,COLOUR_CORRECT_BLUR_FRAC = 0.6):
    LEFT_EYE_POINTS = list(range(42, 48))
    RIGHT_EYE_POINTS = list(range(36, 42))
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                              np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0,0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0,0)
    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /im2_blur.astype(np.float64))

def change_face(color_fac):
    # 颜色矫正
    source_image_align_correct = correct_colors(target_image,source_image_align,tgt_landmarks_np,color_fac)
    # 换脸
    output_im = (np.multiply(target_image,(1-combined_mask)) 
                  + np.multiply(source_image_align_correct,combined_mask)).astype('uint8')
    cv2.imshow('output_im_correct'+str(color_fac), source_image_align_correct.astype('uint8'))
    cv2.imshow('output_im_change'+str(color_fac), output_im.astype('uint8')) 

if __name__ == '__main__':

    plt.ion()
    plt.figure('换脸')

    # 预训练的68点人脸标记点检测器
    predictor_path = os.path.join(pwd_path, "shape_predictor_68_face_landmarks.dat")
    predictor = dlib.shape_predictor(predictor_path)
    # 读取原始图片和要替换的脸部图片
    source_image = cv2.imread(source_img_path)
    target_image = cv2.imread(target_img_path)
    # 检测人脸特征点
    src_landmarks = detect_face_landmarks(source_image, predictor)
    tgt_landmarks = detect_face_landmarks(target_image, predictor)

    # 显示图片特征点
    src_copy = source_image.copy()
    src_copy = show_img_landmarks(src_copy, src_landmarks)
    tgt_copy = target_image.copy()
    tgt_copy = show_img_landmarks(tgt_copy, tgt_landmarks)
    # cv2.imshow('source_image', src_copy)
    # cv2.imshow('target_image', tgt_copy)

    plt.subplot(3,4,11)
    plt.title("脸来源图-关键点")
    # bgr转换rgb
    plt.imshow(src_copy[...,::-1])
    plt.subplot(3,4,12)
    plt.title("脸目标图-关键点")
    plt.imshow(tgt_copy[...,::-1])

    # 转换为numpy数组
    src_landmarks_np = np.array([[p.x, p.y] for p in src_landmarks.parts()])
    tgt_landmarks_np = np.array([[p.x, p.y] for p in tgt_landmarks.parts()])

    src_size = (source_image.shape[0], source_image.shape[1])
    src_mask = get_face_mask(src_size, src_landmarks_np)  # 脸图人脸掩模
    plt.subplot(3,4,1)
    plt.title("脸来源图")
    plt.imshow(source_image[...,::-1])
    plt.subplot(3,4,2)
    plt.title("脸来源图-掩码")
    plt.imshow(src_mask,"gray")

    tgt_size = (target_image.shape[0], target_image.shape[1])
    tgt_mask = get_face_mask(tgt_size, tgt_landmarks_np)  # 脸图人脸掩模
    plt.subplot(3,4,3)
    plt.title("脸目标图")
    plt.imshow(target_image[...,::-1])
    plt.subplot(3,4,4)
    plt.title("脸目标图-掩码")
    plt.imshow(tgt_mask,"gray")

    # 计算到目标图的仿射变换矩阵
    trans_mat = transformation_from_points(tgt_landmarks_np,src_landmarks_np)

    # 仿射变换，对齐来源图像到目标图上
    source_image_align = wrap_im(source_image,trans_mat,target_image.shape[:3])
    plt.subplot(3,4,5)
    plt.title("脸来源图-对齐")
    plt.imshow(source_image_align[...,::-1])

    # 仿射变换，对齐来源图像掩码到目标图上
    src_mask_align = wrap_im(src_mask,trans_mat,target_image.shape[:2])

    # 融合掩码
    combined_mask = np.max([tgt_mask, src_mask_align],
                            axis=0)    
    # combined_mask = np.min([tgt_mask, src_mask_align],
    #                         axis=0)
    # combined_mask = tgt_mask
    plt.subplot(3,4,6)
    plt.title("对齐融合掩码")
    combined_mask = np.stack((combined_mask,combined_mask,combined_mask),axis=2)
    plt.imshow(combined_mask[...,0],"gray")

    plt.subplot(3,4,7)
    # combined_mask = combined_mask/255
    plt.title("来源图掩码对应底图")
    plt.imshow(np.multiply(source_image_align,combined_mask).astype('uint8')[...,::-1])

    plt.subplot(3,4,8)
    # combined_mask = combined_mask/255
    plt.title("目标图掩码对应底图")
    plt.imshow(np.multiply(target_image,combined_mask).astype('uint8')[...,::-1])

    # 颜色矫正
    source_image_align_correct = correct_colors(target_image,source_image_align,tgt_landmarks_np,0.8)
    plt.subplot(3,4,9)
    plt.title("颜色矫正")
    plt.imshow(source_image_align_correct[...,::-1].astype('uint8'))

    # 换脸
    output_im = (np.multiply(target_image,(1-combined_mask)) 
                  + np.multiply(source_image_align_correct,combined_mask)*1
                  + np.multiply(target_image,combined_mask)*0.0 ).astype('uint8')
    plt.subplot(3,4,10)
    plt.title("换脸")
    plt.imshow(cv2.cvtColor(output_im.copy().astype('uint8'),cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # cv2.imshow('output_im', source_image_align_correct.astype('uint8'))
    # cv2.imshow('output_im_correct', output_im.astype('uint8'))

    # change_face(0.1)    
    # change_face(0.5)    
    # change_face(0.9)

    plt.ioff()
    plt.show()

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
