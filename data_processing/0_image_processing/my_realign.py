import face_alignment
from skimage import io
import cv2
import glob
import os
import numpy as np
import PIL
import scipy
import shutil
import time
import scipy.ndimage

def vis(lm, img, title):
    print('lm shape: ', lm.shape)
    vis_img = np.array(img)
    for i in range(lm.shape[0]):
        point = (int(lm[i, 0]), int(lm[i, 1]))
        # print(point,vis_img.shape)
        cv2.circle(vis_img, point, 5, (0, 255, 0), 5)

        cv2.putText(vis_img, str(i), point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    origin_w, origin_h, _ = vis_img.shape
    scale = 1.0
    if max(origin_w, origin_h) > 1024:
        scale = 1024 / max(origin_w, origin_h)
        vis_img = cv2.resize(vis_img, (int(origin_h * scale), int(origin_w * scale)))

    cv2.imshow(title, vis_img)

    cv2.waitKey(0)


def get_vis_img(lm, img):
    assert len(lm) == 5
    vis_img = np.array(img)
    key = ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']
    for i in range(len(lm)):
        point = (int(lm[i][0]), int(lm[i][1]))
        # print(point,vis_img.shape)
        cv2.circle(vis_img, point, 10, (0, 255, 0), 5)

        cv2.putText(vis_img, key[i], point, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 2)

    return vis_img


def create_perspective_transform_matrix(src, dst):
    A = (dst.dot(src.T)).dot(np.linalg.inv(src.dot(src.T)))
    return A


def realign_process_image(name, src_path, dst_path, output_size, transform_size, enable_padding,
                          lm,remove_lr_image_resolution = 512):  # item_idx, item, dst_dir="realign1500", output_size=1500, transform_size=4096, enable_padding=True):

    # if os.path.isfile(dst_path): return

    # Parse landmarks.
    # pylint: disable=unused-variable
    lm = np.array(lm)
    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise

    lm_new = lm.copy()

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    q_scale = 1.8
    x = q_scale * x
    y = q_scale * y
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Load in-the-wild image.

    if not os.path.isfile(src_path):
        print(f'\nCannot find source image {src_path}. Please run "--wilds" before "--align".')
        return
    img = PIL.Image.open(src_path)
    img = img.convert('RGB')

    # Shrink.
    start_time = time.time()
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        lm_new = lm_new / shrink
        # vis(lm_new,img,title='shrink')
        quad /= shrink
        qsize /= shrink
    # print("shrink--- %s seconds ---" % (time.time() - start_time))

    # Crop.
    start_time = time.time()
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))

    # (left, top, right, bottom)
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)

        lm_new[:, 0] -= crop[0]
        lm_new[:, 1] -= crop[1]
        # vis(lm_new, img,title='crop')
        quad -= crop[0:2]

    if img.size[0] < remove_lr_image_resolution and img.size[1] < remove_lr_image_resolution:
        print('low resolution image!', img.size)
        return None, None

    # print("crop--- %s seconds ---" % (time.time() - start_time))

    # Pad.
    start_time = time.time()
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))

    # vis(lm_new, img, title='before padding')
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.array(img)
        img = np.pad(img.astype(np.float32), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        lm_new[:, 0] += pad[0]
        lm_new[:, 1] += pad[1]
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        low_res = cv2.resize(img, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
        blur = qsize * 0.02 * 0.1
        low_res = scipy.ndimage.gaussian_filter(low_res, [blur, blur, 0])
        low_res = cv2.resize(low_res, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4)

        img += (low_res - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        median = cv2.resize(img, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
        median = np.median(median, axis=(0, 1))

        img += (median - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

        # vis(lm_new, img,title='padding')
    # print("pad--- %s seconds ---" % (time.time() - start_time))

    # Transform.
    start_time = time.time()
    # print('quad: ',quad,quad.shape) # 4,2

    lm_quad = quad + 0.5
    # lm_perspective_transform_matrix = create_perspective_transform_matrix(
    #     [(lm_quad[0][0], lm_quad[0][1]), (lm_quad[1][0], lm_quad[1][1]), (lm_quad[2][0], lm_quad[2][1]), (lm_quad[3][0], lm_quad[3][1])],
    #     [(0, 0), (transform_size, 0), (transform_size, transform_size), (0, transform_size)]
    # )
    src = lm_quad.T  # 2,4
    src = np.concatenate([src, np.ones((1, 4))], axis=0)  # 3,4
    dst = np.array([[0, 0], [0, transform_size], [transform_size, transform_size], [transform_size, 0]]).T
    dst = np.concatenate([dst, np.ones((1, 4))], axis=0)  # 3,4
    lm_perspective_transform_matrix = create_perspective_transform_matrix(src, dst)
    lm_new_pad = np.ones((lm_new.shape[0], 1))
    lm_new = np.concatenate([lm_new, lm_new_pad], axis=1).T  # 3,68
    # print('lm_perspective_transform_matrix shape:',lm_perspective_transform_matrix.shape)
    # print('lm_perspective_transform_matrix: ',lm_perspective_transform_matrix)
    # print('lm_new shape: ',lm_new.shape)

    lm_new = lm_perspective_transform_matrix.dot(lm_new)[:2, :].T

    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    # print(img.size,transform_size,quad)
    # vis(lm_new, img, title='quad transform')
    if output_size < transform_size:
        lm_new *= output_size / transform_size
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
        # vis(lm_new, img, title='final resize')
    # print("transform--- %s seconds ---" % (time.time() - start_time))

    # Save aligned image.
    #print('save image to ', dst_path)
    img.save(dst_path)
    return img, lm_new


def get_5p_landmarks(face_landmarks):
    # ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']
    crop_lm = []
    # 按照肖像的左右

    # def extract_5p(lm):
    #     lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    #     lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
    #         lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
    #     lm5p = lm5p[[1, 2, 0, 3, 4], :]
    #     return lm5p

    # 37 40
    crop_lm.append(
        [
            int((face_landmarks[36][0] + face_landmarks[39][0]) / 2),
            int((face_landmarks[36][1] + face_landmarks[39][1]) / 2),
        ]
    )
    # 43 46
    crop_lm.append(
        [
            int((face_landmarks[42][0] + face_landmarks[45][0]) / 2),
            int((face_landmarks[42][1] + face_landmarks[45][1]) / 2),
        ]
    )
    # 31
    crop_lm.append(
        [
            int(face_landmarks[30][0]), int(face_landmarks[30][1])
        ]
    )
    # 49
    crop_lm.append(
        [
            int(face_landmarks[48][0]), int(face_landmarks[48][1])
        ]
    )
    # 55
    crop_lm.append(
        [
            int(face_landmarks[54][0]), int(face_landmarks[54][1])
        ]
    )
    return crop_lm




