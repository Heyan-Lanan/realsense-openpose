import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from PIL import Image
from ultralytics.utils.plotting import Annotator, colors, save_one_box

kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
# [1 nose, 2 left eye, 3 right eye, 4 left ear, 5 right ear, 6 left shoulder, 7 right shoulder, 8 left elbow,
# 9 right elbow, 10 left wrist, 11 right wrist, 12 left hip, 13 right hip, 14 left knee, 15 right knee,
# 16 left ankle, 17 right ankle]
skeleton = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
]
v_list = [[0.0, 0.0]] * 20
gamma = 0.9
pc = rs.pointcloud()
align = rs.align(rs.stream.color)


# print(len(im))
# print(len(im[0]))

def kpts(kpts, shape=None, radius=5, kpt_line=True, im=None):
    """
    Plot keypoints on the image.

    Args:
        kpts (tensor): Predicted keypoints with shape [17, 3]. Each keypoint has (x, y, confidence).
        shape (tuple): Image shape as a tuple (h, w), where h is the height and w is the width.
        radius (int, optional): Radius of the drawn keypoints. Default is 5.
        kpt_line (bool, optional): If True, the function will draw lines connecting keypoints
                                   for human pose. Default is True.

    Note:
        `kpt_line=True` currently only supports human pose plotting.
    """

    nkpt, ndim = kpts.shape
    is_pose = nkpt == 17 and ndim in {2, 3}
    kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
    for i, k in enumerate(kpts):
        color_k = [int(x) for x in kpt_color[i]] if is_pose else colors(i)
        x_coord, y_coord = k[0], k[1]
        if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
            if len(k) == 3:
                conf = k[2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

    if kpt_line:
        ndim = kpts.shape[-1]
        for i, sk in enumerate(skeleton):
            pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
            pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
            if ndim == 3:
                conf1 = kpts[(sk[0] - 1), 2]
                conf2 = kpts[(sk[1] - 1), 2]
                if conf1 < 0.5 or conf2 < 0.5:
                    continue
            if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                continue
            if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                continue
            cv2.line(im, pos1, pos2, [int(x) for x in limb_color[i]], thickness=2, lineType=cv2.LINE_AA)
            # print(list(map(int, his_wrist[0:2])))
            # start = list(map(int, his_wrist[0:2]))
            # end = list(map(int, now_wrist[0:2]))

        im = np.asanyarray(im)
        # print(im.shape)
        # im = im[0:480, 0:640, :]
        # print(im)
        return im


def get_vertices_1(color_frame, depth_frame):
    points = pc.calculate(depth_frame)
    pc.map_to(color_frame)
    vertices = points.get_vertices()
    vertices = np.asanyarray(vertices).view(np.float32).reshape(-1, 3)  # xyz
    return vertices


if __name__ == "__main__":
    # Configure depth and color streams
    # Load a model
    model = YOLO('yolov8n-pose.pt')  # load an official model
    # model = YOLO('path/to/best.pt')  # load a custom model

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)
    wrist_1 = [0.0, 0.0]
    wrist_2 = []
    mean_time = 0

    try:
        while True:

            current_time = cv2.getTickCount()

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays

            depth_image = np.asanyarray(depth_frame.get_data())
            depth_data = np.asanyarray(depth_frame.get_data(), dtype="float16")
            color_image = np.asanyarray(color_frame.get_data())

            results = model(color_image)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            # images = np.hstack((color_image, depth_colormap))
            # Show images
            # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            # print(color_image)
            result = results[0]
            keypoints = result.keypoints
            keypoints = np.array(keypoints.data.cpu())[0]
            point = keypoints[10]
            vertices = get_vertices_1(color_frame, depth_frame)
            # vertices = np.array(vertices)
            # print(vertices.shape)
            # print(point)
            # print(640 * point[1] + point[0])
            x, y, z = vertices[min(int(640 * point[1] + point[0]), 307200 - 1)]

            text = "x={:+.2f}, y={:+.2f}, z={:+.2f}".format(x, y, z)

            im = kpts(kpts=keypoints, im=color_image, shape=color_image.shape)
            # cv2.imshow('color_image', color_image)
            current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
            if mean_time == 0:
                mean_time = current_time
            else:
                mean_time = mean_time * 0.95 + current_time * 0.05

            point = [int(i) for i in point]
            cv2.putText(im, text, point[0:2], cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
            cv2.putText(im, 'FPS: {}'.format(int(1 / mean_time * 10) / 10),
                        (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
            cv2.imshow('frame', im)
            cv2.imshow('depth_image', depth_colormap)
            wrist_1 = wrist_2

            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pipeline.stop()

# Predict with the model
# results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
# results = model('img.png')

# Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     # print(keypoints)
#     probs = result.probs  # Probs object for classification outputs
#     print(np.array(keypoints.data.cpu()).shape)
#     im = cv2.imread('img.png')
#     kpts(kpts=np.array(keypoints.data.cpu())[0], im=im, shape=(len(im), len(im[0])))

# result.show()  # display to screen
# result.save(filename='result_1.jpg')  # save to disk
