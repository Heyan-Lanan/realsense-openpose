import pyrealsense2 as rs

import numpy as np
import cv2

from openpose_light import OpenposeLight


# kpt_names = ['nose', 'neck',
#              'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
#              'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
#              'r_eye', 'l_eye',
#              'r_ear', 'l_ear']

class RealsensePose:
    def __init__(self, checkpoints_path, w=640, h=480):
        self.openpose = OpenposeLight(checkpoints_path)

        self.pipeline = rs.pipeline()

        self.pc = rs.pointcloud()

        self.align = rs.align(rs.stream.color)

        self.init_realsense(w, h)

    def init_realsense(self, w, h):
        config = rs.config()
        config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        profile = self.pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: ", depth_scale)

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()

        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        return color_frame, depth_frame

    def get_vertices_1(self, color_frame, depth_frame):
        points = self.pc.calculate(depth_frame)
        self.pc.map_to(color_frame)
        vertices = points.get_vertices()
        vertices = np.asanyarray(vertices).view(np.float32).reshape(-1, 3)  # xyz
        return vertices

    def keypoint_at(self, poses, index):
        point = [0, 0]
        if poses and poses[0].keypoints[index].min() >= 0:
            point = poses[0].keypoints[index].tolist()
        return tuple(point)

    def run(self):

        mean_time = 0

        while True:
            current_time = cv2.getTickCount()

            color_frame, depth_frame = self.get_frames()
            vertices = self.get_vertices_1(color_frame, depth_frame)

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            poses = self.openpose.predict(color_image)
            rendered_image = self.openpose.draw_poses(color_image, poses)

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            draw_index = [2, 3, 4]
            for i in draw_index:
                point = self.keypoint_at(poses, i)
                cv2.circle(depth_colormap, point, 3, (0, 255, 0))
                x, y, z = vertices[640 * point[1] + point[0]]
                text = "x={:+.2f}, y={:+.2f}, z={:+.2f}".format(x, y, z)
                cv2.putText(depth_colormap, text, point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))

            current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
            if mean_time == 0:
                mean_time = current_time
            else:
                mean_time = mean_time * 0.95 + current_time * 0.05
            text = 'FPS: {}'.format(int(1 / mean_time * 10) / 10)
            cv2.putText(depth_colormap, text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

            cv2.imshow('Depth', depth_colormap)
            cv2.imshow('Color', rendered_image)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    from config import OPENPOSE_PATH

    rs_pose = RealsensePose(OPENPOSE_PATH)
    rs_pose.run()
