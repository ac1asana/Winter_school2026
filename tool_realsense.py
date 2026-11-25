"""
Tools for Watchful Eyes
Date: 25-11-2026
"""
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import pyrealsense2 as rs

WINDOW_NAME = "RealSense Video Capture"
RED = (0, 0, 255)

class Camera:
    """
    Class to set up and record video from a RealSense D405 camera.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        output_dir: str = "recordings",
        save_depth: bool = True,
    ):
        """
        Args:
            width: Frame width in pixels, used for both colour and depth.
            height: Frame height in pixels, used for both colour and depth.
            fps: Frame rate of the camera and of the saved video.
            output_dir: Folder where the mp4 files are saved.
            save_depth: If True, save colour and depth side by side,
                otherwise save only the colour stream.
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.output_dir = Path(output_dir)
        self.save_depth = save_depth

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.colorizer = rs.colorizer()

        self.writer = None
        self.frames_recorded = 0

    def connect(self):
        """
        Function to check the camera is connected and start streaming.
        """
        if len(rs.context().query_devices()) == 0:
            raise RuntimeError("No RealSense device found, check the USB connection")

        self.config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )
        self.config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )

        # Make sure the camera supports the chosen resolution and fps
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        if not self.config.can_resolve(pipeline_wrapper):
            raise RuntimeError(
                f"Camera does not support {self.width}x{self.height} at {self.fps} fps"
            )

        profile = self.pipeline.start(self.config)
        device_name = profile.get_device().get_info(rs.camera_info.name)
        print(f"{device_name} connected successfully")

    def start_recording(self):
        """
        Function to create a new mp4 file and start writing frames to it.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"recording_{timestamp}.mp4"

        # Side-by-side video is twice as wide as a single stream
        frame_width = self.width * 2 if self.save_depth else self.width
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(output_file), fourcc, self.fps, (frame_width, self.height)
        )

        if not self.writer.isOpened():
            self.writer = None
            raise RuntimeError(f"Could not create video file {output_file}")

        self.frames_recorded = 0
        print(f"Recording started: {output_file}")

    def stop_recording(self):
        """
        Function to stop recording and save the mp4 file.
        """
        if self.writer is None:
            return

        self.writer.release()
        self.writer = None
        print(f"Recording saved ({self.frames_recorded / self.fps:.1f} seconds)")

    def record(self, duration: float | None = None, start_immediately: bool = False):
        """
        Function to show a live preview and record video.

        Press 'r' to start/stop recording and 'q' to quit.

        Args:
            duration: Length of each recording in seconds. If None, recording
                continues until 'r' or 'q' is pressed.
            start_immediately: Start recording as soon as the camera is ready.
                If a duration is also given, the program exits once the
                recording is finished.
        """
        self.connect()
        max_frames = round(duration * self.fps) if duration else None

        if start_immediately:
            self.start_recording()

        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                # Convert frames to numpy arrays (depth is colorized for viewing)
                depth_color_frame = self.colorizer.colorize(depth_frame)
                depth_image = np.asanyarray(depth_color_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Stack both images horizontally for side-by-side preview
                images = np.hstack((color_image, depth_image))

                if self.writer is not None:
                    self.writer.write(images if self.save_depth else color_image)
                    self.frames_recorded += 1

                    # Stop automatically once the chosen duration is reached
                    if max_frames and self.frames_recorded >= max_frames:
                        self.stop_recording()
                        if start_immediately:
                            break

                # Show a red dot and timer on the preview while recording
                if self.writer is not None:
                    elapsed = self.frames_recorded / self.fps
                    cv2.circle(images, (20, 20), 8, RED, -1)
                    cv2.putText(
                        images,
                        f"REC {elapsed:.1f}s",
                        (35, 27),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        RED,
                        2,
                    )

                cv2.imshow(WINDOW_NAME, images)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("r"):
                    if self.writer is None:
                        self.start_recording()
                    else:
                        self.stop_recording()
                elif key == ord("q"):
                    break

        finally:
            # Save any recording in progress, stop the pipeline and close windows
            self.stop_recording()
            self.pipeline.stop()
            cv2.destroyAllWindows()
            # Let OpenCV process the close event so the window does not hang in Jupyter
            cv2.waitKey(1)


if __name__ == "__main__":
    camera = Camera(width=640, height=480, fps=30, output_dir="/home/")
    camera.record(duration=10, start_immediately=True)
