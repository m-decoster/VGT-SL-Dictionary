"""A runtime for the SLR model underlying the dictionary."""
from typing import NamedTuple

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime

mp_holistic = mp.solutions.holistic


class Model:
    """A pre-trained SLR model that can be used to extract embeddings."""

    def __init__(self, model_path: str):
        """Initialize the model.

        :param model_path: Path to the ONNX model file."""
        self.ort_session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(model_path)

    def get_embedding(self, input_features: np.ndarray) -> np.ndarray:
        """Extract an embedding vector for given input features.

        :param input_features: The input features: an array of poses of shape (N, K, 3).
        :return: The output embedding."""
        input_features = np.expand_dims(input_features, 1)  # Add empty batch axis.
        input_features = input_features.astype(np.float32)  # Cast to 32-bit float (is: double).
        input_features = input_features[:, :, :75, :]  # Drop face keypoints.
        return self.ort_session.run(None, {'input': input_features})[0][0]  # Output is (1, 1, D).


def load_video(video_path: str) -> np.ndarray:
    """Load a video into a NumPy array of frames.

    :param video_path: The path to the video file.
    :return: An array of frames, of shape (N, H, W, 3).
    :raise FileNotFoundError: The video file could not be found."""
    cap = cv2.VideoCapture(video_path)

    if cap is None or not cap.isOpened():
        raise FileNotFoundError(f'Could not open video file {video_path}.')

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    return np.stack(frames)


def preprocess_video(frames: np.ndarray) -> np.ndarray:
    """Pre-process a video for use in the model.

    :param frames: The input RGB frames of shape (N, H, W, 3).
    :return: Features that can be used as input to the model."""
    return extract_keypoints(frames)


def extract_keypoints(frames: np.ndarray) -> np.ndarray:
    """Extract keypoints from a video.

    :param frames: The input RGB frames of shape (N, 3, H, W).
    :return: An array of shape (N, K, M) containing N poses of K M-dimensional keypoints."""
    out = []
    with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True) as holistic:
        for frame in frames:
            frame_landmarks: NamedTuple = holistic.process(frame)
            frame_output = []

            if frame_landmarks.pose_landmarks:
                landmarks = np.stack([np.array([l.x, l.y, l.z]) for l in frame_landmarks.pose_landmarks.landmark])
                frame_output.append(landmarks)
            else:
                frame_output.append(np.full((33, 3), np.nan))

            if frame_landmarks.left_hand_landmarks:
                landmarks = np.stack(
                    [np.array([l.x, l.y, l.z]) for l in frame_landmarks.left_hand_landmarks.landmark])
                frame_output.append(landmarks)
            else:
                frame_output.append(np.full((21, 3), np.nan))
            if frame_landmarks.right_hand_landmarks:
                landmarks = np.stack(
                    [np.array([l.x, l.y, l.z]) for l in frame_landmarks.right_hand_landmarks.landmark])
                frame_output.append(landmarks)
            else:
                frame_output.append(np.full((21, 3), np.nan))
            if frame_landmarks.face_landmarks:
                landmarks = np.stack(
                    [np.array([l.x, l.y, l.z]) for l in frame_landmarks.face_landmarks.landmark])
                frame_output.append(landmarks)
            else:
                frame_output.append(np.full((468, 3), np.nan))

            out.append(np.concatenate(frame_output, axis=0))

        return np.stack(out)
