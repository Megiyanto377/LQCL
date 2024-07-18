import cv2
import pickle
import numpy as np
from transform import *
from utils import *
from scipy.interpolate import CubicSpline, BarycentricInterpolator

def read_txt_lines(filepath):
    assert os.path.isfile( filepath ), "Error when trying to read txt file, path does not exist: {}".format(filepath)
    with open( filepath ) as myfile:
        content = myfile.read().splitlines()
    return content


class AVSRDataLoader(object):
    """AVSRDataLoader.
    The class extracts row audio waveforms or mouth RoIs from a video and landmarks.

    """

    def __init__(
        self,
        mean_face_path="20words_mean_face.npy",
        crop_width=96,
        crop_height=96,
        start_idx=48,
        stop_idx=68,
        window_margin=12,
        convert_gray=True):
        """__init__.

        :param mean_face_path: str, the directory for the reference face.
        :param crop_width: int, the width of the cropped patch.
        :param crop_height: int, the height of the cropped patch.
        :param start_idx: int, the starting index for cropping the bounding box.
        :param stop_idx: int, the ending inex for cropping the bounding box.
        :param window_margin: int, the window size for smoothing landmarks.
        :param convert_gray: bool, save as grayscale if set as True.
        """

        self._reference = np.load(os.path.join( os.path.dirname(__file__), mean_face_path))
        self._crop_width = crop_width
        self._crop_height = crop_height
        self._start_idx = start_idx
        self._stop_idx = stop_idx
        self._window_margin = window_margin
        self._convert_gray = convert_gray


    def preprocess(self, video_pathname, landmarks_pathname):
        """preprocess.

        :param video_pathname: str, the filename for the video.
        :param landmarks_pathname: str, the filename for the landmarks.
        """
        # -- Step 1, extract landmarks from pkl files.
        if isinstance(landmarks_pathname, str):
            with open(landmarks_pathname, "rb") as pkl_file:
                landmarks = pickle.load(pkl_file)
        else:
            landmarks = landmarks_pathname
        # -- Step 2, pre-process landmarks: interpolate frames that not being detected.
        preprocessed_landmarks = self.landmarks_interpolate(landmarks)
        # -- Step 3, exclude corner case: 1) no landmark in all frames; 2) number of frames is less than window length.
        # Check if preprocessed_landmarks is empty or has fewer frames than self._window_margin
        if preprocessed_landmarks.size == 0 or preprocessed_landmarks.shape[0] < self._window_margin:
            return
        # -- Step 4, affine transformation and crop patch 
        sequence, transformed_frame, transformed_landmarks = self.crop_patch(video_pathname, preprocessed_landmarks)
        assert sequence is not None, "cannot crop from {}.".format(filename)
        return sequence

    
    def interpolate_landmark_dimension(self,frames):
        """
        Interpolates missing values for a single dimension (x or y) of a landmark across frames.
    
        Args:
        - frames (np.array): 1D numpy array of positions for a single dimension (x or y) of a landmark.
    
        Returns:
        - np.array: Interpolated positions for the dimension.
        """
        x = np.arange(len(frames))
        valid_mask = ~np.isnan(frames)
    
        if not np.any(valid_mask):  # All values are missing
            return frames  # Return as is, or handle differently as needed
    
        spline_interpolator = CubicSpline(x[valid_mask], frames[valid_mask])
        return spline_interpolator(x)
    
    def landmarks_interpolate(self,landmarks):
        """
        Applies interpolation across all landmarks and dimensions (x, y) for a sequence of frames.
    
        Args:
        - landmarks (list or np.array): 3D structure of shape [frames, landmarks, coordinates].
    
        Returns:
        - np.array: 3D numpy array with missing values interpolated.
        """
        # Convert landmarks to a NumPy array if it's not already one
        landmarks_np = np.array(landmarks)
    
        num_frames, num_landmarks, num_coords = landmarks_np.shape
    
        # Initialize an array to hold interpolated landmarks
        interpolated_landmarks = np.zeros_like(landmarks_np)
    
        for landmark_idx in range(num_landmarks):
            for coord_idx in range(num_coords):
                frames = landmarks_np[:, landmark_idx, coord_idx]
                interpolated_landmarks[:, landmark_idx, coord_idx] = self.interpolate_landmark_dimension(frames)
    
        return interpolated_landmarks


    def crop_patch(self, video_pathname, landmarks):
        """crop_patch.

        :param video_pathname: str, the filename for the processed video.
        :param landmarks: List, the interpolated landmarks.
        """

        frame_idx = 0
        frame_gen = load_video(video_pathname)
        while True:
            try:
                frame = frame_gen.__next__() ## -- BGR
            except StopIteration:
                break
            if frame_idx == 0:
                sequence = []

                sequence_frame = []
                sequence_landmarks = []

            window_margin = min( self._window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx)
            smoothed_landmarks = np.mean([landmarks[x] for x in range(frame_idx - window_margin, frame_idx + window_margin + 1)], axis=0)
            smoothed_landmarks += landmarks[frame_idx].mean(axis=0) - smoothed_landmarks.mean(axis=0)
            transformed_frame, transformed_landmarks = self.affine_transform(frame, smoothed_landmarks, self._reference, grayscale=self._convert_gray)
            sequence.append( cut_patch( transformed_frame,
                                        transformed_landmarks[self._start_idx:self._stop_idx],
                                        self._crop_height//2,
                                        self._crop_width//2,))

            sequence_frame.append( transformed_frame)
            sequence_landmarks.append( transformed_landmarks)

            frame_idx += 1
        return np.array(sequence), np.array(sequence_frame), np.array(sequence_landmarks)


    def affine_transform(self, frame, landmarks, reference, grayscale=False, target_size=(256, 256),
                         reference_size=(256, 256), stable_points=(28, 33, 36, 39, 42, 45, 48, 54),
                         interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT,
                         border_value=0):
        """affine_transform.

        :param frame:
        :param landmarks:
        :param reference: ndarray, the neutral reference frame.
        :param grayscale: bool, save as grayscale if set as True.
        :param target_size: tuple, size of the output image.
        :param reference_size: tuple, size of the neural reference frame.
        :param stable_points: tuple, landmark idx for the stable points.
        :param interpolation: interpolation method to be used.
        :param border_mode: Pixel extrapolation method .
        :param border_value: Value used in case of a constant border. By default, it is 0.
        """
        # Prepare everything
        if grayscale and frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stable_reference = np.vstack([reference[x] for x in stable_points])
        stable_reference[:, 0] -= (reference_size[0] - target_size[0]) / 2.0
        stable_reference[:, 1] -= (reference_size[1] - target_size[1]) / 2.0

        # Warp the face patch and the landmarks
        transform = cv2.estimateAffinePartial2D(np.vstack([landmarks[x] for x in stable_points]),
                                                stable_reference, method=cv2.LMEDS)[0]
        transformed_frame = cv2.warpAffine(frame, transform, dsize=(target_size[0], target_size[1]),
                                    flags=interpolation, borderMode=border_mode, borderValue=border_value)
        transformed_landmarks = np.matmul(landmarks, transform[:, :2].transpose()) + transform[:, 2].transpose()

        return transformed_frame, transformed_landmarks


    def load_audio(self, data_filename):
        """load_audio.

        :param data_filename:
        """
        sequence = load_audio(data_filename, specified_sr=16000, int_16=False)
        return sequence

    def _get_boundary(self, filename):
        info_txt = filename.replace(".mp4",".txt")
        info = read_txt_lines(info_txt)

        utterance_duration = float( info[4].split(' ')[1] )
        half_interval = int( utterance_duration/2.0 * 25)
        n_frames = 25


        mid_idx = ( n_frames -1 ) // 2  # video has n frames, mid point is (n-1)//2 as count starts with 0
        left_idx = max(0, mid_idx-half_interval-1)
        right_idx = min(mid_idx+half_interval+1, n_frames)

        boundary = np.zeros(n_frames)
        boundary[left_idx:right_idx] = 1
        return boundary
    
    def load_video(self, data_filename, landmarks_filename=None):
        """load_video.

        :param data_filename: str, the filename for input sequence.
        :param landmarks_filename: str, the filename for the landmarks of the target speaker.
        """
        assert landmarks_filename is not None
        sequence = self.preprocess(
            video_pathname=data_filename,
            landmarks_pathname=landmarks_filename,
        )
        label = data_filename.split("/")[-1].split("_")[0]
        
        boundary = self._get_boundary(data_filename)
        # print(boundary)
        return sequence, label, boundary

    def load_data(self, modality, data_filename, landmarks_filename=None):
        """load_data.

        :param modality: str, the modality type of the input.
        :param data_filename: str, the filename for input sequence.
        :param landmarks_filename: str, the filename for the landmarks of the target speaker.
        """
        if modality == "raw_audio":
            return self.load_audio(data_filename)
        elif modality == "video":
            return self.load_video(data_filename, landmarks_filename=landmarks_filename)
