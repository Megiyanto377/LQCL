import os
import sys
import cv2
import librosa
import numpy as np


def load_video(filename):
    """load_video.

    :param filename: str, the fileanme for a video sequence.
    """
    cap = cv2.VideoCapture(filename)                                             
    while(cap.isOpened()):                                                       
        ret, frame = cap.read() # BGR                                            
        if ret:                                                                  
            yield frame                                                          
        else:                                                                    
            break                                                                
    cap.release()


def load_audio(audio_filename, specified_sr=16000, int_16=True):
    """load_audio.

    :param audio_filename: str, the filename for an audio waveform.
    :param specified_sr: int, expected sampling rate, the default value is 16KHz.
    :param int_16: boolean, return 16-bit PCM if set it as True.
    """
    try:
        if audio_filename.endswith('npy'):
            audio = np.load(audio_filename)
        elif audio_filename.endswith('npz'):
            audio = np.load(audio_filename)['data']
        else:
            audio, sr = librosa.load(audio_filename, sr=None)
            audio = librosa.resample(audio, sr, specified_sr) if sr != specified_sr else audio
    except IOError:
        sys.exit()
    if int_16 and audio.dtype==np.float32:
        audio = ((audio - 1.) * (65535./2.) + 32767.).astype(np.int16)
        audio = np.array(np.clip(np.round(audio), -2**15, 2**15-1), dtype=np.int16)
    if not int_16 and audio.dtype==np.int16:
        audio = ((audio - 32767.) * 2 / 65535. + 1).astype(np.float32)
    return audio


def save2npz(filename, data=None):
    """save2npz.

    :param filename: str, the fileanme where the data will be saved.
    :param data: ndarray, arrays to save to the file.
    """
    assert data is not None, "data is {}".format(data)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    np.savez_compressed(filename, data=data)
