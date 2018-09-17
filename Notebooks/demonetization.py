import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from librosa.feature import mfcc, melspectrogram, rmse, spectral_bandwidth, zero_crossing_rate
from data_gathering_functions import rip_video, generate_captions_audio, generate_captions_vtt, segment_audio


class Segment():

    def __init__(self, name, y, sr, per_order, text, label):
        self.name = name
        self.audio_timeseries = y
        self.sr = sr
        self.per_order = per_order
        self.text = text
        self.melspectrogram_ = melspectrogram(y, sr = sr)
        self.mfcc_ = mfcc(y, sr = sr)
        self.rmse_ = rmse(y)[0]
        self.spectral_bandwidth_ = spectral_bandwidth(y, sr = sr)[0]
        self.zero_crossing_rate_ = zero_crossing_rate(y)[0]
        self.label = label


def create_segment(path, vid_info, captions, label):
    """
    function that will create a segment from the given file and given captions

    """
    filename = vid_info[0]
    per_order = vid_info[1]
    y, sr = librosa.core.load(path)
    return Segment(filename, y, sr, per_order, captions, label)


def generate_segment_list(label, audio_file_type, vid_output, client, time_per_segment, cutoff, segment_folder):

    segment_list = []

    for path in glob(f'{vid_output}/*.{audio_file_type}'):
        filetype_split = path.split(f'.{audio_file_type}')[0]
        vtt_path = f"{os.getcwd()}/{filetype_split}.en.vtt"
        if os.path.isfile(vtt_path):
            video_captions = generate_captions_vtt(vtt_path, time_per_segment, cutoff)
            video_infos, _ = segment_audio(client, path, time_per_segment, cutoff, segment_folder, True)
        else:
            video_infos, video_captions = segment_audio(client, path, time_per_segment, cutoff, segment_folder, False)
        for vid_info, vid_caption in zip(video_infos, video_captions):
            segment_list.append(create_segment(f'{segment_folder}/{vid_info[0]}', vid_info, vid_caption, label))

    return segment_list

def create_df(vid_list):

    columns_dict = {
        'name': [vid.name for vid in vid_list],
        'per_video': [vid.per_order for vid in vid_list],
        'text': [vid.text for vid in vid_list],
        'zero_median': [np.median(vid.zero_crossing_rate_) for vid in vid_list],
        'zero_var': [np.var(vid.zero_crossing_rate_) for vid in vid_list],
        'rmse_median': [np.median(vid.rmse_) for vid in vid_list],
        'rmse_var': [np.var(vid.rmse_) for vid in vid_list],
        'spec_median': [np.median(vid.spectral_bandwidth_) for vid in vid_list],
        'spec_var': [np.var(vid.spectral_bandwidth_) for vid in vid_list],
        'label': [vid.label for vid in vid_list]
    }


    return pd.DataFrame(columns_dict)


def plot_waveform(path, start_sec, end_sec):
    y, sr = librosa.core.load(path)
    S_full, phase = librosa.magphase(librosa.stft(y))
    idx = slice(*librosa.time_to_frames([start_sec, end_sec], sr=sr))
    librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                             y_axis='log', sr=sr)
    plt.title(path)
    plt.colorbar()






# def vocal_separation(path, start_sec, end_sec):
#     y, sr = librosa.core.load(path)
#     # And compute the spectrogram magnitude and phase
#     S_full, phase = librosa.magphase(librosa.stft(y))
#     idx = slice(*librosa.time_to_frames([start_sec, end_sec], sr=sr))
#     # We'll compare frames using cosine similarity, and aggregate similar frames
#     # by taking their (per-frequency) median value.
#     #
#     # To avoid being biased by local continuity, we constrain similar frames to be
#     # separated by at least 2 seconds.
#     #
#     # This suppresses sparse/non-repetetitive deviations from the average spectrum,
#     # and works well to discard vocal elements.
#
#     S_filter = librosa.decompose.nn_filter(S_full,
#                                            aggregate=np.median,
#                                            metric='cosine',
#                                            width=int(librosa.time_to_frames(2, sr=sr)))
#
#     # The output of the filter shouldn't be greater than the input
#     # if we assume signals are additive.  Taking the pointwise minimium
#     # with the input spectrum forces this.
#     S_filter = np.minimum(S_full, S_filter)
#     # We can also use a margin to reduce bleed between the vocals and instrumentation masks.
#     # Note: the margins need not be equal for foreground and background separation
#     margin_i, margin_v = 2, 10
#     power = 2
#
#     mask_i = librosa.util.softmask(S_filter,
#                                    margin_i * (S_full - S_filter),
#                                    power=power)
#
#     mask_v = librosa.util.softmask(S_full - S_filter,
#                                    margin_v * S_filter,
#                                    power=power)
#
#     # Once we have the masks, simply multiply them with the input spectrum
#     # to separate the components
#
#     S_foreground = mask_v * S_full
#     S_background = mask_i * S_full
#     # sphinx_gallery_thumbnail_number = 2
#
#     return (sr, idx, S_full, S_background, S_foreground)
#
# def plot_vocalsep(vocalsep_tuple, video_title, nrows, index):
#
#     sr, idx, S_full, S_background, S_foreground = vocalsep_tuple
#
#     plt.subplot(nrows, 1, index+1)
#     librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
#                              y_axis='log', sr=sr)
#     plt.title(video_title)
#     plt.colorbar()
#
#     plt.subplot(nrows, 1, index+2)
#     librosa.display.specshow(librosa.amplitude_to_db(S_background[:, idx], ref=np.max),
#                              y_axis='log', sr=sr)
#     plt.title('Background')
#     plt.colorbar()
#
#     plt.subplot(nrows, 1, index+3)
#     librosa.display.specshow(librosa.amplitude_to_db(S_foreground[:, idx], ref=np.max),
#                              y_axis='log', x_axis='time', sr=sr)
#     plt.title('Foreground')
#     plt.colorbar()
