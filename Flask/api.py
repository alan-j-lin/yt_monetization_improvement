import os
import librosa
import pickle
import youtube_dl
import pandas as pd
import numpy as np
import webvtt
import nlp_processing
from glob import glob
from pydub import AudioSegment
from librosa.feature import rmse, spectral_bandwidth, zero_crossing_rate
# Imports the Google Cloud client library
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

"""
Functions in this folder have the same name but are different than the ones
in demonetization.py and data_gathering_functions.py
"""

class Segment():

    def __init__(self, name, y, sr, per_order, text):
        self.name = name
        self.audio_timeseries = y
        self.sr = sr
        self.per_order = per_order
        self.text = text
        self.rmse_ = rmse(y)[0]
        self.spectral_bandwidth_ = spectral_bandwidth(y, sr = sr)[0]
        self.zero_crossing_rate_ = zero_crossing_rate(y)[0]
        self.label = None

##############################################################################

def get_classification(data):

    """
    function that classifies whether or not a video is explicit or education
    returns dictionary with label

    Parameters
    ----------
    data is a dictionary:
    'url': url of video in question

    Returns
    -------
    result is a dictionary:
    'label': returns whether video is explicit or education
    """

    # loading pickle files from the model folder
    with open('./model/vectorizer.pkl', 'rb') as filename:
        vec = pickle.load(filename)
        filename.close()
    with open('./model/reducer.pkl', 'rb') as filename:
        red = pickle.load(filename)
        filename.close()
    with open('./model/standard_scaler.pkl', 'rb') as filename:
        ssX = pickle.load(filename)
    with open('./model/model.pkl', 'rb') as filename:
        model = pickle.load(filename)
        filename.close()

    # Instantiates a client
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '_alanpersonal-984c98451486.json'
    client = speech.SpeechClient()
    segment_folder = '.'

    url = data['url']

    path, vtt_path = rip_video(url)

    vid_list = generate_segment_list(client, path, vtt_path, 30, 10, segment_folder)
    X_test = create_df(vid_list)
    X_test_scaled = transform_testdata(X_test, vec, red, ssX)
    y_predict = model.predict(X_test_scaled)
    vid_probability = len([i for i in y_predict if i == 1]) / len(y_predict)
    print(vid_probability)

    if vtt_path != 'None':
        os.remove(vtt_path)
    os.remove(path)
    for seg in glob('*.wav'):
        os.remove(seg)

    if vid_probability >= 0.5:
        response = {
            'label': 'explicit',
            'probability': vid_probability
        }
        
    else:
        response = {
            'label': 'education',
            'probability': vid_probability
        }
    return response

##############################################################################

def transform_testdata(X_test, vectorizer, reducer, standard_scaler):
    test_text_data = X_test['text'].values
    test_numeric_data = pd.concat((X_test.iloc[:, 1], X_test.iloc[:, 3:-1]), axis = 1)
    vect_data = vectorizer.transform(test_text_data)
    red_data = reducer.transform(vect_data)
    combined_data = np.hstack((red_data, test_numeric_data))
    X_test_scaled = standard_scaler.transform(combined_data)
    return X_test_scaled

##############################################################################

def rip_video(url):

    """

    Function that uses youtube_dl to the rip the audio off the video from a given url.
    Outputs the mp3 file to the output_folder and can also capture the autogenerate captions.
    If the url is a playlist n_start and n_end determines how many videos are ripped.

    """

    class MyLogger(object):
        def debug(self, msg):
            pass

        def warning(self, msg):
            pass

        def error(self, msg):
            print(msg)

    def my_hook(d):
        if d['status'] == 'finished':
            print('Done downloading, now converting ...')

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'logger': MyLogger(),
        'progress_hooks': [my_hook],
        'writeautomaticsub': True,
        'noplaylist': True
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    try:
        file_paths = (glob('*.mp3')[0], glob('*.vtt')[0])
    except IndexError:
        file_paths = (glob('*.mp3')[0], 'None')

    return file_paths

##############################################################################

def generate_segment_list(client, path, vtt_path, time_per_segment, cutoff, segment_folder):

    segment_list = []

    if os.path.isfile(vtt_path):
        video_captions = generate_captions_vtt(vtt_path, time_per_segment, cutoff)
        video_infos, _ = segment_audio(client, path, time_per_segment, cutoff, segment_folder, True)
    else:
        video_infos, video_captions = segment_audio(client, path, time_per_segment, cutoff, segment_folder, False)
    for vid_info, vid_caption in zip(video_infos, video_captions):
        segment_list.append(create_segment(f'{segment_folder}/{vid_info[0]}', vid_info, vid_caption))

    return segment_list


def generate_captions_vtt(caption_path, time_per_segment, cutoff):
    """

    Function that returns a lists of captions from a given caption '.vtt' file
    Each entry in the list is the captions that were stated in the specified time_per_segment

    """

    vtt = webvtt.read(caption_path)

    caption_dict = {
        'time': [cap.start for cap in vtt.captions],
        'caption': [cap.text.split('\n') for cap in vtt.captions]
    }

    cap_df = pd.DataFrame(caption_dict)
    cap_df['time'] = cap_df['time'].apply(lambda x: pd.to_datetime(x))
    cap_df['agg_time'] = cap_df['time'].apply(lambda x: (x - cap_df['time'].iloc[0]).seconds - cutoff)

    caption_segments = []
    time_steps = [t for t in range(0, cap_df['agg_time'].iloc[-1] - cutoff, time_per_segment)]
    for idx in range(len(time_steps) - 1):
        acc = []
        for line in cap_df[(time_steps[idx] <= cap_df['agg_time'])
                           & (cap_df['agg_time'] < time_steps[idx + 1]) ]['caption'].values:
            acc += line

        seen = set()
        ordered_uniques = [line for line in acc if not (line in seen or seen.add(line))]
        caption_segments.append(' '.join(ordered_uniques))

    return caption_segments

def generate_captions_audio(audio_path, client):

    """

    Function that loads the given audio file into memory and uses the Google speech to
    text API to generate captions. Returns a list of the joined transcripts. Files have
    to be less than 1 min and in .wav format to use the client properly.

    """

    # Loads the audio into memory
    with open(audio_path, 'rb') as audio_file:
        content = audio_file.read()
        audio = types.RecognitionAudio(content=content)

    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code='en-US')

    # Detects speech in the audio file
    response = client.recognize(config, audio)
    transcripts = [response.alternatives[0].transcript for response in response.results]

    return ' '.join(transcripts)

def segment_audio(client, path, time_per_seg, cutoff, output_folder = None, caption_exists = True):

    """

    function that will segment the given path into time_segment (seconds) chunks, subtracting the cutoff
    (in seconds) from the beginning and end

    """
    # Checks to see if an output folder is given as that determines the path that should be given
    if output_folder is None:
        output = f'{os.getcwd()}/'
    else:
        output = f'{os.getcwd()}/{output_folder}/'
    # Load audio file that is likely generate from the rip_video function
    audio_file = AudioSegment.from_mp3(path)
    # determines the length of the audio segments based on the time_per_seg and cutoff input parameters
    file_duration = len(audio_file) / 1000
    remainder = (file_duration - (2*cutoff)) % time_per_seg
    start_time = cutoff * 1000
    end_time = int(file_duration - (remainder + cutoff)) * 1000
    time_range = range(start_time, end_time, time_per_seg * 1000)
    n_times = len(time_range)
    # file_tuples and caption_segments are what ends up being outputted by the function
    file_tuples = []
    caption_segments = []

    for idx in range(n_times - 1):
        cut_audio = audio_file[time_range[idx]:time_range[idx+1]]
        mono_audio = cut_audio.set_channels(1)
        filetype = 'wav'
        single_path = path.split('.')[0].split('/')[-1]
        i_filename = f"{single_path}-{idx}.{filetype}"
        i_path = f"{output}{i_filename}"
        mono_audio.export(i_path, format = filetype);
        if caption_exists == False:
            segment_captions = generate_captions_audio(i_path, client)
            caption_segments.append(segment_captions)
        file_tuples.append((i_filename, idx/n_times))

    print(f'Finished with: {path}')

    return file_tuples, caption_segments

def create_segment(path, vid_info, captions):
    """
    function that will create a segment from the given file and given captions

    """
    filename = vid_info[0]
    per_order = vid_info[1]
    y, sr = librosa.core.load(path)
    return Segment(filename, y, sr, per_order, captions)

##############################################################################

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
