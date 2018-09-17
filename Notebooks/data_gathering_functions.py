import os
import pandas as pd
import youtube_dl
import webvtt
from pydub import AudioSegment
# Imports the Google Cloud client library
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types


def rip_video(url, output_folder, write_transcript = True, n_start = 1, n_end = 10, ):

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
        'outtmpl': output_folder,
        'progress_hooks': [my_hook],
        'writeautomaticsub': write_transcript,
        'playliststart': n_start,
        'playlistend':n_end
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

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
