## Project Kojak MVP

### Domain

For Project Kojak the domain I am tackling will be audio analysis. I would say I lack major experience with audio as I haven't attempted this before and have never done any sound process/mixing previously.I don't even play an instrument.

### Business Problem  

The business problem I am trying to tackle is to improve Youtube's demonetization algorithm for LGBT and sex education content creators by trying to properly classify their videos against explicit content. Currently there is a pain point where it seems that many sex education or LGBT education related content gets automatically demonetized as soon as videos pertaining to those subjects are uploaded. This is of particular issue for content creators as for any uploaded video the majority of views occur within the first 24 hours, and Youtube does not reimburse the content creators for any money they would have been lost during that period even if they are able to successfully appeal to get their video remonetized. For most content creators ads are still a major source of income, especially if they are small scale.

Working with the conceit that Youtube is not specifically targeting these types of videos of demonetized, I feel that their current demonetization algorithm could be improved as some content creators speculate it is keyword based. In terms of the source material I have domain knowledge there are number of relevant content creators that either I or my girlfriend subscribe to.

### Data

In terms of raw data, there is no dataset of youtube and explicit videos that I found on the publicly available for download. Instead, I plan on generate my own dataset of audio files by doing screen audio recordings where I play videos and capture the audio using quicktime utilizing the Sunflower plugin. I then utilize Audacity to split the audio files on the detected silences to generate individual audio files for each video that was played. If I plan to have 100 videos of each type, with a maximum of 10 min for each video, then it would take me a total of 33 hours to capture all the raw audio which I believe is doable.

For analyzing the audio data, there are two approaches that I plan on taking. One is to analyze the waveforms of the audio and the other is to analyze the transcripts. To generate the transcripts from each of the videos I plan on feeding the audio data to a speech to text generator to create a rough transcript for each video.

In terms of features I am still unsure about the features I will be generating from the waveforms to distinguish them from each other. For the audio waveforms I will be initially using the features that can be extracted from the librosa [packages] For the transcripts, I believe the diversity of words and the most common words will be useful features to capture.

[packages]: https://librosa.github.io/librosa/feature.html

One specific source of data that I will not be using is the title and description of videos as I believe the way that the explicit videos would be titled and described on Youtube would be different than how they are presented on the explicit websites.

### Known Unknowns
* I haven't decided if analyzing the entire waveform for each video, random sampling multiple 30 second increments from each video, or treating each random 30 second increment as an independent data point will be the most useful
* Time series techniques that would be applicable
* Which features that I generate from the audio waveforms would be useful
* Accuracy of the speech to text transcripts, how long they will take to generate
