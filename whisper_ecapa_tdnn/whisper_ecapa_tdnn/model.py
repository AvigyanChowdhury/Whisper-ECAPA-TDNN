# setting the path 
path = 'aepyx.wav'

language = 'English' #param ['any', 'English']


model_size = 'base' #param ['tiny', 'base', 'small', 'medium', 'large']


model_name = model_size
if language == 'English' and model_size != 'large':
  model_name += '.en'

import whisper
import datetime

import subprocess

import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cpu"))

from pyannote.audio import Audio
from pyannote.core import Segment
import wave
import contextlib
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
import numpy as np
from time import time
import os
from argparse import ArgumentParser
from tabulate import tabulate
import matplotlib.pyplot as plt


# loading the whisper model
model = whisper.load_model(model_size)

start_time = time()
# transcription and segmentation of the audio using whisper model
result = model.transcribe(path)
end_time = time()

# the segments of the audio which conatin the id, start_time, end_time and the text is stored in the segments
segments = result["segments"]
print(segments)

# the time of inference for the segmentation and the ASR
print("Transcription/ASR time:", round(end_time-start_time,4),"secs")

# function to get the time for each segment
def get_duration(path:str):
  try:
    with contextlib.closing(wave.open(path,'r')) as f:
      frames = f.getnframes()
      rate = f.getframerate()
      duration = frames / float(rate)
  except:
    import librosa
    duration = librosa.get_duration(path=path)

  return duration


audio = Audio()
# the time for segmentation
start_time = time()

# function to find the embedding of each segment using ecapa-tdnn model
def segment_embedding(segment):
  start = segment["start"]

  # Whisper overshoots the end timestamp in the last segment
  duration = get_duration(path)
  end = min(duration, segment["end"])
  clip = Segment(start, end)
  waveform, sample_rate = audio.crop(path, clip)

  # Convert waveform to single channel
  waveform = waveform.mean(dim=0, keepdim=True)
  return embedding_model(waveform.unsqueeze(0))

end_time = time()
print("Segmentation of audio time:",round(end_time-start_time,4),"secs\n")

# start time is used to find the inference time for finding the embeddings
start_time = time()

# embeddings array of size (len(segments),192) dimension of 0 values is made
embeddings = np.zeros(shape=(len(segments), 192))

# Loop to find the embeddings of the segments
for i, segment in enumerate(segments):
  embeddings[i] = segment_embedding(segment)
print(embeddings)
# if there are any nan value in the array they are converted to 0
embeddings = np.nan_to_num(embeddings)
print("Length of embeddings generated:",len(embeddings),"\n\n")


# returns the time in seconds
def timee(secs):
  return datetime.timedelta(seconds=round(secs))


# KMeans clustering which is used with the elbow method to find the optimal number of speakers in the embedding
# elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(embeddings)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()


# Calculate the rate of change of the slope
slope_change = [inertia[i] - inertia[i - 1] for i in range(1, len(inertia))]

# Calculate the second derivative of the slope
second_derivative = [slope_change[i] - slope_change[i - 1] for i in range(1, len(slope_change))]

# Find the optimal number of clusters based on the second derivative
optimal_n_clusters = second_derivative.index(max(second_derivative)) + 2  # Adding 2 because index starts from 0
print("Number of speakers:", optimal_n_clusters,"\n")


# agglomerative clustering which has shown most accuracy among the other clustering methods
# the optimal number of speakers found using the elbow method is used in the agglomerative clustering
clustering = AgglomerativeClustering(n_clusters = optimal_n_clusters).fit(embeddings)
labels = clustering.labels_
for i in range(len(segments)):
  segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)


end_time = time()
print("Embedding and clustering time:",round(end_time-start_time,4),"secs\n")
print("Labels:",labels+1,"\n")

# # Storing the transcripts in a text file
# transcript = path[:-4]+".txt"
# f = open(transcript, "w",encoding="UTF-8")

# storing the speaker label, time-stamp and the transcription in 'txt' file
# for (i, segment) in enumerate(segments):
#   if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
#     f.write("\n" + segment["speaker"] + ' ' + str(segment["start"]) + '\n')
#   f.write(segment["text"][1:] + ' ')
# f.close()

# prints the result
# print(open(transcript, 'r', encoding="UTF-8").read())


# Dumping the segment start time and end time and the labels in rttm file
rttm_file = path[:-4]+'_hyp.rttm'
# writing the start_time, end_time and the labeling of the speakers
with open(rttm_file, mode='w', encoding='UTF-8') as f:
  for (i, segment) in enumerate(segments):
    if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
      start = segment["start"]
      duration = segment["end"]-segment["start"]
      label = segment["speaker"]
      line = f"SPEAKER {path[:-4]} 1 {start} {duration} <NA> <NA> {label} <NA> <NA>"
      f.write(line)
      if i!=len(segments)-1:
        f.write('\n')

print("The rttm file is generated\n")

# reading the rttm files
with open(rttm_file, 'r', encoding='UTF-8') as f:
  lines = f.read().split('\n')[: -1]
  lists = []
  for i in lines:
    spl = i.split(' ')
    lists.append(spl[8])
    print(i)
  labels = list(map(int, lists))

