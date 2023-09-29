Integration of Whisper ASR model ,ECAPAT-TDNN model and  Evaluation for Speaker Diarization
Metrics calculated using Diarization Metric in One repository

Support DER, JER, CDER, SER and BER

Usage with docker
Run Docker image in order to test with existing audio file from Voxconverse
'''docker pull avigyan/whisper_ecapa_tdnn'''
'''docker run avigyan/whisper_ecapa_tdnn'''

In order to test using custom wav files perform the following:

'''1.Clone repository'''
'''2.Add wav file and rttm in app folder'''
'''3.Copy the location of the wav file in the path variable```
Example:
```path = "/content/usbgm.wav"``
'''4.change reference rttm name in main.py'''
'''5.Build new image or RUN'''

new image

'''docker build -t whisper_ecapa_tdnn:tagname'''

Run

'''docker run whisper_ecapa_tdnn:tagname'''

Usage without docker(Linux environment required)

'''1.Clone repository'''
'''2.Add wav file and rttm in app folder'''
'''3.Copy the location of the wav file in the path variable```
Example:
```path = "/content/usbgm.wav"```
'''4.change reference rttm name in main.py'''
'''5.Move to app directory and install requirements.txt using ''pip install -r requirements.txt''' 5.run python main.py'''


Results:

  collar    MS    FA    SC    OVL    DER    JER    CDER    SER    BER    ref_part    fa_dur    fa_seg    fa_mean
--------  ----  ----  ----  -----  -----  -----  ------  -----  -----  ----------  --------  --------  ---------
    0.00  0.14  0.00  0.07   0.15   0.21   0.34    0.28   0.37   0.37        0.37      0.00      0.00       0.00
Reference
https://github.com/nryant/dscore
https://github.com/liutaocode/dscore-ovl
https://github.com/SpeechClub/CDER_Metric
https://github.com/X-LANCE/BER
https://github.com/pyannote/pyannote-audio
