Integration of Whisper ASR model ,ECAPAT-TDNN model and  Evaluation for Speaker Diarization
Metrics calculated using Diarization Metric in One repository

Support DER, JER, CDER, SER and BER

<h3>Usage with docker</h3> <br>
Run Docker image in order to test with existing audio file from Voxconverse  <br>
'''docker pull avigyan/whisper_ecapa_tdnn'''  <br>
'''docker run avigyan/whisper_ecapa_tdnn'''  <br>

In order to test using custom wav files perform the following:  <br>

'''1.Clone repository'''  <br> 
'''2.Add wav file and rttm in app folder'''  <br>
'''3.Copy the location of the wav file in the path variable```  <br>
Example:  <br>
```path = "/content/usbgm.wav"``  <br>
'''4.change reference rttm name in main.py'''  <br>
'''5.Build new image or RUN'''  <br>

new image  <br>

'''docker build -t whisper_ecapa_tdnn:tagname'''  <br>

Run ****
 
'''docker run whisper_ecapa_tdnn:tagname''' <br>

<h3>Usage without docker(Linux environment required) </h3> <br>

'''1.Clone repository'''  <br>
'''2.Add wav file and rttm in app folder'''  <br>
'''3.Copy the location of the wav file in the path variable```  <br>
Example: <br>
```path = "/content/usbgm.wav"```  <br>
'''4.change reference rttm name in main.py'''  <br>
'''5.Move to whisper_ecapa_tdnn directory and install requirements.txt using ''pip install -r requirements.txt''' <br>
5.run python main.py'''  <br>


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
