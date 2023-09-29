FROM python:3.10-bookworm

COPY ./whisper_ecapa_tdnn /app/app

WORKDIR /app/app



RUN pip install -r requirements.txt
RUN chmod +x dscore-ovl/scorelib/md-eval-22.pl



CMD [ "python","main.py" ]




