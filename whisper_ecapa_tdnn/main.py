# main.py

import subprocess

# First line of code
subprocess.run(["python", "model.py"])

# change reference rttm file
subprocess.run(["python", "evaluation.py", "-r", "aepyx.rttm", "-s", "aepyx_hyp.rttm"])