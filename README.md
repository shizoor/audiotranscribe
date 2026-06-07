# audiotranscribe
Script for creating subtitles for media files


# archivedownload (linux)

Download all the hours of slack from Archive.org


# Windows instructions (needs an Nvidia)

python -m venv .venv

.\.venv\Scripts\activate.ps1

pip install git+https://github.com/openai/whisper.git

# Make a file called cudatest.py : 

import torch

print(torch.cuda.is_available())


python cudatest.py

# False is bad.  

# Get cuda installed to the environment (refer to torch and cuda docs if this is out of date) : 

python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python cudatest.py

# Check python cudatest.py returns True.  

python cudatest.py

pip install soundfile

Create a subdirectory called "input", mp3s go in there.  
"output" will contain the srt and txt file outputs.  If it doesn't exist it'll be created.  

python transcribe.py


Linux and Mac users will be able amend these instructions to fit, use python3 instead of python, pip3 instead of pip, etc.
On Linux and Mac python refers to python2, but on Windows it refers to python3.  



