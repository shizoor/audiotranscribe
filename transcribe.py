import os
#from pydub import AudioSegment
import whisper
import soundfile as sf
from datetime import datetime, timezone


# Convert MP3 to WAV using soundfile
def convert_mp3_to_wav(mp3_path, wav_path):
    data, samplerate = sf.read(mp3_path)
    sf.write(wav_path, data, samplerate)

# Function to format time for SRT
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:06.3f}".replace(".", ",")

# Function to write SRT file
def write_srt(segments, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            start_time = format_time(segment["start"])
            end_time = format_time(segment["end"])
            text = segment["text"].strip()
            f.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")

def aware_utcnow():
    return datetime.now(timezone.utc)

# Function to write plain text file
def write_txt(segments, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for segment in segments:
            f.write(segment["text"].strip() + "\n")

# Main function to process files
def transcribe_mp3_to_srt_and_txt(input_dir, output_dir):
    # Load the Whisper model
    model = whisper.load_model("large")  # Choose the model size (tiny, base, small, medium, large)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each MP3 file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".mp3"):
            # Construct file paths
            mp3_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            wav_path = os.path.join(output_dir, f"{base_name}.wav")
            srt_path = os.path.join(output_dir, f"{base_name}.srt")
            txt_path = os.path.join(output_dir, f"{base_name}.txt")

            print(aware_utcnow(), f"Checking if file {txt_path} exists..")
            if not os.path.exists(txt_path) :
             
                # Convert MP3 to WAV
                print(aware_utcnow(), f"Converting {filename} to WAV...")
                convert_mp3_to_wav(mp3_path, wav_path)

                # Transcribe the WAV file
                print(aware_utcnow(), f"Transcribing {filename}...")
                result = model.transcribe(wav_path, word_timestamps=True)

                # Write SRT file
                print(aware_utcnow(), f"Writing SRT file for {filename}...")
                write_srt(result["segments"], srt_path)

                # Write plain text file
                print(aware_utcnow(), f"Writing TXT file for {filename}...")
                write_txt(result["segments"], txt_path)

                print(aware_utcnow(), f"Finished processing {filename}")

                os.remove(wav_path)
                print(aware_utcnow(), "Deleted " + wav_path)
            else : 
                print(f"{txt_path} exists, skipping.")



# Run the program
if __name__ == "__main__":
    # Set your input and output directories
    input_directory = "./input"  # Replace with your input directory
    output_directory = "./output"  # Replace with your output directory

    # Transcribe all MP3 files in the input directory
    transcribe_mp3_to_srt_and_txt(input_directory, output_directory)
