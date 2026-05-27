"""Test speaker functionality by downloading and playing audio.

This module tests the speaker system by downloading an audio file, resampling it
to the correct sample rate, and playing it through the ToddlerBot speaker.
"""

import os

import numpy as np
import requests
import sounddevice as sd
import soxr
from pydub import AudioSegment

from toddlerbot.sensing.speaker import Speaker

if __name__ == "__main__":
    # URL of the audio file (Replace with the actual URL)
    # Download the MP3
    audio_path = "/tmp/downloaded_audio.mp3"

    if not os.path.exists(audio_path):
        audio_url = "https://www.soundjay.com/free-music_c2026/midnight-ride-01a.mp3"

        print("Downloading audio file...")
        response = requests.get(audio_url)
        with open(audio_path, "wb") as f:
            f.write(response.content)
        print(f"Audio downloaded to {audio_path}")

    # Load MP3 with pydub
    song = AudioSegment.from_mp3(audio_path)

    # Convert to NumPy float32 array
    samples = np.array(song.get_array_of_samples()).astype(np.float32) / (1 << 15)
    if song.channels == 2:
        samples = samples.reshape((-1, 2))
    samplerate = song.frame_rate

    # Resample to 44100 Hz
    new_samplerate = 48000
    data_resampled = soxr.resample(samples, samplerate, new_samplerate)

    # Play
    speaker = Speaker()
    sd.play(data_resampled, device=speaker.device)
    sd.wait()
