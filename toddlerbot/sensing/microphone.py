"""Microphone interface module for audio input.

Provides Microphone class for capturing audio input from USB camera microphones
or other audio input devices.
"""

import sounddevice as sd


class Microphone:
    """Microphone class for handling audio input from a microphone device."""

    def __init__(self, mic_names=["USB 2.0 Camera", "Arducam USB Camera"]):
        """Initializes the object by searching for a microphone device with the specified name.

        Args:
            mic_name (str): The name of the microphone to search for. Defaults to "USB 2.0 Camera".

        Attributes:
            device (int or None): The index of the found microphone device, or None if not found.
        """
        self.device = None
        for i, device in enumerate(sd.query_devices()):
            if any(mic_name in device["name"] for mic_name in mic_names):
                self.device = i
                print(f"Found microphone device at index: {i}")
                break
