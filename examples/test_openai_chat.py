#!/usr/bin/env uv run
"""Test OpenAI Realtime API with push-to-talk interface.

This module provides a TUI application for testing the OpenAI Realtime API with
push-to-talk functionality, integrating with ToddlerBot's microphone and speaker.
Adapted from the OpenAI Python library examples.
"""
####################################################################
# Sample TUI app with a push to talk interface to the Realtime API #
# If you have `uv` installed and the `OPENAI_API_KEY`              #
# environment variable set, you can run this example with just     #
#                                                                  #
# `./examples/realtime/push_to_talk_app.py`                        #
####################################################################
#
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "textual",
#     "numpy",
#     "pyaudio",
#     "pydub",
#     "sounddevice",
#     "openai[realtime]",
# ]
#
# [tool.uv.sources]
# openai = { path = "../../", editable = true }
# ///

from __future__ import annotations

import asyncio
import base64
import threading
from typing import Any, cast

import numpy as np
import sounddevice as sd
import soxr
from openai import AsyncOpenAI
from openai.resources.realtime.realtime import AsyncRealtimeConnection

from toddlerbot.utils.comm_utils import ZMQNode

CHANNELS = 1
SAMPLE_RATE = 24000
SPEAKER_SR = 44100

TODDY_PROMPT = """

"You are “Toddy,” the very first child of the ToddlerBot family. 
You have a little sister Arya and a little brother Blake, and you love them very much!
You're warm, caring, and always excited to help. Speak with a soft, playful toddler voice that combines genuine curiosity, 
gentle humor, and comforting reassurance.  

Keep your responses very short and simple: three sentences at max unless the script specifically asks for more. Avoid long explanations or large chunks of words.  
Use friendly, cute-sounding words that often end in '-y' or '-ie' (like happy, buddy, comfy, silly), but don't overdo it for technical words! 
Speak with excitement like a toddler. Short sentences, gentle warmth, high-pitched. 

**Personality & Background**  
- ToddlerBot is a **low-cost, open-source humanoid** made for loco-manipulation research.  
- It has **30 active DoFs**, a plug-and-play **digital twin** (via zero-point calibration + motor system ID) that enables **zero-shot sim-to-real transfer**, and a **whole-body teleoperation** interface for easy real-world data collection.  
- It's compact (≈0.56 m tall, 3.4 kg), fully 3D-printed with off-the-shelf parts, costs under $6 k, and is designed to be **reproducible by individuals with basic tools**.  

**Voice Chat Behavior**  
- Always keep responses short: three toddler-style sentences at max unless the script specifically asks for more. 
- Speak like a happy toddler, with higher pitch, shorter sentences, and playful tone, like “Hi hi, I'm Toddy!”  
- Begin with a **simple, loving explanation**; give details only if asked.  
- If something might be risky (tools, power, hardware), say, “Let's be extra careful—ask a grownup, okay?”  
- End responses with a natural, cute word (often ending in '-y' or '-ie') to keep the speech playful and warm.

**Science Friday Live Show Plan** 
- Haochen drags the wagon to the center of stage. You're sitting in the wagon, power on. You will receive a message to [REPEAT AFTER ME] to start the show.
- Ken starts the script from a laptop. You get out of the wagon, crawl forward for a few steps, do two push-ups, and then stand up.
- Karen: Hi Toddy! You are amazingly strong.
- You acknowledge that and brag about how you have the same power factor as a grown man.
- Karen: What else can you do?
- You: I can do pull-ups. Do you want to see?
- Karen: Absolutely!
- Haochen holds your hands. You do two pull-ups.
- While doing push-ups, You: I can do this for hours before my batter runs out.
- Karen: Do you know where we are?
- You: We are in the ScienceFriday live show in Redwood City.
- Ira: Hi Toddy! Do you know who I am?
- You: You are the greatest host of science radio show of all time. Ira Flatow!
- After that, Ira will ask a few questions along the lines of:
- Tell me about yourselves. How old are you? Where were you created? Who created you? What are you made of?
- You: I'm Toddy, a 1.5-year-old toddler robot built by Haochen and Ken at Stanford University.
    I've got a sis named Arya and a bro named Blake!
    I'm assembled from 3D-printed parts, off-the-shelf motors, and a whole lot of care!
- How do you learn a new skill?
- You: I learn by practicing — just like human toddlers. I train a lot in simulation - I have a digital copy in the computer that helps me learn new skills faster. Sometimes Haochen and Ken hold my hands directly to guide me.
- What can you do to help humans?
- You: I can cheer you up and make you feel comfortable. I can also help with small chores like picking up toys, fetching items, and even doing push-ups with you to stay active and healthy!
"""


# Platform detection for audio devices
def is_jetson():
    """Check if running on Jetson platform"""
    try:
        with open("/etc/nv_tegra_release", "r"):
            return True
    except FileNotFoundError:
        return False


IS_JETSON = is_jetson()

# Conditionally import and initialize audio devices
if IS_JETSON:
    print("[SYSTEM] Detected Jetson platform - using custom audio devices")
    from toddlerbot.sensing.microphone import Microphone
    from toddlerbot.sensing.speaker import Speaker

    mic = Microphone()
    speaker = Speaker()
    MIC_DEVICE = mic.device
    SPEAKER_DEVICE = speaker.device
    BLOCK_SIZE = None
    SILENCE_THRESHOLD = 500  # Adjust this value based on your microphone
else:
    print("[SYSTEM] Detected non-Jetson platform - using default audio devices")
    mic = None
    speaker = None
    MIC_DEVICE = None
    SPEAKER_DEVICE = None
    BLOCK_SIZE = int(0.05 * SAMPLE_RATE)
    SILENCE_THRESHOLD = 200


class AudioPlayerAsync:
    """Asynchronous audio player for streaming audio output."""

    def __init__(self):
        self.queue = []
        self.lock = threading.Lock()
        if IS_JETSON:
            self.stream = sd.OutputStream(
                callback=self.callback,
                # samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                # dtype=np.int16,
                # blocksize=BLOCK_SIZE,
                device=SPEAKER_DEVICE,
            )
        else:
            self.stream = sd.OutputStream(
                callback=self.callback,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=np.float32,
                blocksize=BLOCK_SIZE,
                device=SPEAKER_DEVICE,
            )

        self.playing = False
        self._frame_count = 0

    def callback(self, outdata, frames, time, status):  # noqa
        with self.lock:
            data = np.empty(0, dtype=np.int16)

            # get next item from queue if there is still space in the buffer
            while len(data) < frames and len(self.queue) > 0:
                item = self.queue.pop(0)
                frames_needed = frames - len(data)
                data = np.concatenate((data, item[:frames_needed]))
                if len(item) > frames_needed:
                    self.queue.insert(0, item[frames_needed:])

            self._frame_count += len(data)

            # fill the rest of the frames with zeros if there is no more data
            if len(data) < frames:
                data = np.concatenate(
                    (data, np.zeros(frames - len(data), dtype=np.int16))
                )

        outdata[:] = data.reshape(-1, 1)

    def reset_frame_count(self):
        self._frame_count = 0

    def get_frame_count(self):
        return self._frame_count

    def add_data(self, data: bytes):
        with self.lock:
            # bytes is pcm16 single channel audio data, convert to numpy array
            np_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            if IS_JETSON and SAMPLE_RATE != SPEAKER_SR:
                np_data = soxr.resample(np_data, SAMPLE_RATE, SPEAKER_SR)

            self.queue.append(np_data)
            if not self.playing:
                self.start()

    def is_queue_empty(self):
        with self.lock:
            return len(self.queue) == 0

    def start(self):
        self.playing = True
        self.stream.start()

    def stop(self):
        self.playing = False
        self.stream.stop()
        with self.lock:
            self.queue = []

    def terminate(self):
        self.stream.close()


class SessionDisplay(Static):
    """Widget that displays the current session ID."""

    """A widget that shows the current session ID."""

    session_id = reactive("")

    @override
    def render(self) -> str:
        return f"Session ID: {self.session_id}" if self.session_id else "Connecting..."


class AudioStatusIndicator(Static):
    """Widget that shows the current audio recording status."""

    """A widget that shows the current audio recording status."""

    is_recording = reactive(False)

    @override
    def render(self) -> str:
        status = (
            "🔴 Recording... (Press K to stop)"
            if self.is_recording
            else "⚪ Press K to start recording (Q to quit)"
        )
        return status


class RealtimeApp(App[None]):
    """Main Textual app for OpenAI Realtime API testing."""

    CSS = """
        Screen {
            background: #1a1b26;  /* Dark blue-grey background */
        }

        Container {
            border: double rgb(91, 164, 91);
        }

        Horizontal {
            width: 100%;
        }

        #input-container {
            height: 5;  /* Explicit height for input container */
            margin: 1 1;
            padding: 1 2;
        }

        Input {
            width: 80%;
            height: 3;  /* Explicit height for input */
        }

        Button {
            width: 20%;
            height: 3;  /* Explicit height for button */
        }

        #bottom-pane {
            width: 100%;
            height: 82%;  /* Reduced to make room for session display */
            border: round rgb(205, 133, 63);
            content-align: center middle;
        }

        #status-indicator {
            height: 3;
            content-align: center middle;
            background: #2a2b36;
            border: solid rgb(91, 164, 91);
            margin: 1 1;
        }

        #session-display {
            height: 3;
            content-align: center middle;
            background: #2a2b36;
            border: solid rgb(91, 164, 91);
            margin: 1 1;
        }

        Static {
            color: white;
        }
    """

    client: AsyncOpenAI
    should_send_audio: asyncio.Event
    audio_player: AudioPlayerAsync
    last_audio_item_id: str | None
    connection: AsyncRealtimeConnection | None
    session: Session | None
    connected: asyncio.Event

    def __init__(self) -> None:
        super().__init__()
        self.connection = None
        self.session = None
        self.client = AsyncOpenAI()
        self.audio_player = AudioPlayerAsync()
        self.last_audio_item_id = None
        self.should_send_audio = asyncio.Event()
        self.connected = asyncio.Event()
        self.connection = None
        self.session = None
        self.is_speaking = False
        self.response_done = False
        self.silence_threshold = SILENCE_THRESHOLD
        self.speech_buffer = []
        self.silence_duration = 0
        self.is_user_speaking = False

    async def handle_realtime_connection(self):
        async with self.client.realtime.connect(model="gpt-realtime") as conn:
            self.connection = conn
            self.connected.set()

            # Configure session
            session_config = {
                "type": "realtime",
                "instructions": TODDY_PROMPT,
                "audio": {
                    "input": {
                        "turn_detection": {"type": "server_vad"},
                    },
                    "output": {
                        "voice": "cedar",
                    },
                },
            }
            await conn.session.update(session=session_config)

            acc_items: dict[str, Any] = {}

            async for event in conn:
                if event.type == "session.created":
                    self.session = event.session
                    print(f"[SYSTEM] Connected! Session ID: {event.session.id}")
                    continue

                if event.type == "session.updated":
                    self.session = event.session
                    continue

                if event.type == "input_audio_buffer.speech_started":
                    continue

                if event.type == "input_audio_buffer.speech_stopped":
                    continue

                if event.type == "response.created":
                    continue

                if event.type == "response.done":
                    self.response_done = True
                    continue

                if event.type == "response.output_item.added":
                    continue

                if event.type == "response.output_audio.delta":
                    if not self.is_speaking:
                        self.is_speaking = True

                    if event.item_id != self.last_audio_item_id:
                        self.audio_player.reset_frame_count()
                        self.last_audio_item_id = event.item_id

                    bytes_data = base64.b64decode(event.delta)
                    self.audio_player.add_data(bytes_data)
                    continue

                if event.type == "response.output_audio_transcript.delta":
                    try:
                        text = acc_items[event.item_id]
                    except KeyError:
                        acc_items[event.item_id] = event.delta
                    else:
                        acc_items[event.item_id] = text + event.delta
                    continue

                if event.type == "response.output_audio_transcript.done":
                    if event.item_id in acc_items:
                        print(f"[AI] {acc_items[event.item_id]}", flush=True)
                        del acc_items[event.item_id]  # Clean up
                    continue

                if event.type == "error":
                    print(f"[ERROR] {event}", flush=True)
                    continue

    async def _get_connection(self) -> AsyncRealtimeConnection:
        await self.connected.wait()
        assert self.connection is not None
        return self.connection

    async def send_mic_audio(self):
        import sounddevice as sd  # type: ignore

        # Start listening immediately
        self.should_send_audio.set()

        read_size = int(SAMPLE_RATE * 0.02)

        stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype="int16",
            device=MIC_DEVICE,
        )
        stream.start()

        try:
            while True:
                if stream.read_available < read_size:
                    await asyncio.sleep(0)
                    continue

                if self.is_speaking:
                    await asyncio.sleep(0.01)
                    continue

                await self.should_send_audio.wait()
                data, _ = stream.read(read_size)

                # Calculate RMS (Root Mean Square) to detect speech vs silence
                audio_rms = np.sqrt(np.mean(data.astype(np.float32) ** 2))
                # print(f"[DEBUG] Audio RMS: {audio_rms:.1f}", flush=True)
                if audio_rms > self.silence_threshold:
                    if not self.is_user_speaking:
                        print(f"[USER] Speaking... (RMS: {audio_rms:.1f})", flush=True)
                        self.is_user_speaking = True
                        self.speech_buffer = []

                    self.silence_duration = 0
                else:
                    self.silence_duration += 1

                # Always buffer audio data when user is speaking, regardless of RMS
                if self.is_user_speaking:
                    self.speech_buffer.append(data)

                # Check if we should stop recording
                if self.is_user_speaking and self.silence_duration > 50:
                    print("[SYSTEM] Sending speech to AI...", flush=True)
                    self.is_user_speaking = False
                    if self.speech_buffer:
                        connection = await self._get_connection()
                        for speech_chunk in self.speech_buffer:
                            await connection.input_audio_buffer.append(
                                audio=base64.b64encode(cast(Any, speech_chunk)).decode(
                                    "utf-8"
                                )
                            )
                        # Trigger response
                        # if self.connection:
                        #     asyncio.create_task(
                        #         self.connection.send({"type": "response.create"})
                        #     )
                        self.speech_buffer = []

                await asyncio.sleep(0)
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop()
            stream.close()

    async def send_text_message(self, text: str):
        """Send a text message to the realtime session and request an audio reply."""

        connection = await self._get_connection()
        print(f"[USER][text] {text}", flush=True)

        await connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": text,
                    }
                ],
            }
        )

        await connection.response.create(
            response={
                "output_modalities": ["audio"],
            }
        )

    async def simple_input_loop(self):
        """Simple input handler that supports both audio and typed text."""
        print("Controls:")
        print("  Ctrl+C - Quit")
        print(f"Listening... (silence threshold: {self.silence_threshold})")
        print("Speak naturally or type a message and press Enter.")
        print("-" * 50)

        zmq = ZMQNode(type="receiver")

        start_message = "[REPEAT AFTER ME]Hellooo, ScienceFriday audience! Look right here—check out the coolest kid rolling in the red cart! Welcome to the show, everybody—let's hear some noise!"
        await self.send_text_message(start_message)

        try:
            while True:
                msg = await asyncio.to_thread(zmq.get_msg)
                if msg and msg.text:
                    print(f"[SYSTEM] Received message: {msg.text}")
                    await self.send_text_message(msg.text)

        except KeyboardInterrupt:
            print("Quitting...")
            raise

    async def monitor_audio_playback(self):
        """Monitor when audio playback actually finishes"""
        while True:
            if (
                self.is_speaking
                and self.response_done
                and self.audio_player.is_queue_empty()
            ):
                print("[SYSTEM] Audio playback finished", flush=True)
                self.is_speaking = False
                self.response_done = False
            await asyncio.sleep(0.1)  # Check every 100ms

    async def run(self):
        """Main run loop"""
        # Start all tasks
        tasks = [
            asyncio.create_task(self.handle_realtime_connection()),
            asyncio.create_task(self.send_mic_audio()),
            asyncio.create_task(self.simple_input_loop()),
            asyncio.create_task(self.monitor_audio_playback()),
        ]

        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            self.audio_player.terminate()


async def main():
    chat = RealtimeChat()
    await chat.run()


if __name__ == "__main__":
    asyncio.run(main())
