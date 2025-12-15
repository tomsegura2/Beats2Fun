import os
from typing import List

import librosa

from . import BeatInput, BeatOption, BeatList

file_desc = 'Audio file (*.mp3, *.wav, *.ogg)|*.mp3;*.wav;*.ogg'


def _detect_beats(audio_path: str) -> List[float]:
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, trim=False)

    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    duration = librosa.get_duration(y=y, sr=sr)

    if not beat_times or beat_times[0] > 0.01:
        beat_times = [0.0] + beat_times

    beat_times = [b for b in beat_times if b < duration]

    if duration > 0 and (not beat_times or duration - beat_times[-1] > 0.01):
        beat_times.append(duration)

    if len(beat_times) < 2:
        return []

    return beat_times


class AudioBeatOption(BeatOption):
    def __init__(self, path: str):
        super().__init__(-1, "detected")
        self.path = path

    def load(self):
        beat_times = _detect_beats(self.path)
        if not beat_times:
            raise Exception("No beats detected in {}".format(self.path))
        self.beat_list = BeatList(beat_times)


class AudioParser(BeatInput):
    file_desc = file_desc
    extensions = ['.mp3', '.wav', '.ogg']

    def read_file(self, path):
        super().read_file(path)
        self.options.append(AudioBeatOption(path))
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.song = path

    @staticmethod
    def write_file(option, path):
        raise Exception("Writing beats for audio detection not supported")


def process_input(input, option=None):
    if not os.path.isfile(input):
        return False

    filename, ext = os.path.splitext(input)
    if ext not in AudioParser.extensions:
        return False

    beat_times = _detect_beats(input)
    if not beat_times:
        return False

    return (input, beat_times)


def find_options(input):
    if not os.path.isfile(input):
        return False

    filename, ext = os.path.splitext(input)
    if ext not in AudioParser.extensions:
        return False

    return [{
        'level': -1,
        'name': os.path.basename(filename)
    }]
