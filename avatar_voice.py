"""Generate Russian speech locally with the Qwen3-TTS reference voice."""
import argparse
from pathlib import Path
import subprocess
import wave

import imageio_ffmpeg
import numpy as np

from avatar_paths import MODELS


def decode_audio(source, rate, limit=None):
    duration = ['-t', str(limit)] if limit is not None else []
    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-loglevel', 'error', '-i', str(source),
         *duration, '-vn', '-ac', '1', '-ar', str(rate), '-f', 'f32le', 'pipe:1'],
        stdout=subprocess.PIPE, check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
    )
    return np.frombuffer(result.stdout, dtype='<f4').copy()


def write_wave(path, samples, rate):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as output:
        output.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
        output.writeframes((np.clip(samples, -1, 1) * 32767).astype('<i2').tobytes())


class Voice:
    def __init__(self, engine, reference, transcript=''):
        self.engine = engine
        pcm = decode_audio(reference, 24000, limit=20)
        if len(pcm) < 48000:
            raise ValueError('Use at least two seconds of clear reference speech.')
        self.reference = self.engine.extract_voice_ref(pcm)
        self.transcript = transcript

    def render(self, text, output):
        samples, rate = self.engine.synthesize(
            text=text, lang='russian', ref_spk_emb=self.reference.ref_spk_emb,
            ref_codes=self.reference.ref_codes if self.transcript else None,
            ref_text=self.transcript or None, seed=42, max_new_tokens=350,
        )
        if not len(samples):
            raise RuntimeError('Qwen3-TTS returned empty audio')
        write_wave(output, samples, rate)


def open_engine():
    from qwentts_cpp import QwenTTS

    return QwenTTS(MODELS/'tts/qwen-talker-1.7b-base-Q8_0.gguf',
                   MODELS/'tts/qwen-tokenizer-12hz-Q8_0.gguf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--transcript', type=Path)
    parser.add_argument('--text', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with open_engine() as engine:
        transcript = args.transcript.read_text(encoding='utf-8').strip() if args.transcript else ''
        Voice(engine, args.reference, transcript).render(args.text, args.output)
