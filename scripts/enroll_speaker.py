"""Create an optional CAM++ profile that restricts lip motion to one speaker."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from avatar_speaker import VoiceEmbedding
from avatar_voice import decode_audio

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path)
    parser.add_argument('reference', type=Path)
    args = parser.parse_args()
    audio = decode_audio(args.reference, 16000, limit=30)
    if len(audio) < 32000:
        parser.error('Use at least two seconds of clear speech.')
    model = VoiceEmbedding()
    vectors = [model.compute(audio[index:index+32000])
               for index in range(0, len(audio)-31999, 32000)]
    reference = np.mean(vectors, axis=0)
    norm = np.linalg.norm(reference)
    if not np.isfinite(norm) or norm < 1e-6:
        parser.error('Could not compute a valid speaker embedding.')
    args.profile.mkdir(parents=True, exist_ok=True)
    with (args.profile/'owner.npy').open('xb') as file:
        np.save(file, (reference/norm).astype(np.float32), allow_pickle=False)
