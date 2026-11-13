"""Prepare matching portrait pixels and DH mouth geometry from a local image or video."""
import argparse
import os
import pickle
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import imageio_ffmpeg
import mediapipe as mp
import numpy as np

from avatar_paths import DH, MODELS


def prepare(source, output, start, seconds):
    if (output / 'combined_data.json.gz').exists():
        raise FileExistsError(f'Choose a new profile: {output}')
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(DH))
    os.chdir(DH)
    # Some OpenCV wheels cannot read an absolute Windows path containing Cyrillic.
    def read_image(path, flags=cv2.IMREAD_COLOR):
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)

    cv2.imread = read_image
    from data_preparation_web import generate_combined_data, step1_crop_mouth

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_buffer=(MODELS / 'face_landmarker.task').read_bytes()),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
    )
    is_image = source.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with TemporaryDirectory(prefix='avatar-prep-') as work:
        work = Path(work)
        input_options = ['-loop', '1'] if is_image else ['-ss', str(start)]
        subprocess.run(
            [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', *input_options,
             '-i', str(source), '-an', '-t', str(seconds),
             '-vf', 'scale=720:720:force_original_aspect_ratio=increase,crop=720:720,fps=25',
             '-pix_fmt', 'yuv420p', '-c:v', 'libx264', '-threads', '2', str(work / 'processed.mp4')],
            check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
        rows = []
        capture = cv2.VideoCapture(str(work / 'processed.mp4'))
        try:
            with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if not rows:
                        portrait = frame.copy()
                    if is_image and rows:
                        rows.append(rows[0])
                        continue
                    image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    result = detector.detect(image)
                    if len(result.face_landmarks) != 1:
                        raise ValueError(f'Expected one visible face in frame {len(rows)}')
                    rows.append([(p.x * 720, p.y * 720, p.z * 720)
                                 for p in result.face_landmarks[0]])
        finally:
            capture.release()
        if len(rows) < 20:
            raise ValueError('The source must have at least 20 usable frames at 25 FPS.')
        points = np.asarray(rows)
        with (work / 'processed.pkl').open('wb') as file:
            pickle.dump(points, file)
        rects, standardized = step1_crop_mouth(points, 720, 720)
        assets = work / 'assets'
        assets.mkdir()
        generate_combined_data(rects, standardized, str(work), str(assets))
        (output / 'portrait.png').write_bytes(cv2.imencode('.png', portrait)[1].tobytes())
        (output / 'combined_data.json.gz').write_bytes((assets / 'combined_data.json.gz').read_bytes())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('profile', type=Path)
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--seconds', type=float, default=2)
    args = parser.parse_args()
    prepare(args.source.resolve(strict=True), args.profile.resolve(), args.start, args.seconds)
