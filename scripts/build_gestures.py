"""Extract reusable head and shoulder motion from a local, neutral video."""
import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import imageio_ffmpeg
import mediapipe as mp
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from avatar_paths import MODELS

WIDTH, HEIGHT, FPS = 480, 270, 12.5


def read_frames(source, start, duration):
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-loglevel', 'error',
               '-ss', str(start), '-i', str(source), '-t', str(duration), '-an',
               '-vf', f'scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT},fps={FPS}',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    raw = subprocess.run(command, check=True, stdout=subprocess.PIPE,
                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, HEIGHT, WIDTH, 3)


def track(frames):
    vision = mp.tasks.vision
    face_options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=(MODELS/'face_landmarker.task').read_bytes()),
        running_mode=vision.RunningMode.VIDEO, output_facial_transformation_matrixes=True,
    )
    pose_options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=(MODELS/'pose_landmarker_lite.task').read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
    )
    rows = []
    with (vision.FaceLandmarker.create_from_options(face_options) as face,
          vision.PoseLandmarker.create_from_options(pose_options) as pose):
        for index, frame in enumerate(frames):
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            timestamp = round(index * 1000 / FPS)
            detected = face.detect_for_video(image, timestamp)
            torso = pose.detect_for_video(image, timestamp)
            if not detected.face_landmarks or not torso.pose_landmarks:
                rows.append(None)
                continue
            matrix = np.asarray(detected.facial_transformation_matrixes[0])[:3, :3]
            u, _, v = np.linalg.svd(matrix)
            angles = Rotation.from_matrix(u @ v).as_euler('xyz', degrees=True)
            a, b = (torso.pose_landmarks[0][index] for index in (11, 12))
            if min(a.visibility,b.visibility) < .5:
                rows.append(None)
                continue
            shoulder = [(a.x+b.x)*WIDTH/2, (a.y+b.y)*HEIGHT/2,
                        np.hypot((b.x-a.x)*WIDTH, (b.y-a.y)*HEIGHT),
                        np.degrees(np.arctan2((a.y-b.y)*HEIGHT, (a.x-b.x)*WIDTH))]
            points = detected.face_landmarks[0]
            center = [(points[234].x+points[454].x)*WIDTH/2,
                      (points[234].y+points[454].y)*HEIGHT/2]
            rows.append([*angles, *shoulder, *center])
    return rows


def extract(rows):
    clips = []
    for offset in range(0, len(rows)-49, 25):
        window = rows[offset:offset+50]
        if any(row is None for row in window):
            continue
        data = savgol_filter(np.asarray(window), 9, 2, axis=0)
        rel = data - data[0]
        angles = rel[:,:3]
        if np.ptp(angles, axis=0).max() < 1.8:
            continue
        angles *= min(1.5, 1/max(np.max(np.abs(angles)/[3.8,6.,2.2]),.001))
        torso = np.column_stack((rel[:,6]*.35, rel[:,3:5]*720/HEIGHT*.22,
                                 rel[:,5]/max(1, data[0,5])*.22))
        torso = np.clip(torso, [-.8,-4,-3,-.008], [.8,4,3,.008])
        shift = np.clip(rel[:,7:9]/180*.14, -.008, .008)
        clip = np.column_stack((angles, torso, shift))
        fade = np.linspace(0,1,8)[:,None]
        fade = fade**2 * (3 - 2*fade)
        clip[:8] *= fade
        clip[-8:] = clip[-8:]*(1-fade) + clip[-1:]*fade
        clips.append(clip.astype(np.float32))
    if len(clips) < 2:
        raise ValueError('Record a longer seated video with visible shoulders and several small turns.')
    return np.stack(clips)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('profile', type=Path)
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--seconds', type=float, default=32)
    args = parser.parse_args()
    clips = extract(track(read_frames(args.video, args.start, args.seconds)))
    args.profile.mkdir(parents=True, exist_ok=True)
    with (args.profile/'gestures.npz').open('xb') as file:
        np.savez(file, clips=clips, fps=np.float32(FPS))
    print(f'{len(clips)} gesture clips saved.')
