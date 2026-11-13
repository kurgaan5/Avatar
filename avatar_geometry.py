"""Face landmarks and the LivePortrait model used by the portrait renderer."""
import hashlib
import sys

import cv2
import mediapipe as mp
import numpy as np
import torch
import yaml

from avatar_paths import LIVEPORTRAIT, MODELS

def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()

def read(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)

EYES = ((33, 133, 159, 145), (362, 263, 386, 374))


def face_tracker(mode):
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=(MODELS / "face_landmarker.task").read_bytes()),
        running_mode=mode,
        num_faces=1,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def face_points(tracker, frame, timestamp=None):
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = tracker.detect(image) if timestamp is None else tracker.detect_for_video(image, timestamp)
    if not result.face_landmarks:
        return None
    height, width = frame.shape[:2]
    return np.array([(point.x * width, point.y * height) for point in result.face_landmarks[0]], dtype=np.float32)


def rotation(angles):
    # LivePortrait operates on row-vector keypoints and expects the transposed Euler rotation.
    (a, b, c), (d, e, f) = angles.deg2rad().cos().unbind(), angles.deg2rad().sin().unbind()
    return torch.stack((b * c, b * f, -e, d * e * c - a * f, d * e * f + a * c,
                        d * b, a * e * c + d * f, a * e * f - d * c, a * b)).reshape(1, 3, 3)


def crop_matrix(points, factor=1.85):
    face = points[[10,152,234,454,127,356,172,397], :2]
    lo, hi = face.min(0), face.max(0)
    side = max(hi-lo)*factor
    center = (lo+hi)/2 - [0, side*.045]
    k = 256/side
    return np.float32([[k,0,128-center[0]*k],[0,k,128-center[1]*k]])


class Identity:
    @staticmethod
    def image(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).cuda().permute(2, 0, 1).unsqueeze(0).float().div_(255)

    def measure(self, image):
        with torch.autocast("cuda", dtype=torch.float16):
            info = self.motion(image)
        result = {key: info[key].float() for key in ("kp", "exp", "t", "scale")}
        result["kp"] = result["kp"].reshape(1, -1, 3)
        result["exp"] = result["exp"].reshape(1, -1, 3)
        result["angles"] = torch.stack([
            (info[key].float().softmax(dim=1) * self.angle_bins).sum() * 3 - 97.5
            for key in ("pitch", "yaw", "roll")
        ])
        return result

    @torch.inference_mode()
    def __init__(self, source, points):
        sys.path.insert(0, str(LIVEPORTRAIT.parent))
        from LivePortrait.src.modules.appearance_feature_extractor import AppearanceFeatureExtractor
        from LivePortrait.src.modules.motion_extractor import MotionExtractor
        from LivePortrait.src.modules.spade_generator import SPADEDecoder
        from LivePortrait.src.modules.stitching_retargeting_network import StitchingRetargetingNetwork
        from LivePortrait.src.modules.warping_network import WarpingNetwork
        params = yaml.safe_load((LIVEPORTRAIT/'src/config/models.yaml').read_text(encoding='utf-8'))['model_params']
        weights = LIVEPORTRAIT/'weights/liveportrait'
        self.hashes = {}
        def load(cls, key):
            path = weights/f'base_models/{key}.pth'
            self.hashes[str(path.relative_to(LIVEPORTRAIT))] = digest(path)
            net = cls(**params[key+'_params'])
            net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            return net.cuda().eval()
        appearance = load(AppearanceFeatureExtractor, 'appearance_feature_extractor')
        self.motion = load(MotionExtractor, 'motion_extractor')
        self.warp = load(WarpingNetwork, 'warping_module')
        self.decode = load(SPADEDecoder, 'spade_generator')
        for layer in self.decode.modules():
            if hasattr(layer, 'weight_orig'):
                torch.nn.utils.remove_spectral_norm(layer)
        self.decode = self.decode.half().to(memory_format=torch.channels_last)
        path = weights/'retargeting_models/stitching_retargeting_module.pth'
        self.hashes[str(path.relative_to(LIVEPORTRAIT))] = digest(path)
        state = torch.load(path, map_location='cpu', weights_only=True)
        self.eyes = StitchingRetargetingNetwork(**params['stitching_retargeting_module_params']['eye']).cuda().eval()
        self.eyes.load_state_dict({k.removeprefix('module.'):v for k,v in state['retarget_eye'].items()})
        self.angle_bins = torch.arange(66, device='cuda', dtype=torch.float32)
        self.affine = crop_matrix(points)
        hi_affine = self.affine.copy()*2
        self.crop = cv2.warpAffine(source, hi_affine, (512,512), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        im = self.image(cv2.resize(self.crop,(256,256),interpolation=cv2.INTER_AREA))
        self.src = self.measure(im)
        with torch.autocast('cuda', dtype=torch.float16):
            self.feature = appearance(im)
        self.keys = self.keypoints(rotation(self.src['angles']), self.src['t'], self.src['scale'])
        self.eye_ratio = self.keys.new_tensor([np.linalg.norm(points[t]-points[b])/np.linalg.norm(points[l]-points[r]) for l,r,t,b in EYES])
        self.eye_base = self.eye_delta(self.eye_ratio.mean())
        self.neutral = self.render(np.zeros(3), np.zeros((1,3)), 1, 0)
        self.gray = cv2.cvtColor(self.neutral, cv2.COLOR_BGR2GRAY)
        self.grid = np.stack(np.meshgrid(np.arange(512),np.arange(512)),axis=-1).astype(np.float32)
        self.inverse = cv2.invertAffineTransform(hi_affine)
        # Blend only the generated head/upper neck. Keep original body and backdrop.
        yy,xx = np.mgrid[:512,:512]
        edge = np.minimum.reduce([xx,511-xx,yy,511-yy])
        self.mask = np.minimum(np.clip((edge-8)/40,0,1),np.clip((500-yy)/85,0,1)).astype(np.float32)
        self.mask = self.mask*self.mask*(3-2*self.mask)
        eye_mask = np.zeros((512,512),np.float32)
        for l,r,t,b in EYES:
            p = cv2.transform(points[None,[l,r]],hi_affine)[0]
            center = np.rint(p.mean(0)).astype(int)
            width = np.linalg.norm(p[1]-p[0])
            cv2.ellipse(eye_mask,tuple(center),(round(width*.8),round(width*.45)),0,0,360,1,-1)
        self.eye_mask = cv2.GaussianBlur(eye_mask,(0,0),6)

    def eye_delta(self, desired):
        f = torch.cat([self.keys.flatten(),self.eye_ratio,desired.reshape(1)])
        return self.eyes(f[None]).reshape_as(self.keys)

    def keypoints(self, r, t, scale):
        p = scale.reshape(1,1,1)*(self.src['kp']@r+self.src['exp'])
        p[...,:2] += t[:,None,:2]
        return p

    @torch.inference_mode()
    def render(self, angles, translation, scale, closure):
        r = rotation(self.src['angles'] + self.src['angles'].new_tensor(angles))
        t = self.src['t'] + self.src['t'].new_tensor(translation)
        keys = self.keypoints(r,t,self.src['scale']*scale)
        desired = self.eye_ratio.mean()*(1-closure) + .018*closure
        keys += self.eye_delta(desired)-self.eye_base
        with torch.autocast('cuda',dtype=torch.float16):
            w = self.warp(self.feature,kp_source=self.keys,kp_driving=keys)['out']
            out = self.decode(w.contiguous(memory_format=torch.channels_last))
        im = out[0].permute(1,2,0).mul(255).clamp(0,255).byte().cpu().numpy()
        return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

