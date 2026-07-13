import dlib
from drive import open_url
from pathlib import Path
import argparse
import sys
from bicubic import BicubicDownSample
import torchvision
from shape_predictor import align_face
from device import device

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

parser = argparse.ArgumentParser(description='PULSE')

parser.add_argument('-input_dir', type=str, default='realpics', help='directory with unprocessed images')
parser.add_argument('-output_dir', type=str, default='input', help='output directory')
parser.add_argument('-output_size', type=int, default=32, help='size to downscale the input images to, must be power of 2')
parser.add_argument('-seed', type=int, help='manual seed to use')
parser.add_argument('-cache_dir', type=str, default='cache', help='cache directory for model weights')

args = parser.parse_args()

cache_dir = Path(args.cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True,exist_ok=True)

predictor_path = Path("shape_predictor_68_face_landmarks.dat")
if predictor_path.exists():
    print("Using local Shape Predictor")
    f = str(predictor_path)
else:
    print("Downloading Shape Predictor")
    # f=open_url("https://drive.google.com/uc?id=1huhv8PYpNNKbGCLOaYUjOgR1pY5pmbJx", cache_dir=cache_dir, return_path=True)
    f=open_url("https://ericeaglstun.com/misc/shape_predictor_68_face_landmarks.dat", cache_dir=cache_dir, return_path=True)
predictor = dlib.shape_predictor(f)

input_dir = Path(args.input_dir)
if not input_dir.is_dir():
    sys.exit(f"ERROR: input directory {input_dir} does not exist")

images = sorted(p for p in input_dir.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
if not images:
    sys.exit(f"ERROR: no images found in {input_dir} "
             f"(looked for: {', '.join(sorted(IMAGE_SUFFIXES))})")

written = 0
failed = []

for im in images:
    try:
        faces = align_face(str(im),predictor)
    except Exception as e:
        print(f"{im.name}: ERROR: could not process ({e})")
        failed.append(im.name)
        continue

    if not faces:
        print(f"{im.name}: ERROR: no face detected, nothing written")
        failed.append(im.name)
        continue

    for i,face in enumerate(faces):
        if(args.output_size):
            factor = 1024//args.output_size
            assert args.output_size*factor == 1024
            D = BicubicDownSample(factor=factor)
            face_tensor = torchvision.transforms.ToTensor()(face).unsqueeze(0).to(device)
            face_tensor_lr = D(face_tensor)[0].cpu().detach().clamp(0, 1)
            face = torchvision.transforms.ToPILImage()(face_tensor_lr)

        out_path = Path(args.output_dir) / (im.stem+f"_{i}.png")
        face.save(out_path)
        print(f"{im.name}: wrote {out_path}")
        written += 1

print(f"\nDone: {written} face(s) written to {args.output_dir} "
      f"from {len(images)} input image(s); {len(failed)} failed.")

if failed:
    sys.exit(f"ERROR: no usable face in: {', '.join(failed)}")
