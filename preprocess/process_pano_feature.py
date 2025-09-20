import os
import argparse
import lmdb
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
import pickle

import msgpack
import msgpack_numpy

msgpack_numpy.patch()

import torch

from transformers import CLIPImageProcessor
from flamingo.modeling_flamingo import FlamingoForConditionalGeneration


def build_feature_extractor(args):
    model = FlamingoForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto",
    )
    model.eval()
    preprocessor = CLIPImageProcessor()

    return model, preprocessor


def process_features(args):
    torch.set_grad_enabled(False)
    env = lmdb.open(args.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    model, preprocessor = build_feature_extractor(args)

    pano_keys = []
    images = []

    txn = env.begin()

    for i, (key, value) in tqdm(enumerate(txn.cursor())):
        pano_key = key.decode('ascii')
        image_flt = np.frombuffer(value, dtype=np.uint8)
        image_flt = cv2.imdecode(image_flt, cv2.IMREAD_COLOR)
        image = image_flt.reshape(1500, 1500, 3)
        image = Image.fromarray(image)

        tmp_img = preprocessor(image, return_tensors="pt")["pixel_values"][0]

        pano_keys.append(pano_key)
        images.append(tmp_img)

        if i % args.batch_size == 0:
            images_tensors = torch.stack(images, dim=0).cuda()
            fts = model._encode_vision_x(images_tensors)
            for i, key in enumerate(pano_keys):
                torch.save(fts[i].cpu(), os.path.join(args.output_dir, key + '.pt'))
            pano_keys = []
            images = []

    if len(pano_keys) > 0:
        images_tensors = torch.stack(images, dim=0).cuda()
        fts = model._encode_vision_x(images_tensors)
        for i, key in enumerate(pano_keys):
            torch.save(fts[i].cpu(), os.path.join(args.output_dir, key + '.pt'))
    return


def build_db(args):
    tensor_files = [x for x in os.listdir(args.output_dir) if x.endswith('.pt')]
    tensor_files.sort()

    env = lmdb.open(args.output_dir, map_size=int(1e12))
    for tensor_file in tensor_files:
        panokey = tensor_file[:-3]
        fts = torch.load(os.path.join(args.output_dir, tensor_file))
        fts = fts.numpy()
        txn = env.begin(write=True)
        txn.put(panokey.encode('ascii'), msgpack.packb(fts))
        txn.commit()

    env.close()

def build_feature_file(args):
    os.makedirs(args.output_dir, exist_ok=True)
    process_features(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select models for processing panorama features.')
    parser.add_argument('--lmdb_path', type=str)
    parser.add_argument('--model_path', default='luodian/OTTER-Image-LLaMA7B-LA-InContext', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    args = parser.parse_args()

    build_feature_file(args)
