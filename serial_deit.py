# Copyright (c) Meta Platforms, Inc. and affiliates
# test
import torch
from typing import Any
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import DeiTImageProcessor, DeiTForImageClassification, ViTForImageClassification
import pippy
from pippy.IR import annotate_split_points, Pipe, PipeSplitWrapper

from PIL import Image
import requests
from accelerate import Accelerator
import torch.distributed.rpc as rpc
import torch.profiler as profiler
import logging

import argparse

def run_serial(model, imgs):

    result = None

    # for i in tqdm(range(num_iter)):
    for img in imgs:
        
        if result == None:
            output = model(img)
            result = output.logits
        else:
            output = model(img)
            result = torch.cat((result, output.logits), dim=0)

    return result
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk_size', type=int, default=1)
    parser.add_argument('--num_threads', type=int, default=4)
    parser.add_argument('--num_interop_threads', type=int, default=4)
    args = parser.parse_args()


    MODEL_NAME = "facebook/deit-small-distilled-patch16-224"
    # MODEL_NAME = "facebook/deit-small-patch16-224"
    # MODEL_NAME = "facebook/deit-tiny-distilled-patch16-224"
    # MODEL_NAME = "facebook/deit-tiny-patch16-224"

    DEVICE = "cpu"

    WARMUP = 0
    NUM_TEST = 3
    # NUM_IMGS = 200

    # MINI_BATCH_SIZE = 2
    MINI_BATCH_SIZE = args.chunk_size
    NUM_CHUNKS = 200

    SERIAL_BATCH_SIZE = MINI_BATCH_SIZE
    # PIPELINE_BATCH_SIZE = NUM_CHUNKS * MINI_BATCH_SIZE
    NUM_IMGS = NUM_CHUNKS * MINI_BATCH_SIZE
    # INPUT_PER_ITER = 4

    torch.manual_seed(0)

    model = DeiTForImageClassification.from_pretrained(MODEL_NAME)
    # model = ViTForImageClassification.from_pretrained(MODEL_NAME)

    # model = torch.compile(model)
    model.eval()
        
    serial_input = torch.randn(NUM_CHUNKS, SERIAL_BATCH_SIZE, 3, 224, 224)

    '''
    Running Serial
    '''
    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(args.num_interop_threads)
    print(f'intra op threads num: {torch.get_num_threads()} | inter op threads num: {torch.get_num_interop_threads()}')
    # print(f'inter op threads num: {torch.get_num_interop_threads()}')
    fps_list = []
    print("Running Serial...")
    with torch.no_grad():
        for i in tqdm(range(1, NUM_TEST+WARMUP+1)):
            
            # tmp_imgs = torch.unsqueeze(imgs, dim=1)

            start_time = time.perf_counter()
            reference_output = run_serial(model=model, imgs=serial_input)
            end_time = time.perf_counter()
            
            if i <= WARMUP:
                continue

            fps = NUM_IMGS / (end_time-start_time)
            fps_list.append(fps)


    print('Throughput without pipeline (input batch size = %d): %.4f fps'%(SERIAL_BATCH_SIZE, np.mean(fps_list)), end='\n\n')

if __name__ == "__main__":
    main()