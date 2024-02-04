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

# parallel-scp -r -A -h ~/hosts.txt ~/Pipeline-ViT/ ~/
# torchrun   --nnodes=2   --nproc-per-node=1   --node-rank=0   --master-addr=192.168.1.102   --master-port=50000   pipeline_deit.py

def run_serial(model, input_data, num_iter=100):

    # for i in tqdm(range(num_iter)):
    for i in range(num_iter):
        output = model(input_data)

def run_pipeline(driver, input_data, num_iter=10):

    # for i in tqdm(range(num_iter)):
    for i in range(num_iter):
        output = driver(input_data)


if __name__ == "__main__":

    MODEL_NAME = "facebook/deit-small-distilled-patch16-224"
    # MODEL_NAME = "facebook/deit-small-patch16-224"
    # MODEL_NAME = "facebook/deit-tiny-distilled-patch16-224"
    # MODEL_NAME = "facebook/deit-tiny-patch16-224"
    num_img = 100
    WARMUP = 1
    NUM_TEST = 10
    MINI_BATCH_SIZE = 1
    NUM_CHUNKS = 150
    NUM_INPUT = NUM_CHUNKS * MINI_BATCH_SIZE
    # INPUT_PER_ITER = 4

    torch.manual_seed(0)

    model = DeiTForImageClassification.from_pretrained(MODEL_NAME)
    # model = ViTForImageClassification.from_pretrained(MODEL_NAME)
    # model.eval()
        
    import os
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    os.environ["TP_SOCKET_IFNAME"]="eth0" 
    os.environ["GLOO_SOCKET_IFNAME"]="eth0"

    # import torch.distributed.rpc as rpc
    import torch.distributed as dist

    # Initialize distributed environment
    import torch.distributed as dist
    dist.init_process_group(rank=rank, world_size=world_size)

    # rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size, backend=rpc.BackendType.TENSORPIPE)
    # rpc.init_rpc(
    #     f"worker{rank}", 
    #     rank=rank, 
    #     world_size=world_size, 
    #     rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
    #         rpc_timeout=500
    #     )
    # )

    print(f"**************** My Rank: {rank} ****************")
    print(f'RANK:{os.environ["RANK"]}')
    print(f'LOCAL_RANK:{os.environ["LOCAL_RANK"]}')
    print(f'WORLD_SIZE:{os.environ["WORLD_SIZE"]}')
    print(f'LOCAL_WORLD_SIZE:{os.environ["LOCAL_WORLD_SIZE"]}')



    # split_policy = pippy.split_into_equal_size(world_size)
    annotate_split_points(model, {'deit.encoder.layer.5': PipeSplitWrapper.SplitPoint.END})

    # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    # image = Image.open(requests.get(url, stream=True).raw)
    # feature_extractor = DeiTImageProcessor.from_pretrained(MODEL_NAME)
    # input = feature_extractor(images=image, return_tensors="pt")
    # image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    # inputs = image_processor(images=image, return_tensors="pt").pixel_values
    serial_input = torch.randn(1, 3, 224, 224)
    pipeline_input = torch.randn(NUM_INPUT, 3, 224, 224)

    pipe = Pipe.from_tracing(model, NUM_CHUNKS, example_args=(pipeline_input,))
    # print(pipe)
    from pippy.PipelineStage import PipelineStage
    device = 'cpu'
    stage = PipelineStage(pipe, rank, device)
    # print(stage)
    print(" Original module params ".center(80, "*"))
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(" Pipeline parallel sub module params ".center(80, "*"))
    params = sum(p.numel() for p in stage.submod.parameters() if p.requires_grad)
    print(f"submod_{os.environ['RANK']} {params // 10 ** 6}M params", end='\n\n')


    if rank == 0:
        stage(pipeline_input)
    elif rank == world_size - 1:
        pipeline_output = stage()
        # with torch.no_grad():
        #     # model_compiled = torch.compile(model)
        #     # model_compiled.eval()
        #     model.eval()
        #     serial_output = model(pipeline_input)
        #     serial_output = torch.tensor(serial_output.logits)
        #     are_outputs_same = torch.testing.assert_close(pipeline_output, serial_output)
        print("success")
    else:
        stage()
    print('done')
