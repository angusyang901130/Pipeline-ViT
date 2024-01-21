# Copyright (c) Meta Platforms, Inc. and affiliates
# test
import torch
from typing import Any
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import DeiTImageProcessor, DeiTForImageClassification, ViTForImageClassification
from pippy.IR import Pipe
import pippy
import pippy.fx
from pippy import run_pippy
from pippy.hf import PiPPyHFTracer, inject_pipeline_forward

from PIL import Image
import requests
from accelerate import Accelerator
import torch.distributed.rpc as rpc
import torch.profiler as profiler
import logging

# parallel-scp -r -A -h ~/hosts.txt ~/Pipeline-ViT/ ~/
# torchrun   --nnodes=4   --nproc-per-node=1   --node-rank=0   --master-addr=192.168.1.100   --master-port=50000   pipeline_deit.py
MODEL_NAME = "facebook/deit-small-distilled-patch16-224"
# MODEL_NAME = "facebook/deit-small-patch16-224"
# MODEL_NAME = "facebook/deit-tiny-distilled-patch16-224"
# MODEL_NAME = "facebook/deit-tiny-patch16-224"

WARMUP = 5
NUM_RUN = 50
CHUNK_SIZE = 8
NUM_CHUNKS = 6
NUM_INPUT = NUM_CHUNKS * CHUNK_SIZE
# INPUT_PER_ITER = 4

torch.manual_seed(0)

mn = DeiTForImageClassification.from_pretrained(MODEL_NAME)
# mn = ViTForImageClassification.from_pretrained(MODEL_NAME)
mn.eval()
    
import os
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
os.environ["TP_SOCKET_IFNAME"]="eth0" 
os.environ["GLOO_SOCKET_IFNAME"]="eth0"

import torch.distributed.rpc as rpc
import torch.distributed as dist

rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size, backend=rpc.BackendType.TENSORPIPE)
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

from pippy.PipelineDriver import PipelineDriverFillDrain
from pippy.microbatch import TensorChunkSpec

args_chunk_spec: Any = (TensorChunkSpec(0),)
kwargs_chunk_spec: Any = {}
output_chunk_spec: Any = TensorChunkSpec(0)

split_policy = pippy.split_into_equal_size(world_size)
# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
# feature_extractor = DeiTImageProcessor.from_pretrained(MODEL_NAME)
# input = feature_extractor(images=image, return_tensors="pt")
# image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
# inputs = image_processor(images=image, return_tensors="pt").pixel_values
serial_input = torch.randn(1, 3, 224, 224)
pipeline_input = torch.randn(NUM_INPUT, 3, 224, 224)
warmup_input = torch.randn(world_size, 3, 224, 224)

# print(inputs)
# print(max(inputs))
# print(min(inputs))
# input_dict = {
#     'pixel_values': inputs,
# }

input_dict = {
    'pixel_values': input,
}

concrete_args = pippy.create_default_args(
    mn,
    except_keys=input_dict.keys(),
)

driver, stage_mod = pippy.all_compile(
    mn,
    num_ranks=world_size,
    num_chunks=NUM_CHUNKS,
    split_policy=split_policy,
    tracer=PiPPyHFTracer(),
    concrete_args=concrete_args,
    index_filename=None,
    checkpoint_prefix=None,
    # args_chunk_spec=args_chunk_spec,
    # kwargs_chunk_spec=kwargs_chunk_spec,
    # output_chunk_spec=output_chunk_spec
)

print(" Pipeline parallel sub module params ".center(80, "*"))
params = sum(p.numel() for p in stage_mod.parameters() if p.requires_grad)
print(f"submod_{os.environ['RANK']} {params // 10 ** 6}M params", end='\n\n')
# stage_mod.print_readable()

if rank == 0:

    print(" Original module params ".center(80, "*"))
    params = sum(p.numel() for p in mn.parameters() if p.requires_grad)
    print(f"Original {params // 10 ** 6}M params", end='\n\n')
    # print(mn)
    print(" Calculating Latency ".center(80, "*"))

    # timings = []
    # with torch.no_grad():
    #     for i in range(1, NUM_RUN+WARMUP+1):
               
    #         start_time = time.perf_counter()
    #         # reference_output = mn(**inputs)
    #         reference_output = mn(serial_input)
    #         # reference_output = mn(pipeline_input[i-WARMUP-1])
    #         end_time = time.perf_counter()
            
    #         if i <= WARMUP:
    #             continue

    #         timings.append(end_time - start_time)

    #         if (i - WARMUP) % (NUM_RUN / 5) == 0:
    #             print('Iteration %d/%d, avg time per image %.2f ms'%(i-WARMUP, NUM_RUN, np.mean(timings)*1000))

    # print('Latency per image without pipeline (input batch size = 1): %.2f ms'%((np.mean(timings))*1000), end='\n\n')

    # with torch.no_grad():
    #     start_time = time.perf_counter()
    #     reference_output = mn(pipeline_input)
    #     end_time = time.perf_counter()

    #     print('Latency per image without pipeline (input batch size = NUM_INPUT): %.2f ms'%((end_time - start_time) / NUM_INPUT * 1000), end='\n\n')

    # time.sleep(10)
    timings = []
    
    with torch.no_grad():

        # for i in tqdm(range(1, NUM_RUN+WARMUP+1)):
        for i in range(1, NUM_RUN+WARMUP+1):

            start_time = time.perf_counter()
            # output = driver(**inputs)
            output = driver(pipeline_input)
            end_time = time.perf_counter()

            if i <= WARMUP:
                continue

            timings.append((end_time - start_time) / NUM_INPUT)

            if (i - WARMUP) % (NUM_RUN / 5) == 0:
                print('Iteration %d/%d, avg time per image %.2f ms'%(i-WARMUP, NUM_RUN, np.mean(timings)*1000))

    print('Latency per image with %d pipeline stages: %.2f ms'%(world_size, (np.mean(timings))*1000), end='\n\n')
    


    # Run the original code and get the output for comparison


    # output = driver(**inputs)
    # reference_output = mn(**inputs)

    # output = driver(input)
    # reference_output = mn(input)
    
    # torch.testing.assert_close(output, reference_output)

    # print(" Pipeline parallel model ran successfully! ".center(80, "*"))

# destroy_process_group()
rpc.shutdown()
