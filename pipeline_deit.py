# Copyright (c) Meta Platforms, Inc. and affiliates
# test
import torch
from typing import Any
import time
import numpy as np
from collections import defaultdict
from transformers import DeiTImageProcessor, DeiTForImageClassification
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

# parallel-scp -r -A -h ~/hosts.txt ~/Pipeline-ViT/ ~/
# torchrun   --nnodes=4   --nproc-per-node=1   --node-rank=0   --master-addr=192.168.1.100   --master-port=50000   pipeline_deit.py
# MODEL_NAME = "facebook/deit-small-distilled-patch16_224"
MODEL_NAME = "facebook/deit-tiny-distilled-patch16-224"

torch.manual_seed(0)

mn = DeiTForImageClassification.from_pretrained(MODEL_NAME)
mn.eval()
    
import os
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
os.environ["TP_SOCKET_IFNAME"]="eth0" 
os.environ["GLOO_SOCKET_IFNAME"]="eth0"

import torch.distributed.rpc as rpc
import torch.distributed as dist

rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size, backend=rpc.BackendType.TENSORPIPE)

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

num_ranks = world_size
split_policy = pippy.split_into_equal_size(num_ranks)
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
feature_extractor = DeiTImageProcessor.from_pretrained(MODEL_NAME)
inputs = feature_extractor(images=image, return_tensors="pt")
# image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
# inputs = image_processor(images=image, return_tensors="pt").pixel_values
# inputs = torch.randn(1, 3, 224, 224)
# print(inputs)
# print(max(inputs))
# print(min(inputs))
input_dict = {
    'pixel_values': inputs,
}
concrete_args = pippy.create_default_args(
    mn,
    except_keys=input_dict.keys(),
)

driver, stage_mod = pippy.all_compile(
    mn,
    num_ranks=num_ranks,
    num_chunks=1,
    split_policy=split_policy,
    tracer=PiPPyHFTracer(),
    concrete_args=concrete_args,
    index_filename=None,
    checkpoint_prefix=None,
)

print(" Pipeline parallel sub module params ".center(80, "*"))
params = sum(p.numel() for p in stage_mod.parameters() if p.requires_grad)
print(f"submod_{os.environ['RANK']} {params // 10 ** 6}M params", end='\n')

if rank == 0:

    print(" Original module params ".center(80, "*"))
    params = sum(p.numel() for p in mn.parameters() if p.requires_grad)
    print(f"Original {params // 10 ** 6}M params", end='\n')

    print(" Calculating Latency ".center(80, "*"))
    num_runs = 100
    timings = []
    with torch.no_grad():
        for i in range(1, num_runs+1):
            start_time = time.perf_counter()
            reference_output = mn(**inputs)
            end_time = time.perf_counter()
            timings.append(end_time - start_time)

            if i%(num_runs/5)==0:
                print('Iteration %d/%d, avg batch time %.2f ms'%(i, num_runs, np.mean(timings)*1000))

    print('Latency per query without pipeline: %.2f ms'%((np.mean(timings))*1000), end='\n')

    timings = []
    
    with torch.no_grad():
        for i in range(1, num_runs+1):
            start_time = time.perf_counter()
            output = driver(**inputs)
            end_time = time.perf_counter()
            timings.append(end_time - start_time)
            if i%(num_runs/5)==0:
                print('Iteration %d/%d, avg batch time %.2f ms'%(i, num_runs, np.mean(timings)*1000))

    print('Latency per query: %.2f ms'%((np.mean(timings))*1000), end='\n')
    

    # Run the original code and get the output for comparison


    output = driver(**inputs)
    reference_output = mn(**inputs)
    
    torch.testing.assert_close(output, reference_output)

    print(" Pipeline parallel model ran successfully! ".center(80, "*"))

# destroy_process_group()
rpc.shutdown()
