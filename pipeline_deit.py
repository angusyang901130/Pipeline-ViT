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
import pippy.fx
from pippy import run_pippy
from pippy.hf import PiPPyHFTracer, inject_pipeline_forward
from pippy.IR import annotate_split_points, Pipe, PipeSplitWrapper

from PIL import Image
import requests
from accelerate import Accelerator
import torch.distributed.rpc as rpc
import torch.profiler as profiler
import logging

# parallel-scp -r -A -h ~/hosts.txt ~/Pipeline-ViT/ ~/
# torchrun   --nnodes=4   --nproc-per-node=1   --node-rank=0   --master-addr=192.168.1.100   --master-port=50000   pipeline_deit.py

def run_serial(model, input_data, batch_size=1, num_iter=100):

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

    WARMUP = 1
    NUM_TEST = 10
    MINI_BATCH_SIZE = 1
    NUM_CHUNKS = 10
    NUM_INPUT = NUM_CHUNKS * MINI_BATCH_SIZE
    # INPUT_PER_ITER = 4

    torch.manual_seed(0)

    model = DeiTForImageClassification.from_pretrained(MODEL_NAME)
    # model = ViTForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
        
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
        model,
        except_keys=input_dict.keys(),
    )

    driver, stage_mod = pippy.all_compile(
        model,
        num_ranks=world_size,
        num_chunks=NUM_CHUNKS,
        # split_policy=split_policy,
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
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Original {params // 10 ** 6}M params", end='\n\n')
        # print(model)
        print(" Calculating Latency ".center(80, "*"))

        num_img = 100
        fps_list = []

        print("Running Serial...")
        with torch.no_grad():
            for i in tqdm(range(1, NUM_TEST+WARMUP+1)):
                
                start_time = time.perf_counter()
                run_serial(model=model, input_data=serial_input)
                end_time = time.perf_counter()
                
                if i <= WARMUP:
                    continue

                fps = num_img / (end_time-start_time)
                fps_list.append(fps)
                # latency_per_img = (end_time-start_time) / num_img


        # print('Latency per image without pipeline (input batch size = %d): %.2f ms'%(serial_input.shape[0], (end_time-start_time)*1000))
        print('Throughput without pipeline (input batch size = %d): %.4f fps'%(serial_input.shape[0], np.mean(fps_list)), end='\n\n')

        time.sleep(10)
        fps_list = []
        
        print("Running Pipeline...")
        with torch.no_grad():
            
            for i in tqdm(range(1, NUM_TEST+WARMUP+1)):
                start_time = time.perf_counter()
                run_pipeline(driver=driver, input_data=pipeline_input)
                end_time = time.perf_counter()

                if i <= WARMUP:
                    continue

                fps = num_img / (end_time-start_time)
                fps_list.append(fps)
                # latency_per_img = (end_time-start_time) / num_img

        # print('Latency per image with %d pipeline stages (mini batch size = %d): %.2f ms'%(world_size, MINI_BATCH_SIZE, latency_per_img*1000))
        print('Throughput with %d pipeline stages (mini batch size = %d): %.4f fps'%(world_size, MINI_BATCH_SIZE, np.mean(fps_list)), end='\n\n')
        


        # Run the original code and get the output for comparison


        # output = driver(**inputs)
        # reference_output = model(**inputs)

        # output = driver(input)
        # reference_output = model(input)
        
        # torch.testing.assert_close(output, reference_output)

        # print(" Pipeline parallel model ran successfully! ".center(80, "*"))

    # destroy_process_group()
    rpc.shutdown()
