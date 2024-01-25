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
from pippy.PipelineDriver import PipelineDriverFillDrain
from pippy.microbatch import TensorChunkSpec
from pippy.IR import annotate_split_points, Pipe, PipeSplitWrapper

from PIL import Image
import requests
from accelerate import Accelerator
import torch.distributed.rpc as rpc
import torch.profiler as profiler
import logging

import os
import torch.distributed.rpc as rpc
import torch.distributed as dist
import timm

from torch.quantization import quantize_dynamic
from torch.quantization import quantize_fx


# parallel-scp -r -A -h ~/hosts.txt ~/Pipeline-ViT/ ~/
# torchrun   --nnodes=4   --nproc-per-node=1   --node-rank=0   --master-addr=192.168.1.100   --master-port=50000   pipeline_deit.py

# Define a function to quantize the model
def quantize_model(model):

    backend = "qnnpack"
    model.eval()
    # model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    model_prepared = torch.quantization.prepare(model, inplace=False)
    calibration_data = torch.randn(1, 3, 224, 224)
    model_calibrated = torch.quantization.calibrate(model_prepared, calibration_data)
    model = convert(model_calibrated, inplace=False)
    # model_prepared = torch.quantization.prepare(model)

    with torch.inference_mode():
        for _ in range(10):
            x = torch.rand(1, 3, 224, 224)
            model_prepared(x)

    # model_quantized = quantize_dynamic(model)
    model_quantized = quantize_fx.convert_fx(model_prepared)

    return model_quantized

def quantize_proportion(model):
    fp32_param = 0.0
    total = 0.0
    q8_param = 0.0

    # print(model.state_dict())

    for name, param in model.state_dict().items():

        if isinstance(param, torch.Tensor):
            total += torch.numel(param)

            if(param.dtype == torch.float32):
                fp32_param += torch.numel(param)

            if(param.dtype == torch.qint8):
                q8_param += torch.numel(param)

        elif isinstance(param, torch.dtype):
            # print(param)
            pass

        elif isinstance(param, tuple):
            for sub_param in param:
                if isinstance(sub_param, torch.Tensor):
                    total += torch.numel(sub_param)

                    if sub_param.dtype == torch.float32:
                        fp32_param += torch.numel(sub_param)

                    elif sub_param.dtype == torch.qint8:
                        q8_param += torch.numel(sub_param)

    # print(fp32_param, total, q8_param, q8_param / total)

    return q8_param / total


if __name__ == "__main__":

    MODEL_NAME = "facebook/deit-small-distilled-patch16-224"
    # MODEL_NAME = "facebook/deit-small-patch16-224"
    # MODEL_NAME = "facebook/deit-tiny-distilled-patch16-224"
    # MODEL_NAME = "facebook/deit-tiny-patch16-224"

    # MODEL_NAME = 'deit_small_patch16_224'

    WARMUP = 3
    NUM_RUN = 20
    CHUNK_SIZE = 8
    NUM_CHUNKS = 6
    NUM_INPUT = NUM_CHUNKS * CHUNK_SIZE
    # INPUT_PER_ITER = 4

    torch.manual_seed(0)

    # mn = DeiTForImageClassification.from_pretrained(MODEL_NAME)
    # mn = ViTForImageClassification.from_pretrained(MODEL_NAME)
    # mn.eval()

    # model = timm.create_model(MODEL_NAME, pretrained=True)
    model = DeiTForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    os.environ["TP_SOCKET_IFNAME"]="eth0" 
    os.environ["GLOO_SOCKET_IFNAME"]="eth0"

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
        model,
        except_keys=input_dict.keys(),
    )

    model_qat = quantize_model(model)
    print("Quantize Proportion:", quantize_proportion(model_qat))
    # print(model_quat)

    driver, stage_mod = pippy.all_compile(
        model_qat,
        num_ranks=world_size,
        num_chunks=NUM_CHUNKS,
        #split_policy=split_policy,
        tracer=PiPPyHFTracer(),
        concrete_args=concrete_args,
        index_filename=None,
        checkpoint_prefix=None,
        # args_chunk_spec=args_chunk_spec,
        # kwargs_chunk_spec=kwargs_chunk_spec,
        # output_chunk_spec=output_chunk_spec
    )

    print(" Pipeline parallel sub module params ".center(80, "*"))
    params = sum(p.numel() for p in stage_mod.parameters())
    print(f"submod_{os.environ['RANK']} {params // 10 ** 6}M params", end='\n\n')
    # stage_mod.print_readable()

    if rank == 0:

        print(" Original module params ".center(80, "*"))
        params = sum(p.numel() for p in model_qat.parameters())
        print(f"Original {params // 10 ** 6}M params", end='\n\n')
        # print(mn)
        print(" Calculating Latency ".center(80, "*"))

        timings = []
        with torch.no_grad():
            for i in range(1, NUM_RUN+WARMUP+1):
                
                start_time = time.perf_counter()
                # reference_output = mn(**inputs)
                reference_output = model_qat(serial_input)
                # reference_output = mn(pipeline_input[i-WARMUP-1])
                end_time = time.perf_counter()
                
                if i <= WARMUP:
                    continue

                timings.append(end_time - start_time)

                if (i - WARMUP) % (NUM_RUN / 5) == 0:
                    print('Iteration %d/%d, avg time per image %.2f ms'%(i-WARMUP, NUM_RUN, np.mean(timings)*1000))

        print('Latency per image without pipeline (input batch size = 1): %.2f ms'%((np.mean(timings))*1000), end='\n\n')

        # with torch.no_grad():
        #     start_time = time.perf_counter()
        #     reference_output = mn(pipeline_input)
        #     end_time = time.perf_counter()

        #     print('Latency per image without pipeline (input batch size = NUM_INPUT): %.2f ms'%((end_time - start_time) / NUM_INPUT * 1000), end='\n\n')

        # time.sleep(10)
        timings = []
        
        # with torch.no_grad():

        #     # for i in tqdm(range(1, NUM_RUN+WARMUP+1)):
        #     for i in range(1, NUM_RUN+WARMUP+1):

        #         start_time = time.perf_counter()
        #         # output = driver(**inputs)
        #         output = driver(pipeline_input)
        #         end_time = time.perf_counter()

        #         if i <= WARMUP:
        #             continue

        #         timings.append((end_time - start_time) / NUM_INPUT)

        #         if (i - WARMUP) % (NUM_RUN / 4) == 0:
        #             print('Iteration %d/%d, avg time per image %.2f ms'%(i-WARMUP, NUM_RUN, np.mean(timings)*1000))

        # print('Latency per image with %d pipeline stages: %.2f ms'%(world_size, (np.mean(timings))*1000), end='\n\n')
    


        # Run the original code and get the output for comparison


        # output = driver(**inputs)
        # reference_output = mn(**inputs)

        # output = driver(input)
        # reference_output = mn(input)
        
        # torch.testing.assert_close(output, reference_output)

        # print(" Pipeline parallel model ran successfully! ".center(80, "*"))

    # destroy_process_group()
    rpc.shutdown()
