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

import torch.distributed.rpc as rpc
import torch.profiler as profiler

from PIL import Image
import requests
from accelerate import Accelerator
import logging
import threading

# parallel-scp -r -A -h ~/hosts.txt ~/Pipeline-ViT/ ~/
# torchrun   --nnodes=2   --nproc-per-node=1   --node-rank=0   --master-addr=192.168.1.102   --master-port=50000   pipeline_deit.py

def RunSerial(model, imgs):

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


def RunPipeline(stage, imgs, rank, world_size):

    # for i in tqdm(range(num_iter)):
    if rank == 0:
        stage(imgs)
    elif rank == world_size-1:
        output = stage()
    else:
        stage()
    
    if rank == world_size-1:
        return output
    else:
        return None
    

def main():

    lock = threading.Lock()

    MODEL_NAME = "facebook/deit-small-distilled-patch16-224"
    # MODEL_NAME = "facebook/deit-small-patch16-224"
    # MODEL_NAME = "facebook/deit-tiny-distilled-patch16-224"
    # MODEL_NAME = "facebook/deit-tiny-patch16-224"

    DEVICE = "cpu"

    WARMUP = 1
    NUM_TEST = 3
    NUM_IMGS = 200

    MINI_BATCH_SIZE = 1
    NUM_CHUNKS = NUM_IMGS

    SERIAL_BATCH_SIZE = MINI_BATCH_SIZE
    PIPELINE_BATCH_SIZE = NUM_CHUNKS * MINI_BATCH_SIZE
    # INPUT_PER_ITER = 4

    torch.manual_seed(0)

    model = DeiTForImageClassification.from_pretrained(MODEL_NAME)
    # model = ViTForImageClassification.from_pretrained(MODEL_NAME)

    model.eval()
    # print(model)   

    import os        
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    os.environ["TP_SOCKET_IFNAME"]="eth0" 
    os.environ["GLOO_SOCKET_IFNAME"]="eth0"
    os.environ["GLOO_TIMEOUT_SECONDS"] = "3600"
    os.environ["OMP_NUM_THREADS"] = "2"

    # print(f"OMP_NUM_THREAD = {os.environ.get('OMP_NUM_THREADS')}")

    import torch.distributed as dist

    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    # rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size, backend=rpc.BackendType.TENSORPIPE)
    # rpc.init_rpc(
    #     f"worker{rank}", 
    #     rank=rank, 
    #     world_size=world_size, 
    #     rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
    #         rpc_timeout=500
    #     )
    # )

    # print(f'Rank:{os.environ["RANK"]} / World_Size:{os.environ["WORLD_SIZE"]}\nLocal_Rank:{os.environ["LOCAL_RANK"]} / Local_World_Size:{os.environ["LOCAL_WORLD_SIZE"]}')

    # split_policy = pippy.split_into_equal_size(world_size)
    annotate_split_points(model, {'deit.encoder.layer.2': PipeSplitWrapper.SplitPoint.END})
    annotate_split_points(model, {'deit.encoder.layer.5': PipeSplitWrapper.SplitPoint.END})
    annotate_split_points(model, {'deit.encoder.layer.8': PipeSplitWrapper.SplitPoint.END})

    # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    # image = Image.open(requests.get(url, stream=True).raw)
    # feature_extractor = DeiTImageProcessor.from_pretrained(MODEL_NAME)
    # input = feature_extractor(images=image, return_tensors="pt")
    # image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    # inputs = image_processor(images=image, return_tensors="pt").pixel_values

    imgs = torch.randn(NUM_IMGS, 3, 224, 224)

    pipe = Pipe.from_tracing(model, NUM_CHUNKS, example_args=(imgs, ))
    # print(pipe)

    nstages = len(list(pipe.split_gm.children()))
    if rank == 0:

        print(" Module params Info ".center(80, "*"))
        params_in_M = sum(p.numel() for p in model.parameters() if p.requires_grad) / 10**6
        print(f'Original module params: {params_in_M:.4f}M params')

        for i, sm in enumerate(pipe.split_gm.children()):
            params_in_M = sum(p.numel() for p in sm.parameters() if p.requires_grad) / 10**6
            print(f'Pipeline Stage {i} params: {params_in_M:.4f}M params')
    
        print("".center(100, "*"))

    '''
    Running Serial
    '''

    fps_list = []
    serial_fps = 0
    tensor_serial_fps = torch.tensor(0)

    if local_rank == 0:
        print(f"Running Serial on Rank {rank} ...")
        with torch.no_grad():
            for i in tqdm(range(1, NUM_TEST+WARMUP+1)):
                
                tmp_imgs = torch.unsqueeze(imgs, dim=1)
                start_time = time.perf_counter()
                reference_output = RunSerial(model=model, imgs=tmp_imgs)
                end_time = time.perf_counter()
                
                if i <= WARMUP:
                    continue

                fps = NUM_IMGS / (end_time-start_time)
                fps_list.append(fps)
            
        time.sleep(10)

        tensor_serial_fps = torch.tensor(np.mean(fps_list))
        # print(f'tensor_serial_fps = {tensor_serial_fps} on Rank {rank}')

    dist.reduce(tensor_serial_fps, dst=0, op=dist.ReduceOp.SUM)
        
    if rank == 0:

        num_machine = world_size / local_world_size
        serial_fps = tensor_serial_fps.item() / num_machine

        print('Throughput without pipeline (input batch size = %d): %.4f fps'%(SERIAL_BATCH_SIZE, serial_fps), end='\n\n')


    '''
    Wait for serial to be done
    '''
    dist.barrier()
    # os.environ["OMP_NUM_THREADS"] = "2"

    from pippy.PipelineStage import PipelineStage
    stage = PipelineStage(pipe, rank, DEVICE)

    '''
    Running Pipeline
    '''
    fps_list = []
    pipeline_fps = 0
        
    # print(f"Running Pipeline on Rank {rank} ...")
    with torch.no_grad():
        
    # if local_rank == 0:
        for i in tqdm(range(1, NUM_TEST+WARMUP+1)):               
            '''
            To be fair, all threads has to be on same point
            '''

            dist.barrier()

            tensor_start_time = torch.tensor(time.perf_counter())
            pipeline_output = RunPipeline(stage=stage, imgs=imgs, rank=rank, world_size=world_size)
            tensor_end_time = torch.tensor(time.perf_counter())


            dist.barrier()

            dist.reduce(tensor_start_time, dst=0, op=dist.ReduceOp.MIN)
            dist.reduce(tensor_end_time, dst=0, op=dist.ReduceOp.MAX)

            if i <= WARMUP:
                continue

            if rank == 0:
                start_time = tensor_start_time.item()
                end_time = tensor_end_time.item()

                fps = NUM_IMGS / (end_time-start_time)
                fps_list.append(fps)

        # else:
        #     for i in range(1, NUM_TEST+WARMUP+1):             
        #         '''
        #         To be fair, all threads has to be on same point
        #         '''

        #         dist.barrier()

        #         tensor_start_time = torch.tensor(time.perf_counter())
        #         pipeline_output = RunPipeline(stage=stage, imgs=imgs, rank=rank, world_size=world_size)
        #         tensor_end_time = torch.tensor(time.perf_counter())

        #         dist.barrier()

        #         dist.reduce(tensor_start_time, dst=0, op=dist.ReduceOp.MIN)
        #         dist.reduce(tensor_end_time, dst=0, op=dist.ReduceOp.MAX)

        #         if i <= WARMUP:
        #             continue

        #         if rank == 0:
        #             start_time = tensor_start_time.item()
        #             end_time = tensor_end_time.item()

        #             fps = NUM_IMGS / (end_time-start_time)
        #             fps_list.append(fps)

        # if rank == world_size-1:
        #     dist.send(pipeline_output, dst=0)
        # elif rank == 0:
        #     pipeline_output = torch.zeros(reference_output.shape)
        #     dist.recv(pipeline_output, src=world_size-1)


    if rank == 0:
        pipeline_fps = np.mean(fps_list)
        print('Throughput with %d pipeline stages (mini batch size = %d): %.4f fps'%(world_size, MINI_BATCH_SIZE, pipeline_fps))

        speedup = pipeline_fps / serial_fps
        print(f'The speedup of pipeline is {speedup:.4f}x', end='\n\n')
        
        torch.testing.assert_close(pipeline_output, reference_output)
        print(" Pipeline parallel model ran successfully! ".center(80, "*"))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()