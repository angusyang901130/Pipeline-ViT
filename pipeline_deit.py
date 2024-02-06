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
import threading

# parallel-scp -r -A -h ~/hosts.txt ~/Pipeline-ViT/ ~/
# torchrun   --nnodes=2   --nproc-per-node=1   --node-rank=0   --master-addr=192.168.1.102   --master-port=50000   pipeline_deit.py

def PrintThreadInfo(rank):

    print(f"**************** My Rank: {rank} ****************")
    print(f'RANK:{os.environ["RANK"]}')
    print(f'LOCAL_RANK:{os.environ["LOCAL_RANK"]}')
    print(f'WORLD_SIZE:{os.environ["WORLD_SIZE"]}')
    print(f'LOCAL_WORLD_SIZE:{os.environ["LOCAL_WORLD_SIZE"]}')
    print()

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

    WARMUP = 0
    NUM_TEST = 3
    NUM_IMGS = 200

    MINI_BATCH_SIZE = 1
    NUM_CHUNKS = 200

    SERIAL_BATCH_SIZE = MINI_BATCH_SIZE
    PIPELINE_BATCH_SIZE = NUM_CHUNKS * MINI_BATCH_SIZE
    # INPUT_PER_ITER = 4

    torch.manual_seed(0)

    model = DeiTForImageClassification.from_pretrained(MODEL_NAME)
    # model = ViTForImageClassification.from_pretrained(MODEL_NAME)

    # model = torch.compile(model)
    model.eval()
    # print(model)    
        
    import os
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    os.environ["TP_SOCKET_IFNAME"]="eth0" 
    os.environ["GLOO_SOCKET_IFNAME"]="eth0"
    os.environ["GLOO_TIMEOUT_SECONDS"] = "3600"

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

    lock.acquire()
    try:
        PrintThreadInfo(rank=rank)
    finally:
        lock.release()


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

        print(" Original module params ".center(80, "*"))
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)   
        print(f"Original module params: {params // 10 ** 6}M params")

        for i, sm in enumerate(pipe.split_gm.children()):
            params = sum(p.numel() for p in sm.parameters() if p.requires_grad)
            print(f"Pipeline Stage {i} params: {params // 10 ** 6}M params")


    from pippy.PipelineStage import PipelineStage
    stage = PipelineStage(pipe, rank, DEVICE)

    '''
    Running Serial
    '''

    fps_list = []

    print("Running Serial...")
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

    avg_fps = torch.tensor(np.mean(fps_list))

    dist.barrier()
    dist.reduce(avg_fps, dst=0, op=torch.distributed.ReduceOp.SUM)

    avg_fps /= world_size

    print('Throughput without pipeline (input batch size = %d): %.4f fps'%(SERIAL_BATCH_SIZE, avg_fps), end='\n\n')
    time.sleep(10)


    '''
    Wait for serial to be done
    '''
    dist.barrier()

    '''
    Running Pipeline
    '''

    fps_list = []
        
    print("Running Pipeline...")
    with torch.no_grad():

        for i in tqdm(range(1, NUM_TEST+WARMUP+1)):
            
            '''
            To be fair, all threads has to be on same point
            '''

            dist.barrier()

            start_time = torch.tensor(time.perf_counter())
            pipeline_output = RunPipeline(stage=stage, imgs=imgs, rank=rank, world_size=world_size)
            end_time = torch.tensor(time.perf_counter())

            # if rank == 0 or rank == world_size-1:
            #     print(f"Rank {rank} Start Time: {start_time.item()}")
            #     print(f"Rank {rank} End Time: {end_time.item()}")

            dist.barrier()

            dist.reduce(start_time, dst=world_size-1, op=torch.distributed.ReduceOp.MIN)
            dist.reduce(end_time, dst=world_size-1, op=torch.distributed.ReduceOp.MAX)

            # if rank == world_size-1:
            #     print(f"Reduced Start Time: {start_time.item()}")
            #     print(f"Reduced End Time: {end_time.item()}")

            if i <= WARMUP:
                continue

            if rank == world_size - 1:
                fps = NUM_IMGS / (end_time-start_time)
                fps_list.append(fps)


    if rank == world_size - 1:
        print('Throughput with %d pipeline stages (mini batch size = %d): %.4f fps'%(world_size, MINI_BATCH_SIZE, np.mean(fps_list)), end='\n\n')
        
        # torch.testing.assert_close(pipeline_output, reference_output)

        # print(" Pipeline parallel model ran successfully! ".center(80, "*"))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()