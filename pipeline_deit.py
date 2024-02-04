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

<<<<<<< HEAD
def run_serial(model, imgs):

    result = None
=======
def run_serial(model, input_data, num_iter=100):
>>>>>>> 0541214cc9144474f693e3e9dfccbae99fc6573e

    # for i in tqdm(range(num_iter)):
    for img in imgs:

        if result == None:
            output = model(img)
            result = output.logits
        else:
            output = model(img)
            result = torch.cat((result, output.logits), dim=0)

    return result

def run_pipeline(stage, imgs, rank, world_size):

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

    MODEL_NAME = "facebook/deit-small-distilled-patch16-224"
    # MODEL_NAME = "facebook/deit-small-patch16-224"
    # MODEL_NAME = "facebook/deit-tiny-distilled-patch16-224"
    # MODEL_NAME = "facebook/deit-tiny-patch16-224"
<<<<<<< HEAD

    DEVICE = "cpu"

    WARMUP = 0
    NUM_TEST = 1
    NUM_IMGS = 100

    MINI_BATCH_SIZE = 1
    NUM_CHUNKS = 100

    SERIAL_BATCH_SIZE = MINI_BATCH_SIZE
    PIPELINE_BATCH_SIZE = NUM_CHUNKS * MINI_BATCH_SIZE
=======
    num_img = 100
    WARMUP = 1
    NUM_TEST = 10
    MINI_BATCH_SIZE = 1
    NUM_CHUNKS = 100
    NUM_INPUT = NUM_CHUNKS * MINI_BATCH_SIZE
>>>>>>> 0541214cc9144474f693e3e9dfccbae99fc6573e
    # INPUT_PER_ITER = 4

    torch.manual_seed(0)

    model = DeiTForImageClassification.from_pretrained(MODEL_NAME)
    # model = ViTForImageClassification.from_pretrained(MODEL_NAME)

    # model = torch.compile(model)
    model.eval()
        
    import os
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    os.environ["TP_SOCKET_IFNAME"]="eth0" 
    os.environ["GLOO_SOCKET_IFNAME"]="eth0"

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

    print(f"**************** My Rank: {rank} ****************")
    print(f'RANK:{os.environ["RANK"]}')
    print(f'LOCAL_RANK:{os.environ["LOCAL_RANK"]}')
    print(f'WORLD_SIZE:{os.environ["WORLD_SIZE"]}')
    print(f'LOCAL_WORLD_SIZE:{os.environ["LOCAL_WORLD_SIZE"]}')
    print()

    # from pippy.microbatch import TensorChunkSpec

    # args_chunk_spec: Any = (TensorChunkSpec(0),)
    # kwargs_chunk_spec: Any = {}
    # output_chunk_spec: Any = TensorChunkSpec(0)

    # split_policy = pippy.split_into_equal_size(world_size)
    annotate_split_points(model, {'deit.encoder.layer.5': PipeSplitWrapper.SplitPoint.END})

    # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    # image = Image.open(requests.get(url, stream=True).raw)
    # feature_extractor = DeiTImageProcessor.from_pretrained(MODEL_NAME)
    # input = feature_extractor(images=image, return_tensors="pt")
    # image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    # inputs = image_processor(images=image, return_tensors="pt").pixel_values

    # serial_input = torch.randn(SERIAL_BATCH_SIZE, 3, 224, 224)
    # pipeline_input = torch.randn(PIPELINE_BATCH_SIZE, 3, 224, 224)
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

    if rank == world_size-1:

        print("Running Serial...")
        with torch.no_grad():
            for i in tqdm(range(1, NUM_TEST+WARMUP+1)):
                
                tmp_imgs = torch.unsqueeze(imgs, dim=1)
                start_time = time.perf_counter()
                reference_output = run_serial(model=model, imgs=tmp_imgs)
                end_time = time.perf_counter()
                
        #         if i <= WARMUP:
        #             continue

                fps = NUM_IMGS / (end_time-start_time)
                fps_list.append(fps)
                # latency_per_img = (end_time-start_time) / NUM_IMGS


        print('Throughput without pipeline (input batch size = %d): %.4f fps'%(SERIAL_BATCH_SIZE, np.mean(fps_list)), end='\n\n')


    dist.barrier()
    time.sleep(10)

    '''
    Running Pipeline
    '''

    fps_list = []
        
    print("Running Pipeline...")
    with torch.no_grad():

        for i in tqdm(range(1, NUM_TEST+WARMUP+1)):

            start_time = time.perf_counter()
            pipeline_output = run_pipeline(stage=stage, imgs=imgs, rank=rank, world_size=world_size)
            end_time = time.perf_counter()

            if rank == 0 or rank == world_size-1:
                print(f"Rank {rank} Start Time: {start_time}")
                print(f"Rank {rank} End Time: {end_time}")

            if i <= WARMUP:
                continue

            if rank == world_size - 1:
                fps = NUM_IMGS / (end_time-start_time)
                fps_list.append(fps)
            # latency_per_img = (end_time-start_time) / NUM_IMGS

    if rank == world_size - 1:
        print('Throughput with %d pipeline stages (mini batch size = %d): %.4f fps'%(world_size, MINI_BATCH_SIZE, np.mean(fps_list)), end='\n\n')
        
        torch.testing.assert_close(pipeline_output, reference_output)

        print(" Pipeline parallel model ran successfully! ".center(80, "*"))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()