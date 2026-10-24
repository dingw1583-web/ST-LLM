import argparse
import json
import platform
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--cuda',required=True)
    parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--warmup',type=int,default=20)
    parser.add_argument('--repeats',type=int,default=100)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.batch_size<1 or args.warmup<1 or args.repeats<1:
        parser.error('batch-size, warmup and repeats must be positive')
    import torch
    import dgl
    import transformers
    from dgl.dataloading import GraphDataLoader
    from config import get_config
    from dataloader import MixTrafficFlowDataset4DGL
    from evaluation import load_model
    from utils import get_device,mix_collate_fn
    cfg=get_config(args.dataset); device=get_device(index=args.cuda)
    if device.type!='cuda':
        raise ValueError('This benchmark requires CUDA')
    model,metadata=load_model(cfg,args.checkpoint,device)
    ds=MixTrafficFlowDataset4DGL(cfg.TEST_GRAPH_DATA,cfg.HEADER_TEST_DATA)
    if ds.flow_length != cfg.FLOW_PAD_TRUNC_LENGTH:
        raise ValueError("Prepared packet count differs from the checkpoint configuration")
    loader=GraphDataLoader(ds,batch_size=args.batch_size,shuffle=False,collate_fn=mix_collate_fn,num_workers=0)
    def batches():
        while True:
            yield from loader
    iterator=batches(); elapsed=[]; samples=0
    with torch.inference_mode():
        for _ in range(args.warmup):
            header,payload,_=next(iterator)
            header,payload=header.to(device),payload.to(device)
            model(header,payload)
        torch.cuda.synchronize(device)
        baseline_allocated=torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.repeats):
            header,payload,_=next(iterator)
            header,payload=header.to(device),payload.to(device)
            torch.cuda.synchronize(device)
            start=time.perf_counter()
            output=model(header,payload)
            torch.cuda.synchronize(device)
            elapsed.append(time.perf_counter()-start)
            samples+=header.shape[0]
            del output
    import numpy as np
    result=dict(dataset=args.dataset,checkpoint=args.checkpoint,variant=metadata['settings']['variant'],
                sample_definition='One preprocessed traffic-flow sample',timing_scope='Model forward only; excludes disk loading, preprocessing and host-to-device transfer',
                requested_batch_size=args.batch_size,warmup_batches=args.warmup,measured_batches=args.repeats,
                measured_samples=samples,total_forward_seconds=sum(elapsed),
                batch_latency_ms_mean=float(np.mean(elapsed)*1000),batch_latency_ms_p50=float(np.percentile(elapsed,50)*1000),
                batch_latency_ms_p95=float(np.percentile(elapsed,95)*1000),throughput_samples_s=samples/sum(elapsed),
                peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20,
                warmup_allocated_mib=baseline_allocated/2**20,
                total_parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                gpu=torch.cuda.get_device_name(device),gpu_total_memory_mib=torch.cuda.get_device_properties(device).total_memory/2**20,
                torch=torch.__version__,dgl=dgl.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,python=platform.python_version())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
