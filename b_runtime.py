import argparse
import os
import time
import numpy as np
from PIL import Image
import requests
import torch
from torch import nn
from torch.distributed import rpc
from torch.distributed.rpc import RRef
from transformers import BertTokenizer, ViTFeatureExtractor
from transformer import BertTransformerShard, ViTTransformerShard

#########################################################
#             Stitch Shards into one Module             #
#########################################################
class DistTransformer(nn.Module):
    def __init__(self, model_name, model_file, world_size, num_split):
        super().__init__()
        self.world_size = world_size
        self.num_split = num_split
        self.rref_list = []
        self.model_name = model_name
        for i in range(world_size):
            # Build Transformer Shard
            is_first = i == 0
            is_last = i == world_size-1
            if model_name in ['bert-base-uncased', 'bert-large-uncased']:
                rref = rpc.remote(f"worker{i}", BertTransformerShard, args=(i, model_name, model_file, is_first, is_last, partition[2*i], partition[2*i+1], True))
            else:
                rref = rpc.remote(f"worker{i}", ViTTransformerShard, args=(i, model_name, model_file, is_first, is_last, partition[2*i], partition[2*i+1], True))
            self.rref_list.append(rref)

    def forward(self, xs):
        out_futures = []
        for x in iter(xs.split(self.num_split, dim=0)):
            if self.model_name not in ['bert-base-uncased', 'bert-large-uncased']:
                skip = torch.zeros(x.size())
                x = torch.stack((x, skip), 0)
            x_rref = RRef(x)
            for i in range(self.world_size-1):
                x_rref = self.rref_list[i].remote().__call__(x_rref)
            y_rref = self.rref_list[self.world_size-1].rpc_async().__call__(x_rref)
            out_futures.append(y_rref)
        # res = torch.cat(torch.futures.wait_all(out_futures))
        # res = x_rref.to_here()
        # del out_futures
        # gc.collect()
        # return torch.cat(torch.futures.wait_all(out_futures))
        return torch.cat(torch.futures.wait_all(out_futures))


#########################################################
#                   Run RPC Processes                   #
#########################################################

def run_master(model_name, model_file, world_size, split_size, batch_size):
    print("Run mastering \n")
    latencies = []
    throughputs = []
    bert_inputs = np.load("bert_input.npz")['input']
    ## for verification
    # origin_model = ViTForImageClassification.from_pretrained(model_name)
    for si in range(len(split_size)):
        # print(f"Start Calcluate split size {split_size[si]}")
        model =  DistTransformer(model_name, model_file, world_size, split_size[si])
        if model_name in ['bert-base-uncased', 'bert-large-uncased']:
            tokenizer = BertTokenizer.from_pretrained(model_name)
            inputs_sentence = list(bert_inputs[0: batch_size])
            print(len(inputs_sentence))
            inputs = tokenizer(inputs_sentence, padding=True,truncation=True,
                  return_tensors="pt")
            # print(inputs)
            # inputs_list = []
            # inputs_list.append(inputs['input_ids'])
            # inputs_list.append(inputs['token_type_ids'])
            # inputs = []
            # inputs.append(inputs_list)
            # print(f"inputs_list is {inputs_list}, inputs is {inputs}")
            inputs = inputs['input_ids']
        else:
            feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
            inputs = feature_extractor(images=imgs, return_tensors="pt")

        tik = time.time()
        for i in range(num_batches):
            # generate random inputs and labels
            if model_name in ['bert-base-uncased', 'bert-large-uncased']:
                outputs = model(inputs)
            else:
                outputs = model(inputs['pixel_values'])
            print(f"outputs is {outputs}")
            # predicted_class_idx = outputs[0].argmax(-1).item()
            # print("Predicted class:", origin_model.config.id2label[predicted_class_idx])
        ## Calculate time
        tok = time.time()
        latency = tok-tik
        throughput = num_batches*batch_size / latency
        # print(f"Split size is {split_size[si]}, Total program execution time = {tok - tik}")
        latencies.append(latency)
        throughputs.append(throughput)

    best_choice = -1
    best_throughput  = -1
    for i in range(len(split_size)):
        print(f"Split size {split_size[i]}, latency is {latencies[i]}, throughput is {throughputs[i]}")
        if throughputs[i] > best_throughput:
            best_throughput = throughputs[i]
            best_choice = i
    print("\n---------------- Split output line ----------------")
    print(f"\nBest split size is {split_size[best_choice]}, Execution time is {latencies[best_choice]}, throughput is {throughputs[best_choice]}\n")


def run_worker(model_name, model_file, rank, world_size, num_split, batch_size):

    os.environ['MASTER_ADDR'] = args.addr #MASTER_ADDR
    os.environ['MASTER_PORT'] = args.port # MASTER_PORT
    os.environ["TP_SOCKET_IFNAME"] = args.socket_ifname #SOCKET_IFNAME
    os.environ["GLOO_SOCKET_IFNAME"] = args.socket_ifname #SOCKET_IFNAME

    # Higher timeout is added to accommodate for kernel compilation time in case of ROCm.
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=num_worker_threads,rpc_timeout=3000)

    rpc.init_rpc(
        f"worker{rank}",
        rank=rank,
#         backend=rpc.BackendType.PROCESS_GROUP,
        world_size=world_size,
        rpc_backend_options=options
    )
    if rank == 0:
        run_master(model_name, model_file, world_size, num_split, batch_size)

    # block until all rpcs finisha
    rpc.shutdown()

if __name__=="__main__":
    #########################################################
    #                 Check Enviroment Settings             #
    #########################################################
    parser = argparse.ArgumentParser(description="Pipeline Parallelism Runtime")
    parser.add_argument("rank", type=int, help="the rank for the current node")
    parser.add_argument("worldsize", type=int, help="the world size (the number of nodes)")
    parser.add_argument("-m", "--model-name", type=str, default="bert-base-uncased", choices=["google/vit-base-patch16-224",
    "google/vit-large-patch16-224", "google/vit-huge-patch14-224-in21k", "bert-base-uncased", "bert-large-uncased"], help="the neural network model for loading")
    parser.add_argument("-M", "--model-file", type=str, help="the model file, if not in working directory")
    parser.add_argument("-pt", "--partition", default="1,48", help="the partition method")
    parser.add_argument("--addr", type=str, default="127.0.0.1", help="ip address for the master node")
    parser.add_argument("--port", type=str, default="29500", help="communication port for the master node")
    parser.add_argument("-s", "--socket-ifname", type=str, default="lo0", help="socket iframe name, use [ifconfig | ipaddress] to check")
    parser.add_argument("-p","--print", type=str, default = "None", choices=["full", "short", "default"], help="print the [full | short] tensor values")
    parser.add_argument("-t", "--threshold", default=1000, type=int, help="total number of array elements which trigger summarization rather than full repr")
    parser.add_argument("-n", "--num-batches", default=1, type=int, help="total number of batches")
    parser.add_argument("-b", "--batch-size", default=64, type=int, help="batch size")
    parser.add_argument("-w", "--worker-threads", default=128, type=int, help="the number of worker threads for the communication backend")
    parser.add_argument("-sp", "--splits", default="8", help="the list of microbatch size")
    args = parser.parse_args()
    torch.set_printoptions(profile=args.print,threshold=args.threshold)
    ## Force pytorch use CPU
    device = torch.device('cpu')
    # parallel_threads = 2
    # torch.set_num_threads(parallel_threads)
    # torch.set_num_interop_threads(parallel_threads)
    torch.set_grad_enabled(False)
    print(f"Use device: {device},  # parallel intra nodes threads: {torch.get_num_threads()}, # parallel inter nodes threads: {torch.get_num_interop_threads()}")
    #########################################################
    #                 Configuration for Network             #
    #########################################################
    # *****  Define the World Size and partition Method ******#
    partition =   [int(i) for i in args.partition.split(',')]
    num_batches = args.num_batches
    batch_size = args.batch_size
    num_worker_threads = args.worker_threads
    ## random data
    # img = torch.randn(3, 384, 384)
    ## ground truth: Egyptian cat
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    # image = Image.open('./images/panda.jpeg')
    imgs = [image for i in range(batch_size)]

    # ***********************  End  **************************#
    world_size = args.worldsize
    rank=args.rank
    num_split = [int(i) for i in args.splits.split(',')]
    model_name= args.model_name

    model_files_default = {
        'google/vit-base-patch16-224': 'ViT-B_16-224.npz',
        'google/vit-large-patch16-224':'ViT-L_16-224.npz',
        'google/vit-huge-patch14-224-in21k': 'ViT-H_14.npz',
        'bert-base-uncased': 'BERT-B.npz',
        'bert-large-uncased': 'BERT-L.npz',
    }
    model_file = args.model_file
    if model_file is None:
        model_file = model_files_default[model_name]

    print(f"Model name is {model_name}, Batch size is {batch_size}, Split size is: {num_split}, \n Split method is {partition}, GLOO Threads is {num_worker_threads}")

    tik = time.time()
    run_worker(model_name, model_file, rank, world_size, num_split, batch_size)
    tok = time.time()
    print(f"Total program execution time = {tok - tik}")
