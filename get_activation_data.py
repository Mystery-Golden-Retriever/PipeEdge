""" Evaluate accuracy on ImageNet dataset of PipeEdge """
import os
import argparse
import time
import torch
from typing import List
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder,ImageNet
from torchvision import transforms
from transformers import DeiTFeatureExtractor, ViTFeatureExtractor
from runtime import forward_hook_quant_encode, forward_pre_hook_quant_decode
from utils.data import ViTFeatureExtractorTransforms
import model_cfg
from evaluation_tools.evaluation_quant_test import *

class ReportAccuracy():
    def __init__(self, batch_size, output_dir, model_name, partition, quant,e_bit,is_Clamp) -> None:
        self.current_acc = 0.0
        self.total_acc = 0.0
        self.correct = 0
        self.tested_batch = 0
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.partition = partition
        self.quant = quant
        self.e_bit = e_bit
        self.is_Clamp = is_Clamp
        self.model_name = model_name.split('/')[1]
        self.file_name=""
        # generate file path
        if self.is_Clamp:
            clamp_word="withClamp"
        else:
            clamp_word="noClamp"
        if self.e_bit==0:
            self.file_name = os.path.join(self.output_dir, self.model_name, f"b{self.quant}_integer_{clamp_word}_{self.partition[1]}.txt")
        else:
            self.file_name = os.path.join(self.output_dir, self.model_name, f"b{self.quant}_float_e={self.e_bit}_{clamp_word}_{self.partition[1]}.txt")
        # check if the file exists, if Ture, means that we need to delete the previous result            
        assert not os.path.exists(self.file_name),f"{self.file_name} is existed, please delete the result of previous experiment"
        os.makedirs(os.path.dirname(self.file_name), exist_ok=True)
        print(self.file_name)

    def update(self, pred, target):
        self.correct = pred.eq(target.view(1, -1).expand_as(pred)).float().sum()
        self.current_acc = self.correct / self.batch_size
        self.total_acc = (self.total_acc * self.tested_batch + self.current_acc)/(self.tested_batch+1)
        self.tested_batch += 1

    def report(self,):
        print(f"The accuracy so far is: {100*self.total_acc:.2f}")
        with open(self.file_name, 'a') as f:
            f.write(f"{100*self.total_acc:.2f}\n")
                    
def _make_shard(model_name, model_file, stage_layers, stage, q_bits):
    shard = model_cfg.module_shard_factory(model_name, model_file, stage_layers[stage][0],
                                            stage_layers[stage][1], stage)
    shard.register_buffer('quant_bits', q_bits)
    shard.eval()
    return shard

def _forward_model(input_tensor, model_shards,ith_layer):
    num_shards = len(model_shards)
    temp_tensor = input_tensor
    for idx in range(num_shards):
        shard = model_shards[idx]

        # decoder
        # if idx != 0:
        #     temp_tensor = forward_pre_hook_quant_decode(shard, temp_tensor)

        
        # forward
        if isinstance(temp_tensor[0], tuple) and len(temp_tensor[0]) == 2:
            temp_tensor = temp_tensor[0]
        elif isinstance(temp_tensor, tuple) and isinstance(temp_tensor[0], torch.Tensor):
            temp_tensor = temp_tensor[0]
        
        temp_tensor = shard(temp_tensor)
        if isinstance(temp_tensor, tuple):
            temp_tensor = temp_tensor[0]
        
        save_path = f"ResNet50_activation_data/batch_1/layer_{ith_layer}_x.npy"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, temp_tensor.detach().numpy())
        break
        # encoder
        # if idx != num_shards-1:
        #     temp_tensor = (forward_hook_quant_encode(shard, None, temp_tensor),)
    return temp_tensor

def evaluation(args):
    """ Evaluation main func"""
    # localize parameters
    dataset_path = args.dataset_root
    dataset_split = args.dataset_split
    batch_size = args.batch_size
    ubatch_size = args.ubatch_size
    num_workers = args.num_workers
    partition = args.partition
    quant = args.quant
    output_dir = args.output_dir
    model_name = args.model_name
    model_file = args.model_file
    num_stop_batch = args.stop_at_batch
    e_bit=args.e_bit
    is_Clamp = args.is_Clamp
    print(args)
    # if model_file is None:
    #     model_file = model_cfg.get_model_default_weights_file(model_name)

    # load dataset
    if model_name in ['facebook/deit-base-distilled-patch16-224',
                        'facebook/deit-small-distilled-patch16-224',
                        'facebook/deit-tiny-distilled-patch16-224']:
        feature_extractor = DeiTFeatureExtractor.from_pretrained(model_name)
    elif model_name.startswith('torchvision'):
        feature_extractor = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        # transforms.Lambda(lambda x: x.unsqueeze(0))
        ])
        val_dataset = ImageFolder(os.path.join(dataset_path, dataset_split),
                                transform = feature_extractor)
    else:
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        val_transform = ViTFeatureExtractorTransforms(feature_extractor)
        val_dataset = ImageFolder(os.path.join(dataset_path, dataset_split),
                            transform = val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle=True,
        pin_memory=True
    )
    
    # model config
    def _get_default_quant(n_stages: int) -> List[int]:
        return [0] * n_stages
    parts = [int(i) for i in partition.split(',')]
    assert len(parts) % 2 == 0
    num_shards = len(parts)//2
    stage_layers = [(parts[i], parts[i+1]) for i in range(0, len(parts), 2)]
    stage_quant = [int(i) for i in quant.split(',')] if quant else _get_default_quant(len(stage_layers))

    for start in range(2,54):
        stage_layers=[(1,start),(start+1,54)]
        # model construct
        model_shards = []
        q_bits = []
        for stage in range(num_shards):
            q_bits = torch.tensor((0 if stage == 0 else stage_quant[stage - 1], stage_quant[stage]))
            model_shards.append(_make_shard(model_name, model_file, stage_layers, stage, q_bits))
            model_shards[-1].register_buffer('quant_bit', torch.tensor(stage_quant[stage]), persistent=False)
            model_shards[-1].register_buffer('e_bit', torch.tensor(e_bit), persistent=False)
            model_shards[-1].register_buffer('is_Clamp', torch.tensor(is_Clamp), persistent=False)

        # run inference
        start_time = time.time()
        # acc_reporter = ReportAccuracy(batch_size, output_dir, model_name, parts, stage_quant[0],e_bit,is_Clamp)
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(val_loader):
                if batch_idx == num_stop_batch and num_stop_batch:
                    break
                output = _forward_model(input, model_shards,start)
                # _, pred = output.topk(1)
                # pred = pred.t()
                # acc_reporter.update(pred, target)
                # acc_reporter.report()
                break
        # print(f"Final Accuracy: {100*acc_reporter.total_acc}; Quant Bitwidth: {stage_quant}")
        end_time = time.time()
        print(f"total time = {end_time - start_time}")


if __name__ == "__main__":
    """Main function."""
    parser = argparse.ArgumentParser(description="Pipeline Parallelism Evaluation on Single GPU",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Eval configs
    parser.add_argument("-q", "--quant", type=str,
                        help="comma-delimited list of quantization bits to use after each stage")
    parser.add_argument("-pt", "--partition", type=str, default= '1,22,23,48',
                        help="comma-delimited list of start/end layer pairs, e.g.: '1,24,25,48'; "
                             "single-node default: all layers in the model")
    parser.add_argument("-o", "--output-dir", type=str, default="result/")
    parser.add_argument("-st", "--stop-at-batch", type=int, default=None, help="the # of batch to stop evaluation")
    parser.add_argument("-e", "--e-bit", default=1, type=int,
                        help="the number of exponential bit for the float quant, if e=0, we will use integer quant")
    parser.add_argument("-clamp", "--is-Clamp", action='store_true',
                        help="if we do clamp as a preprocess for data")
    # Device options
    parser.add_argument("-d", "--device", type=str, default=None,
                        help="compute device type to use, with optional ordinal, "
                             "e.g.: 'cpu', 'cuda', 'cuda:1'")
    parser.add_argument("-n", "--num-workers", default=16, type=int,
                        help="the number of worker threads for the dataloder")
    # Model options
    parser.add_argument("-m", "--model-name", type=str, default="google/vit-base-patch16-224",
                        choices=model_cfg.get_model_names(),
                        help="the neural network model for loading")
    parser.add_argument("-M", "--model-file", type=str,
                        help="the model file, if not in working directory")
    # Dataset options
    parser.add_argument("-b", "--batch-size", default=64, type=int, help="batch size")
    parser.add_argument("-u", "--ubatch-size", default=8, type=int, help="microbatch size")

    dset = parser.add_argument_group('Dataset arguments')
    dset.add_argument("--dataset-name", type=str, default='ImageNet', choices=['CoLA', 'ImageNet'],
                      help="dataset to use")
    
    
    # In local, use the below commented code, and check if you have already downloaded the dataset
    dset.add_argument("--dataset-root", type=str, default= "",
                      help="dataset root directory (e.g., for 'ImageNet', must contain "
                           "'ILSVRC2012_devkit_t12.tar.gz' and at least one of: "
                           "'ILSVRC2012_img_train.tar', 'ILSVRC2012_img_val.tar'")
    dset.add_argument("--dataset-split", default='ILSVRC2012_img_val/', type=str,
                      help="dataset split (depends on dataset), e.g.: train, val, validation, test")
    
    # In discovery, use the below commented code, you dont need to dowanload the dataset to the discovery.
    # dset.add_argument("--dataset-root", type=str, default= "/project/jpwalter_148/hnwang/datasets/ImageNet/",
    #                   help="dataset root directory (e.g., for 'ImageNet', must contain "
    #                        "'ILSVRC2012_devkit_t12.tar.gz' and at least one of: "
    #                        "'ILSVRC2012_img_train.tar', 'ILSVRC2012_img_val.tar'")
    # dset.add_argument("--dataset-split", default='val', type=str,
    #                   help="dataset split (depends on dataset), e.g.: train, val, validation, test")
    
    dset.add_argument("--dataset-indices-file", default=None, type=str,
                      help="PyTorch or NumPy file with precomputed dataset index sequence")
    dset.add_argument("--dataset-shuffle", type=bool, nargs='?', const=True, default=False,
                      help="dataset shuffle")
    args = parser.parse_args()

    evaluation(args)
