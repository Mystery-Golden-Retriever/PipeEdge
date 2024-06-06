# Tests

Run tests from the top-level project directory (not this directory!) with:

```sh
python -m unittest -v
```


## Testcases for ResNet models

To test the ResNet models with `runtime.py`, run the following commands:

(ImageNet)
`(node 1) python runtime.py 0 2 -s eth0 -m torchvision/resnet18 -pt 1,10,11,21 --dataset-name ImageNet --dataset-root /project/jpwalter_148/hnwang/datasets/ImageNet/ --dataset-split val`
`(node 2) python runtime.py 1 2 -s eth0 --addr 10.125.75.20 -m torchvision/resnet18`

(Single Image)
`(node 1) python runtime.py 0 2 -s eth0 -m torchvision/resnet18 -pt 1,10,11,21`
`(node 2) python runtime.py 1 2 -s eth0 --addr 10.125.75.20 -m torchvision/resnet18`

To test the ResNet models with `evaluation.py`, run the following commands:

`python evaluation.py -q 8,8 -pt 1,10,11,21 -m torchvision/resnet18`
