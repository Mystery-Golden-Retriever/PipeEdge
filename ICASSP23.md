# Experiments Record for ICASSP23 submission

This is the manuscript for re-producing the experiments in paper "QuantPipe: Applying Adaptive Post-training Quantization for Distributed Transformer Pipelines in Dynamic Edge Environments".

***
## Experiment settings
- using the PipeEdge's code version --- demo-20220824:

  `git checkout demo-20220824`

- activate the corresponding environment (QtTest on AGX-Orin-00 / 01 / 02):

  `. Venv/QtTest/bin/activate`

***
## Experiment steps
1. On AGX-Orin-00:

   `WINDOW_SIZE=50 python runtime.py 0 3 -s eth0 -c p2p -M /opt/fogsys/models/ViT-B_16-224.npz -b 64000 -pt 1,22,23,48 -r 1,2 --dataset-name ImageNet --dataset-root /mnt/nvme0n1p1/ImageNet/ -u 64 -ppn 1000 -g --dataset-shuffle True`

2. On AGX-Orin-01:

   `WINDOW_SIZE=50 python runtime.py 1 3 -s eth0 -c p2p -M /opt/fogsys/models/ViT-B_16-224.npz --addr agx-orin-00 -u 64 -d cuda -g -ppn 1000`

3. On AGX-Orin-02:

   `WINDOW_SIZE=50 python runtime.py 2 3 -s eth0 -c p2p -M /opt/fogsys/models/ViT-B_16-224.npz --addr agx-orin-00 -u 64 -d cuda`

4. In the pop-out windows (agx-orin-00 and agx-orin-01), click start buttons. The PipeEdge System will start to run.

5. Monitoring the output rate of agx-orin-01, and do bandwidth control at specific iteration: (all operations are performed on a seperate agx-orin-01 terminal)

   - at iter. 100, run script `~/set_bandwidth.sh 500`

   - at iter. 300, run script `~/update_bandwidth.sh 50`

   - at iter. 500, run script `~/update_bandwidth.sh 250`

   - at iter. 700, run script `~/remove_bandwidth.sh`

***
## Scripts
To use `tc` command on the Jetson Orin devices, it requires some kernel functioin support for Linux-arm-arch64, so first run the followinig:

      sudo insmod /opt/fogsys/kernel-5.10/modules/sch_tbf.ko
      \# do some stuff here
      sudo rmmod sch_tbf

- set_bandwidth.sh

        #!/bin/bash

        sudo tc qdisc add dev eth0 root tbf rate $1Mbit latency 50ms burst 15kb

- update_bandwidth.sh

        #!/bin/bash

        sudo tc qdisc del dev eth0 root
        sudo tc qdisc add dev eth0 root tbf rate $1Mbit latency 50ms burst 15kb

- remove_bandwidth.sh

        #!/bin/bash

<<<<<<< HEAD
        sudo tc qdisc del dev eth0 root
=======
        sudo tc qdisc del dev eth0 root

***
## Automatic bandwidth control for experiment reproduction in ICASSP2023

To reproduce the experiment results we show in the paper of ICASSP2023, one can use the 'auto_bandwidth_control.py' script. Right after boot up the PipeEdge system and click start button on the GUI panel, run command 

`sudo python tools/auto_bandwidth_control.txt`
>>>>>>> e6174d927ba8602c15b1f2515bcdb493cac0c3cc
