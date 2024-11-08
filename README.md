# Mixture of Quantization (MoQ)

Mixture of Quantization is an inference system that slices models and applies different quant formats for the communication of acitivation data.

# Setup
- **Fork (and star) my repo**, then clone the forked version to your local machine or discovery
- If missing any package when running the code, install it using pip or conda regularly
- Note that, for `transformers` package, `version=4.18.0` is highly recommended in this project.

#### Model Preparation
- run save_model_weights.py
```sh
python save_model_weights.py
```
- If you've previously downloaded the model, delete and re-download due to modifications in this version.
 - Ensure you successfully download `ViT-B_16-224.npz`, `ViT-L_16-224.npz`, and `resnet18.pt`.

#### Dataset Preparation
- On local: 
	- Download the dataset from [ImageNet Large Scale Visual Recognition Challenge 2012 (ILSVRC2012) validation dataset (6.3GB)](https://www.image-net.org/challenges/LSVRC/index.php).
  - Unzip and run `valprep.sh` on the downloaded file.
- On Discovery
	- modify `evaluation.py` from lines 204 to 218.
	
   

#### Project Structure
- Create a folder called `result` in your project; 
- if  `result` already exists, ensure it is empty by deleting all inside files.
- **Your project should look like this:**
```sh
.
├── ViT-B_16-224.npz
├── ViT-L_16-224.npz
├── resnet18.pt
├── resnet50.pt
├── evaluation.py
├── runtime.py
├── get_activation_data.py
├── evaluation_tools
│   ├── evaluation_partition.sh
│   └── upload_eval.job
├── pipeedge
│   ├── quantization
│   └── others
├── ILSVRC2012_img_val
│   ├── n0......
│   │   ├── (lots of).JPEG
│   └── n0......
├── result
├── others
```

# Main Instruction

#### Running on Local
- **Command**:
```sh
python evaluation.py -pt 1,10,11,21 -q 8,8 -e 6 -m torchvision/resnet18 -clamp
```
- Meaning: Quantize at the 10th layer using 8-bit total (6 bits for exponential) and clamping for the resnet18 model.

-   Note1: Please wait at the beginning as the program preprocesses the dataset and model. Results will appear line by line, as well as in the `result/{model_name}/{job_info}.txt`

-   Note2: If using `-e 0`, it will automatically switch to `Integer Quant`, otherwise, it will use `Float-Point Quant`

#### Running on Discovery
- Modify `evaluation.py` from lines 204 to 218, switch the dataset source.
- Delete any files inside the result!!! or it will raise error.
- Modify  `evaluation_tools/upload_eval.job` 
	- Change `cd`  to your own path of Discovery.
	- Choose the corresponding command for your model
- Modify `evaluation_tools/evaluation_partition.sh`
	- choose the corresponding command for your model
-   **Command Running on Discovery**:
```sh
cd evaluation_tools
./evaluation_partition.sh
```
-   You will see Discovery generating thousands of jobs.
    -   These are combinations of (bit, e, layer).
-   Results appear at `result/{model_name}/{job_name}.txt`. If you see the txt file, basically it is running successfully.
-   Outputs are also available at `evaluation_tools/slurm_{job_id}.out`.
	- you need to check this `.out` file to check if there is any error, once you submit the job.
- Note: You may need to give `evaluation_partition.sh` root permission first.

# Useful Command Lines

#### Check all jobs Status
```sh
squeue --me
```

#### Check single job status
```sh
jobinfo {job_id}
```

#### Count Running/Waiting Jobs
```sh
squeue --me | awk '
BEGIN {
    abbrev["R"]="(Running)"
    abbrev["PD"]="(Pending)"
    abbrev["CG"]="(Completing)"
    abbrev["F"]="(Failed)"
}
NR>1 {a[$5]++}
END {
    for (i in a) {
        printf "%-2s %-12s %d\n", i, abbrev[i], a[i]
    }
}'
```

#### Cancel all jobs (if you find some error)
```sh
scancel --me
```

#### Delete all `.out` files in evaluation_tools (if you don't want the new `.out` mixed with old file)

```sh
find . -type f -name "*.out" -exec rm {} +
```

# Some important files

- Core code of quantization: `pipeedge\quantization`
- Download model weight: `save_model_weights.py`
- About the model: `README_Profiler.md`
- About the testbed, pipeedge: `README_Pipeedge.md`
- Get model activation data per layer: `get_activation_data.py`
