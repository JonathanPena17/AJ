# Fault Injection Experiment with NVBitFI (LeNetMNIST)
This repository contains all you need to reproduce fault injection experiments with NVBitFI and the LeNetMNIST application on AWS, using an NVIDIA Tesla T4 GPU.
Goal: Study ML model resilience under fault injection (bit flips, timing errors).
Team: AJ
Team member names: Anuja Sawant, Jonathan Peña

#System Requirements
Operating System: Ubuntu 20.04.6 LTS (GNU/Linux 5.15.0-1084-aws x86_64)

Instance Type: g4dn.xlarge (AWS EC2)

vCPUs: 4, RAM: 16 GB, Root Volume: 100 GB

GPU: NVIDIA Tesla T4

NVIDIA Driver: 525.78.01 (compatible with CUDA 11.7)

CUDA Toolkit: 11.7

cuDNN: 8.5 or compatible with CUDA 11.x

Python: 3.8+

Git: Latest available via apt

NVBitFI: Latest (from official repository)

GCC: Version 11 recommended

# Setup Instructions

1. Launch your AWS instance
Use the official Ubuntu 20.04 LTS AMI.

Allocate 100GB storage for root volume.

Select the g4dn.xlarge instance type with a Tesla T4 GPU.

2. Clone NVBitFI Repository: 

git clone https://github.com/NVlabs/nvbitfi.git
cd nvbitfi
make

3. Clone this repository: This repository will contain all the files need to run the injection campaign

git clone https://github.com/JonathanPena17/AJ

4. Copy the lenetMNIST into the scripts/test-apps directory

5. Open and edit the param.py file, include this into the .py file

'NUM_INJECTIONS = 50'
'THRESHOLD_JOBS = 10'

'inst_value_igid_bfm_map = {
        G_GP: [FLIP_SINGLE_BIT, FLIP_TWO_BITS, RANDOM_VALUE, ZERO_VALUE],
        G_FP64: [FLIP_SINGLE_BIT, FLIP_TWO_BITS, RANDOM_VALUE, ZERO_VALUE],
        G_FP32: [FLIP_SINGLE_BIT, FLIP_TWO_BITS, RANDOM_VALUE, ZERO_VALUE],
        G_LD: [FLIP_SINGLE_BIT, FLIP_TWO_BITS, RANDOM_VALUE, ZERO_VALUE],
}'


apps = {   
         '"lenetMNIST": [
                             NVBITFI_HOME + '/test-apps/lenetMNIST', # workload directory
                             'lenet5_mnist', # binary name
                             NVBITFI_HOME + '/test-apps/lenetMNIST/', # path to the binary file
                             1, # expected runtime
                             "train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte" # additional parameters to the run.sh'
        }
   
7. In addition params.py can modified on the ammount of injections and types on injections. See https://github.com/NVlabs/nvbitfi.git for more details






