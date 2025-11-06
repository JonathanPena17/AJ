# Fault Injection Experiment with NVBitFI (LeNetMNIST)

This repository contains all you need to reproduce fault injection experiments with NVBitFI and the LeNetMNIST application on AWS, using an NVIDIA Tesla T4 GPU.

**Goal:** Study ML model resilience under fault injection (bit flips, timing errors).  
**Team:** AJ  
**Team members:** Anuja Sawant, Jonathan Peña

---

## System Requirements

- Operating System: Ubuntu 20.04.6 LTS (GNU/Linux 5.15.0-1084-aws x86_64)
- Instance Type: g4dn.xlarge (AWS EC2)
- vCPUs: 4
- RAM: 16 GB
- Root Volume: 100 GB
- GPU: NVIDIA Tesla T4
- NVIDIA Driver: 525.78.01 (compatible with CUDA 11.7)
- CUDA Toolkit: 11.7
- cuDNN: 8.5 or compatible with CUDA 11.x
- Python: 3.8+
- Git: Latest available via apt
- NVBitFI: Latest (from official repository)
- GCC: Version 11 recommended

---

## Setup Instructions

1. **Launch Your AWS Instance**
   - Use the official Ubuntu 20.04 LTS AMI
   - Allocate 100GB storage for root volume
   - Select the g4dn.xlarge instance type with a Tesla T4 GPU
     
2. **Install NVIDIA Drivers & CUDA Toolkit**
   ```
   # Add repository and key
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
   sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
   sudo apt update
   sudo apt install cuda-11-7 nvidia-driver-525 -y
   echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
   source ~/.bashrc
   ```

   - **Verify installations:**
     ```
     nvidia-smi
     nvcc --version
     ```

3. **Install Python, Git, and GCC**
   ```
   sudo apt install python3 python3-pip git gcc-11 g++-11 -y
   python3 --version
   git --version
   gcc --version
   ```

4. **Clone NVBitFI Repository**
   ```
   git clone https://github.com/NVlabs/nvbitfi.git
   cd nvbitfi
   make
   ```

5. **Clone This Repository**
   ```
   cd ~
   git clone https://github.com/JonathanPena17/AJ.git
   ```

6. **Copy LeNetMNIST to NVBitFI Test Apps**
   ```
   cp -r ~/AJ/lenetMNIST ~/nvbitfi/test-apps/
   ```

7. **Configure `params.py`**
   - Open the params.py file in the NVBitFI scripts directory:
     ```
     cd ~/nvbitfi/scripts
     nano params.py
     ```
   - Add or modify these parameters:
     ```
     NUM_INJECTIONS = 50
     THRESHOLD_JOBS = 10

     inst_value_igid_bfm_map = {
         G_GP: [FLIP_SINGLE_BIT, FLIP_TWO_BITS, RANDOM_VALUE, ZERO_VALUE],
         G_FP64: [FLIP_SINGLE_BIT, FLIP_TWO_BITS, RANDOM_VALUE, ZERO_VALUE],
         G_FP32: [FLIP_SINGLE_BIT, FLIP_TWO_BITS, RANDOM_VALUE, ZERO_VALUE],
         G_LD: [FLIP_SINGLE_BIT, FLIP_TWO_BITS, RANDOM_VALUE, ZERO_VALUE],
     }

     apps = {   
         "lenetMNIST": [
             NVBITFI_HOME + '/test-apps/lenetMNIST',  # workload directory
             'lenet5_mnist',  # binary name
             NVBITFI_HOME + '/test-apps/lenetMNIST/',  # path to the binary file
             1,  # expected runtime
             "train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte"  # additional run.sh parameters
         ]
     }

     ```
8. **Configure `Makefile`**
   - Open the Makefile file in the NVBitFI profiler directory:
     ```
     cd ~/nvbitfi/profiler
     nano Makefile
     ```
   - Modify this parameter:
     ```
     FAST_APPROXIMATE_PROFILE=-DSKIP_PROFILED_KERNELS
     ```

---

## Running the Experiments

- Navigate to NVBitFI tools directory:
  ```
  cd ~/nvbitfi/tools/nvbitfi
  ```
- Run to profile the application:
  ```
   python3 scripts/run_profiler.py
  ```
- Run to generate injection list:
  ```
   python3 scripts/generate_injection_list.py 
  ```
  
- Run the fault injection campaign:
  ```
  python3 scripts/run_injections.py standalone 
  ```
- Monitor progress:
  ```
  tail -f logs/lenetMNIST/nohup.out
  ```
- Parse results:
  ```
  python3 scripts/parse_results.py
  ```
- Results will be located in the log directory:
  ```
  cd ~/nvbitfi/tools/nvbitfi/logs/results

---

## References

- [NVBitFI GitHub Documentation](https://github.com/NVlabs/nvbitfi)

---

