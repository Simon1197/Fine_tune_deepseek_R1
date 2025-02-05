# deepseek-r1

Then command if run on docker (required larger /dev/shm space)
```
docker run --gpus all -it --shm-size=8G nvidia/cuda:12.2.0-devel-ubuntu22.04 /bin/bash
```

Should have create the conda env for GPU running
Step 1. setup the env
Linux
```
mkdir -p ~/miniconda3

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh

bash /tmp/miniconda.sh -b -u -p ~/miniconda3
```

```
source ~/miniconda3/bin/activate

conda init --all   
```

```
conda create --name deepseek python=3.10.0 -y
```

```
conda activate deepseek
```

Step 2. 
install the requirments
```
pip install -r requirments.txt
```

Step 3.
install datasets or implement your custom datasets (Here, we use Nvidia HelpSteer to illustrate)
```
pip install datasets
```

# execute command
```
CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 deepseek.py
```



