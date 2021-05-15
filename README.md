## Dynamic Compression for Communication Efficient Federated Learning

The source code is tested on FASRC. However, you can run it on your local machine as well (by calling mpiexec on the main.py file).

## Installation Requirements
mpi4py, 
tensorflow,
numpy

## Step 1: Create Chkpt file (Run this first)
Model initialization file. Currently we support LeNet, ResNet (for CIFAR-10), and FLNet (a custom network based on residual network building blocks). Open the file in vim and choose `base_model` = "resnet". Valid options are "lenet" and "FLnet". Once you select your model, please run the checkpoint creator.

```bash
python3 chkpt_creator.py
```


## Step 2: Run the Federated learning 
The `main.py` contains the source code to run FL. Currently the code is configured to spawn one server (parameter sever) and 10 client nodes. The communication is between these distributed system is managed by MPI. You can select the compression methods (valid ones are : FBK -- Feedback error correction, Topk -- Topk sparsity, quant -- for quantization, kdecay for either linear decay or exponential decay). Once you have configured the parameters, you can launch the FL training by calling the batch script.

```sbatch run_main.bath
```

## Step 3: Monitor Training Progress

Once you have launched the training, the TFevents files will be generated inside the `\logs` folder. You can monitor the training progress in tensorboard using the following command:

```tensorboard --logdir logs
```
