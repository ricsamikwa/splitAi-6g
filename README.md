# splitAi-6g

splitai-6g is a research framework for **layer-wise partitioning of deep neural networks (DNNs)** across heterogeneous computation nodes in **6G edge-cloud environments**. The project focuses on **energy-efficient and low-latency collaborative inference**, where DNN layers can be dynamically split and deployed on user equipment (UE), base stations, edge servers, or the cloud.

## Features

- Layer-based DNN partitioning with multi-node support  
- Optimization framework minimizing total inference latency and UE energy consumption  
- FLOPs-based computation time and energy modeling  
- Communication cost modeling with UE-specific energy  
- Energy credit mechanism for managing UE energy budget during collaborative inference  
- Reinforcement learning (RL) environment for adaptive split decisions  

## Repository Structure

- **splitAi-6g/**
  - **models/**
    - `vgg16_model.py` → VGG16 model definition  
    - `convert_pretrained_vgg16.py` → Script to convert and save pretrained VGG16 weights
  - **nodes/**
    - `ue_node.py` → UE class handling compute & energy  
    - `network_node.py` → Network compute node class (edge/cloud)  
  - **utils/**
    - `flop_utils.py` → FLOPs computation time calculation  
    - `flops_profile.py` → FLOPs per-layer (segment) profiler using flattened VGG16 layers  
    - `energy_utils.py` → UE-specific energy consumption functions  
    - `param_generator.py` → Generates random CPU frequencies, FLOPs, bandwidth  
    - `comm_utils.py` → Communication latency and energy modeling utilities  
  - `main.py` → End-to-end collaborative inference evaluation  
  - `README.md` → Project documentation  

## Pretrained Weights Setup

Before running the main collaborative inference script, you need to convert and save pretrained VGG16 weights for the custom model.  
This step only needs to be run **once**:

```bash
python3 models/convert_pretrained_vgg16.py
```

## Running SplitAI Inference

After setting up the pretrained weights, you can run the SplitAI collaborative inference:

```bash
python3 main.py
```
