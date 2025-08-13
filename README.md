# splitAi-6g

splitai-6g is a research framework for **layer-wise partitioning of deep neural networks (DNNs)** across heterogeneous computation nodes in **6G edge-cloud environments**. The project focuses on **energy-efficient and low-latency collaborative inference**, where DNN layers can be dynamically split and deployed on user equipment (UE), gNB, Edge, or Core.

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
    - `network_node.py` → Network compute node class (gNB/Edge/Core)  
  - **utils/**
    - `flop_utils.py` → FLOPs computation time calculation  
    - `flops_profile.py` → FLOPs per-layer (segment) profiler using flattened VGG16 layers  
    - `energy_utils.py` → UE-specific energy consumption functions  
    - `param_generator.py` → Generates random CPU frequencies, FLOPs, bandwidth  
    - `comm_utils.py` → Communication latency and energy modeling utilities  
    - `split_generator.py` → Generates random multi-node split configurations
    - `rl_utils.py` → RL-specific utility functions
  - **rl/**
    - **initial_models/** → stores initial model params for a given number of states and actions
    - `generate_model_params.py` → script to instantiate RL model architecture and save initial model params 
    - `ddqn.py` → defines the ddqn algorithm and associated functions and parameters
    - `agent.py` → defines the RL agent and associated functions to execute training or inference
  - **logs/**
    - **random/** → stores kpis and logs of a random split generator
    - **rl/** → stores kpis and logs of rl-based algorithm
      - **ddqn/**
        - **models/** → stores the model params at the end of each training episode
    - **optimum/** → stores kpis and logs of the optimal solution
  - **postprocessing/**
    - `plot_system_kpis.py` → script that reads, parses and plots system kpis
  - **results/**
  - `main.py` → End-to-end collaborative inference evaluation  
  - `README.md` → Project documentation  

## Pretrained Weights Setup

Before running the main collaborative inference script, you need to convert and save pretrained VGG16 weights for the custom model.  
This step only needs to be run **once**:

```bash
python3 models/convert_pretrained_vgg16.py
```
## RL model params setup

Before running the main collaborative inference script, you need to instantiate the RL model architecture and save the weights and biases.
This step only needs to be run **once**:

```bash
python3 rl/generate_model_params.py
```

## Running SplitAI Inference

After setting up the pretrained weights, you can run the SplitAI collaborative inference:

```bash
python3 main.py
```
