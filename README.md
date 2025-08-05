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

split_ai_eval
│── models/
│   └── vgg16_model.py        
│       # Custom VGG16 model definition
│
│── nodes/
│   ├── ue_node.py            
│   │   # User Equipment (UE) class handling compute & energy
│   ├── network_node.py       
│       # Network compute node class (edge/cloud)
│
│── utils/
│   ├── flop_utils.py         
│   │   # FLOPs estimation and computation time calculation
│   ├── energy_utils.py       
│   │   # UE-specific energy consumption functions
│   ├── param_generator.py    
│   │   # Generates random CPU frequencies, FLOPs, bandwidth
│   ├── comm_utils.py         
│       # Communication latency and energy modeling utilities
│
│── scripts/
│   ├── main.py               
│   │   # End-to-end collaborative inference evaluation
│   ├── rl_train.py           
│   │   # Reinforcement learning agent training for adaptive split decisions
│   ├── train_model.py        
│   │   # (Optional) Train and save model weights
│   └── visualize_results.py  
│       # (Optional) Visualization of split and performance
│
│── config.py                 
│   # Configuration file for hyperparameters and setup
│
│── README.md
│── requirements.txt          
│   # Python dependencies

