# Awesome Decision Transformer

This is a collection of research papers for **Decision Transformer (DT)**.
And the repository will be continuously updated to track the frontier of DT.

Welcome to follow and star!

## Table of Contents

- [A Taxonomy of DT Algorithms](#a-taxonomy-of-decision-transformer-algorithms)
- [Papers](#papers)

  - [Arxiv](#arxiv)
  - [ICLR 2023](#iclr-2023) (**<font color="red">New!!!</font>**)
  - [NeurIPS 2022](#neurips-2022)
  - [CoRL 2022](#corl-2022)
  - [ICML 2022](#icml-2022)
  - [AAAI 2022](#aaai-2022)
  - [ICLR 2022](#iclr-2022)
  - [NeurIPS 2021](#neurips-2021)
  - [ICML 2021](#icml-2021)
- [Contributing](#contributing)

## Overview of Transformer

The Decision Transformer was proposed by “Decision Transformer: Reinforcement Learning via Sequence Modeling” by Chen L. et al. It casts (offline) Reinforcement Learning as a **conditional-sequence modeling** problem.

![image info](./architecture.png)

Specifically, DT model is a causal transformer model conditioned on the desired return, (past) states, and actions to generate future actions in an autoregressive manner.

<div align=center>
<img src=./dt-architecture.gif/>
</div>

### Advantage

1. Bypass the need for bootstrapping for long term credit assignment
2. Avoid undesirable short-sighted behaviors due to the discounting future rewards.
3. Enjoy the transformer models widely used in language and vision, which are easy to scale and adapt to multi-modal data.

## Papers

```
format:
- [title](paper link) [links]
  - author1, author2, and author3...
  - publisher
  - key 
  - code 
  - experiment environment
```

### Arxiv

- [Can Offline Reinforcement Learning Help Natural Language Understanding?](https://arxiv.org/abs/2212.03864)
  - Ziqi Zhang, Yile Wang, Yue Zhang, Donglin Wang
  - Key: Language model
  - ExpEnv: MuJoco, Maze 2D

- [Hierarchical Decision Transformer](https://arxiv.org/abs/2209.10447)
  - André Correia, Luís A. Alexandre
  - Key: Hierarchical Learning, Imitation Learning
  - ExpEnv: [MuJoco](https://github.com/openai/mujoco-py), [D4RL](https://github.com/rail-berkeley/d4rl), RoboMimic, Maze 2D

- [PACT: Perception-Action Causal Transformer for Autoregressive Robotics Pre-Training](https://arxiv.org/abs/2209.11133)
  - Rogerio Bonatti, Sai Vemprala, Shuang Ma, Felipe Frujeri, Shuhang Chen, Ashish Kapoor
  - Key: Robotics, Pretrain, Multitask, Representation
  - ExpEnv: MuSHR car, Habitat

- [LATTE: LAnguage Trajectory TransformEr](https://arxiv.org/abs/2208.02918)
  - Arthur Bucker, Luis Figueredo, Sami Haddadin, Ashish Kapoor, Shuang Ma, Sai Vemprala, Rogerio Bonatti
  - Key: MultiModal,  Robotics
  - Code: [official](https://github.com/arthurfenderbucker/latte-language-trajectory-transformer), [official](https://github.com/arthurfenderbucker/nl_trajectory_reshaper)
  - ExpEnv: [CoppeliaSim](https://www.coppeliarobotics.com/)

- [Q-learning Decision Transformer: Leveraging Dynamic Programming for Conditional Sequence Modelling in Offline RL](https://arxiv.org/abs/2209.03993)
  - Taku Yamagata, Ahmed Khalil, Raul Santos-Rodriguez
  - Key: Q-Learning
  - ExpEnv: [D4RL](https://github.com/rail-berkeley/d4rl)

- [Multi-Game Decision Transformers](https://arxiv.org/abs/2205.15241)
  - Kuang-Huei Lee, Ofir Nachum, Mengjiao Yang, Lisa Lee, Daniel Freeman, Winnie Xu, Sergio Guadarrama, Ian Fischer, Eric Jang, Henryk Michalewski, Igor Mordatch
  - Key: Multi-Task,  Finetuning
  - Code: [official](https://sites.google.com/view/multi-game-transformers)
  - ExpEnv: [Atari](https://github.com/openai/gym), [REM](https://github.com/google-research/batch_rl)

- [Offline Pre-trained Multi-Agent Decision Transformer: One Big Sequence Model Tackles All SMAC Tasks](https://arxiv.org/abs/2112.02845)
  - Linghui Meng, Muning Wen, Yaodong Yang, Chenyang Le, Xiyun Li, Weinan Zhang, Ying Wen, Haifeng Zhang, Jun Wang, Bo Xu
  - Key: Multi-Agent RL
  - Code: [official](https://github.com/reinholdm/offline-pre-trained-multi-agent-decision-transformer)
  - ExpEnv: [SMAC](https://github.com/oxwhirl/smac)

- [Transfer learning with causal counterfactual reasoning in Decision Transformers](https://arxiv.org/abs/2110.14355)
  - Ayman Boustati, Hana Chockler, Daniel C. McNamee
  - Key: Causal reasoning, Transfer Learning
  - ExpEnv: [MINIGRID](https://github.com/Farama-Foundation/gym-minigrid)

- [Pretraining for Language Conditioned Imitation with Transformers](https://openreview.net/forum?id=eCPCn25gat)
  - Aaron L Putterman, Kevin Lu, Igor Mordatch, Pieter Abbeel
  - Key: Text-Conditioned Decision
  - ExpEnv: Text-Conditioned Frostbite (MultiModal Benchmark)

- [An Offline Deep Reinforcement Learning for Maintenance Decision-Making](https://arxiv.org/abs/2109.15050)
  - Hamed Khorasgani, Haiyan Wang, Chetan Gupta, Ahmed Farahat
  - Publisher: Annual Conference of the PHM Society 2021
  - Key:  Offline Supervised RL, Remaining Useful Life Estimation
  - ExpEnv:  [NASA C-MAPSS](https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq)

- [A Sequence Modelling Approach to Question Answering in Text-Based Games](https://aclanthology.org/2022.wordplay-1.4/)
  - Gregory Furman, Edan Toledo, Jonathan Shock, Jan Buys
  - Publisher: Proceedings of the 3rd Wordplay: When Language Meets Games Workshop (Wordplay 2022)
  - Key: VQA
  - ExpEnv:  [QAIT](https://github.com/xingdi-eric-yuan/qait_public)

- [Can Wikipedia Help Offline Reinforcement Learning?](https://arxiv.org/abs/2201.12122)
  - Machel Reid, Yutaro Yamada, Shixiang Shane Gu
  - Key: VLN, Transfer Learning
  - Code: [official](https://github.com/machelreid/can-wikipedia-help-offline-rl)
  - ExpEnv: [MuJoco](https://github.com/openai/mujoco-py), [D4RL](https://github.com/rail-berkeley/d4rl), [Atari](https://github.com/openai/gym)

- [Switch Trajectory Transformer with Distributional Value Approximation for Multi-Task Reinforcement Learning](https://arxiv.org/abs/2203.07413)
  - Qinjie Lin, Han Liu, Biswa Sengupta
  - Key: Multi-Task RL, Sparse Reward
  - ExpEnv: [MINIGRID](https://github.com/Farama-Foundation/gym-minigrid)

- [Bootstrapped Transformer for Offline Reinforcement Learning](https://arxiv.org/abs/2206.08569)
  - Kerong Wang, Hanye Zhao, Xufang Luo, Kan Ren, Weinan Zhang, Dongsheng Li
  - Key:  Generation model
  - Code: [official](https://seqml.github.io/bootorl)
  - ExpEnv: [MuJoco](https://github.com/openai/mujoco-py), [D4RL](https://github.com/rail-berkeley/d4rl), [Adroit](https://github.com/aravindr93/hand_dapg)

- [Deep Transformer Q-Networks for Partially Observable Reinforcement Learning](https://arxiv.org/abs/2206.01078)
  - Kevin Esslinger, Robert Platt, Christopher Amato
  - Key: POMDP, Transformer Q-Learning
  - ExpEnv: [GV](https://github.com/abaisero/gym-gridverse), [Car Flag](https://github.com/hai-h-nguyen/pomdp-domains)

- [Multi-Agent Reinforcement Learning is a Sequence Modeling Problem](https://arxiv.org/abs/2205.14953)
  - Muning Wen, Jakub Grudzien Kuba, Runji Lin, Weinan Zhang, Ying Wen, Jun Wang, Yaodong Yang
  - Key: Multi-Agent RL
  - ExpEnv: [SMAC](https://github.com/oxwhirl/smac), [MA MuJoco](https://github.com/schroederdewitt/multiagent_mujoco)

- [Transformers are Adaptable Task Planners](https://arxiv.org/abs/2207.02442)
  - Vidhi Jain, Yixin Lin, Eric Undersander, Yonatan Bisk, Akshara Rai
  - Key: Task Planning, Prompt, Control, Generalization
  - Code: [official](https://anonymous.4open.science/r/temporal_task_planner-Paper148/README.md)
  - ExpEnv: Dishwasher Loading

- [You Can't Count on Luck: Why Decision Transformers Fail in Stochastic Environments](https://arxiv.org/abs/2205.15967)
  - Keiran Paster, Sheila McIlraith, Jimmy Ba
  - Key: Stochastic Environments
  - ExpEnv: Gambling, Connect Four, [2048](https://github.com/FelipeMarcelino/2048-Gym)

- [When does return-conditioned supervised learning work for offline reinforcement learning?](https://arxiv.org/abs/2206.01079)
  - David Brandfonbrener, Alberto Bietti, Jacob Buckman, Romain Laroche, Joan Bruna
  - Key: Theoretical analysis
  - ExpEnv: [MuJoco](https://github.com/openai/mujoco-py), [D4RL](https://github.com/rail-berkeley/d4rl)

- [SimStu-Transformer: A Transformer-Based Approach to Simulating Student Behaviour](https://link.springer.com/chapter/10.1007/978-3-031-11647-6_67)
    - Zhaoxing Li, Lei Shi, Alexandra Cristea, Yunzhan Zhou, Chenghao Xiao, Ziqi Pan
    - Key: Intelligent Tutoring System

- [Attention-Based Learning for Combinatorial Optimization](https://dspace.mit.edu/bitstream/handle/1721.1/144893/Smith-smithcj-meng-eecs-2022-thesis.pdf?sequence=1&isAllowed=y)
    - Carson Smith
    - Key: Combinatorial Optimization
    
### ICLR 2023

- [EDGI: Equivariant Diffusion for Planning with Embodied Agents](https://arxiv.org/abs/2303.12410)
  - Johann Brehmer, Joey Bose, Pim de Haan, Taco Cohen
  - Publisher: ICLR 2023 Reincarnating RL workshop
  - Key: rich geometric structure, equivariant, conditional generative modeling, representation
  - ExpEnv: None 
    

### Neurips 2022

- [Decision making as language generation](https://openreview.net/pdf?id=N47cSU036T)
  - Roland Memisevic, Sunny Panchal, Mingu Lee
  - Publisher:  NeurIPS 2022 Workshop FMDM
  - Key: Generation
  - ExpEnv: Traversals (Toy experiment)

### CoRL 2022

- [Offline Reinforcement Learning for Customizable Visual Navigation](https://openreview.net/forum?id=uhIfIEIiWm_)
  - Dhruv Shah, Arjun Bhorkar, Hrishit Leen, Ilya Kostrikov, Nicholas Rhinehart, Sergey Levine
  - Publisher:  CoRL 2022 (Oral)
  - Key: Visual Navigation
  - ExpEnv: [RECON](https://sites.google.com/view/recon-robot/)


- [Instruction-driven history-aware policies for robotic manipulations](https://arxiv.org/abs/2209.04899)
  - Pierre-Louis Guhur, Shizhe Chen, Ricardo Garcia, Makarand Tapaswi, Ivan Laptev, Cordelia Schmid
  - Publisher:  CoRL 2022 (Oral)
  - Key: Robotics, Language Instruction
  - Code: [official](https://guhur.github.io/hiveformer/)
  - ExpEnv: [RLBench](https://github.com/stepjam/RLBench/)

- [Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation](https://arxiv.org/abs/2209.05451)
  - Mohit Shridhar, Lucas Manuelli, Dieter Fox
  - Publisher:  CoRL 2022
  - Key: Robotics,  Language Grounding, Behavior Cloning
  - Code: [official](https://guhur.github.io/hiveformer/)
  - ExpEnv: [RLBench](https://github.com/stepjam/RLBench/)

### ICML 2022

- [Online Decision Transformer](https://arxiv.org/abs/2202.05607)
  - Qinqing Zheng, Amy Zhang, Aditya Grover
  - Publisher:  ICML 2022 (Oral)
  - Key: Online finetuning,  Max-entropy, Exploration
  - Code: [unofficial](https://github.com/daniellawson9999/online-decision-transformer)
  - ExpEnv: [MuJoco](https://github.com/openai/mujoco-py), [D4RL](https://github.com/rail-berkeley/d4rl)

- [Prompting Decision Transformer for Few-Shot Policy Generalization](https://arxiv.org/abs/2206.13499)
  - Mengdi Xu, Yikang Shen, Shun Zhang, Yuchen Lu, Ding Zhao, Joshua B. Tenenbaum, Chuang Gan
  - Publisher:  ICML 2022 (Poster)
  - Key: Prompt, Few-shot, Generalization
  - Code: [official](https://mxu34.github.io/PromptDT/) (released soon)
  - ExpEnv: [DMC](https://github.com/deepmind/dm_control)

- [Addressing Optimism Bias in Sequence Modeling for Reinforcement Learning](https://proceedings.mlr.press/v162/villaflor22a.html)
  - Adam R Villaflor, Zhe Huang, Swapnil Pande, John M Dolan, Jeff Schneider
  - Publisher:  ICML 2022 (Poster)
  - Key: World model
  - Code: [official](https://mxu34.github.io/PromptDT/) (released soon)
  - ExpEnv: [CARLA](https://leaderboard.carla.org/)

- [AnyMorph: Learning Transferable Polices By Inferring Agent Morphology](https://arxiv.org/abs/2206.12279)
  - Brandon Trabucco, Mariano Phielipp, Glen Berseth
  - Publisher: ICML 2022 (Poster)
  - Key: Morphology, Transfer Learning, Zero Shot
  - ExpEnv: [Modular-RL](https://github.com/huangwl18/modular-rl)

### AAAI 2022

- [Dreaming with Transformers](http://aaai-rlg.mlanctot.info/papers/AAAI22-RLG_paper_24.pdf)
  - Catherine Zeng, Jordan Docter, Alexander Amini, Igor Gilitschenski, Ramin Hasani, Daniela Rus
  - Publisher: AAAI 2022 (RLG Workshop)
  - Key: Dreamer, World Model
  - ExpEnv: [Deepmind Lab](https://github.com/deepmind/lab), [VISTA](https://github.com/vista-simulator/vista)

### ICLR 2022

- [Learning Transferable Policies By Inferring Agent Morphology](https://openreview.net/forum?id=HE3NA4aNJbq)
  - Brandon Trabucco, Mariano Phielipp, Glen Berseth
  - Publisher: ICLR 2022 (GPL Workshop Poster)
  - Key: Morphology, Transfer Learning, Zero Shot
  - ExpEnv: [Modular-RL](https://github.com/huangwl18/modular-rl)

- [Silver-Bullet-3D at ManiSkill 2021: Learning-from-Demonstrations and Heuristic Rule-based Methods for Object Manipulation](https://arxiv.org/abs/2206.06289)
  - Yingwei Pan, Yehao Li, Yiheng Zhang, Qi Cai, Fuchen Long, Zhaofan Qiu, Ting Yao, Tao Mei
  - Publisher: ICLR 2022 (GPL Workshop Poster)
  - Key: Object Manipulation
  - Code: [official](https://github.com/caiqi/Silver-Bullet-3D/)
  - ExpEnv: [ManiSkill](https://github.com/haosulab/ManiSkill)

- [Generalized Decision Transformer for Offline Hindsight Information Matching](https://arxiv.org/abs/2111.10364)
  - Hiroki Furuta, Yutaka Matsuo, Shixiang Shane Gu
  - Publisher: ICLR 2021 (Spotlight)
  - Key: HIM, SMM
  - Code: [official](https://github.com/frt03/generalized_dt)
  - ExpEnv: [MuJoco](https://github.com/openai/mujoco-py), [D4RL](https://github.com/rail-berkeley/d4rl)

### NeurIPS 2021

- [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)

  - Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch
  - Publisher: NeurIPS 2021 (Poster)
  - Key: Conditional sequence modeling
  - Code: [official](https://github.com/kzl/decision-transformer), [DI-engine](https://github.com/opendilab/DI-engine/blob/main/dizoo/d4rl/entry/d4rl_dt_main.py)
  - ExpEnv: [MuJoco](https://github.com/openai/mujoco-py), [D4RL](https://github.com/rail-berkeley/d4rl), [Atari](https://github.com/openai/gym)

- [Offline Reinforcement Learning as One Big Sequence Modeling Problem](https://arxiv.org/abs/2106.02039)
  - Michael Janner, Qiyang Li, Sergey Levine
  - Publisher: NeurIPS 2021 (Spotlight)
  - Key: Conditional sequence modeling, Discretization
  - Code: [official](https://github.com/JannerM/trajectory-transformer)
  - ExpEnv: [MuJoco](https://github.com/openai/mujoco-py), [D4RL](https://github.com/rail-berkeley/d4rl)

- [TransDreamer: Reinforcement Learning with Transformer World Models](https://arxiv.org/abs/2202.09481)
  - Chang Chen, Yi-Fu Wu, Jaesik Yoon, Sungjin Ahn
  - Publisher: NeurIPS 2021 (Deep RL Workshop)
  - Key: Dreamer, World Model
  - ExpEnv: Hidden Order Discovery, [DMC](https://github.com/deepmind/dm_control), [Atari](https://github.com/openai/gym)

### ICML 2021

- [Reinforcement learning as one big sequence modeling problem](https://arxiv.org/abs/2106.02039v1)
  - Michael Janner, Qiyang Li, Sergey Levine
  - Publisher: ICML workshop
  - Key: Conditional sequence modeling, Discretization
  - Code: [official](https://github.com/JannerM/trajectory-transformer)
  - ExpEnv: [MuJoco](https://github.com/openai/mujoco-py), [D4RL](https://github.com/rail-berkeley/d4rl)

## Contributing

Our purpose is to make this repo even better. If you are interested in contributing, please refer to [HERE](CONTRIBUTING.md) for instructions in contribution.

## License

Awesome Decision Transformer is released under the Apache 2.0 license.
