# Awesome Decision Transformer
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)  
![visitor badge](https://visitor-badge.lithub.cc/badge?page_id=opendilab.awesome-decision-transformer&left_text=Visitors) 
![GitHub stars](https://img.shields.io/github/stars/opendilab/awesome-decision-transformer?color=yellow) 
![GitHub forks](https://img.shields.io/github/forks/opendilab/awesome-decision-transformer?color=9cf) 
[![GitHub license](https://img.shields.io/github/license/opendilab/awesome-decision-transformer)](https://github.com/opendilab/awesome-decision-transformer/blob/main/LICENSE)



This is a collection of research papers for **Decision Transformer (DT)**.
And the repository will be continuously updated to track the frontier of DT.

Welcome to follow and star!

## Table of Contents

- [A Taxonomy of DT Algorithms](#a-taxonomy-of-decision-transformer-algorithms)
- [Surveys](#surveys)
- [Papers](#papers)

  - [Arxiv](#arxiv)
  - [ICLR 2025](#iclr-2025) (**<font color="red">New!!!</font>**)
  - [NeurIPS 2024](#neurips-2024) 
  - [IROS 2024](#iros-2024)
  - [ICML 2024](#icml-2024) 
  - [ICLR 2024](#iclr-2024) 
  - [NeurIPS 2023](#neurips-2023)
  - [CoRL 2023](#corl-2023)
  - [IROS 2023](#iros-2023)
  - [ICML 2023](#icml-2023)
  - [ICRA 2023](#icra-2023)
  - [ICLR 2023](#iclr-2023)
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

## Surveys

- [On Transforming Reinforcement Learning With Transformers: The Development Trajectory](https://ieeexplore.ieee.org/abstract/document/10546317)
  - Shengchao Hu, Li Shen, Ya Zhang, Yixin Chen, Dacheng Tao
  - Publisher: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)

- [Large sequence models for sequential decision-making: a survey](https://link.springer.com/article/10.1007/s11704-023-2689-5)
  - Muning Wen, Runji Lin, Hanjing Wang, Yaodong Yang, Ying Wen, Luo Mai, Jun Wang, Haifeng Zhang, Weinan Zhang
  - Publisher: Frontiers of Computer Science

- [A Survey on Transformers in Reinforcement Learning](https://arxiv.org/abs/2301.03044)
  - Wenzhe Li, Hao Luo, Zichuan Lin, Chongjie Zhang, Zongqing Lu, Deheng Ye
  - Publisher: Transactions on Machine Learning Research (TMLR)

- [Transformers in Reinforcement Learning: A Survey](https://arxiv.org/abs/2307.05979)
  - Pranav Agarwal, Aamer Abdul Rahman, Pierre-Luc St-Charles, Simon J.D. Prince, Samira Ebrahimi Kahou


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
- [Context-Former: Stitching via Latent Conditioned Sequence Modeling ](https://browse.arxiv.org/abs/2401.16452)
  - Ziqi Zhang, Jingzehua Xu, Zifeng Zhuang, Jinxin Liu, Donglin wang
  - Key: DT, Latent Conditioned Sequence Modeling
  - ExpEnv: [D4RL](https://github.com/rail-berkeley/d4rl)

- [Real-time Network Intrusion Detection via Decision Transformers](https://arxiv.org/abs/2312.07696)
  - Jingdi Chen, Hanhan Zhou, Yongsheng Mei, Gina Adam, Nathaniel D. Bastian, Tian Lan
  - Key: DT, Network Intrusion Detection
  - ExpEnv: UNSW-NB15

- [Prompt-Tuning Decision Transformer with Preference Ranking](https://arxiv.org/abs/2305.09648)
  - Shengchao Hu, Li Shen, Ya Zhang, Dacheng Tao
  - Key: Prompt-Tuning
  - ExpEnv: [D4RL](https://github.com/rail-berkeley/d4rl)

- [Graph Decision Transformer](https://arxiv.org/abs/2303.03747)
  - Shengchao Hu, Li Shen, Ya Zhang, Dacheng Tao
  - Key: graph transformer
  - ExpEnv: [Atari](https://github.com/openai/gym)


- [Can Offline Reinforcement Learning Help Natural Language Understanding?](https://arxiv.org/abs/2212.03864)
  - Ziqi Zhang, Yile Wang, Yue Zhang, Donglin Wang
  - Key: Language model
  - ExpEnv: MuJoco, Maze 2D

- [SaFormer: A Conditional Sequence Modeling Approach to Offline Safe Reinforcement Learning](https://arxiv.org/abs/2301.12203)
  - Qin Zhang, Linrui Zhang, Haoran Xu, Li Shen, Bowen Wang, Yongzhe Chang, Xueqian Wang, Bo Yuan, Dacheng Tao
  - Key: Offline Safe RL, DT
  - ExpEnv: [D4RL](https://github.com/rail-berkeley/d4rl)

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


- [Deep Transformer Q-Networks for Partially Observable Reinforcement Learning](https://arxiv.org/abs/2206.01078)
  - Kevin Esslinger, Robert Platt, Christopher Amato
  - Key: POMDP, Transformer Q-Learning
  - ExpEnv: [GV](https://github.com/abaisero/gym-gridverse), [Car Flag](https://github.com/hai-h-nguyen/pomdp-domains)


- [SimStu-Transformer: A Transformer-Based Approach to Simulating Student Behaviour](https://link.springer.com/chapter/10.1007/978-3-031-11647-6_67)
    - Zhaoxing Li, Lei Shi, Alexandra Cristea, Yunzhan Zhou, Chenghao Xiao, Ziqi Pan
    - Key: Intelligent Tutoring System

- [Attention-Based Learning for Combinatorial Optimization](https://dspace.mit.edu/bitstream/handle/1721.1/144893/Smith-smithcj-meng-eecs-2022-thesis.pdf?sequence=1&isAllowed=y)
    - Carson Smith
    - Key: Combinatorial Optimization

### ICLR 2025
- [Long-Short Decision Transformer: Bridging Global and Local Dependencies for Generalized Decision-Making](https://openreview.net/forum?id=NHMuM84tRT)
  - Jincheng Wang, Penny Karanasou, Pengyuan Wei, Elia Gatti, Diego Martinez Plasencia, Dimitrios Kanoulas
  - Key: Deep Learning, Reinforcement Learning, Transformer, Decision Transformer, Long-Short Decision Transformer, OfflineRL
  - ExpEnv: D4RL offline RL benchmark, Maze2d, Antmaze

### NeurIPS 2024
- [Adaptive Q-Aid for Conditional Supervised Learning in Offline Reinforcement Learning](https://arxiv.org/pdf/2402.02017)
  - Jeonghye Kim, Suyoung Lee, Woojun Kim, Youngchul Sung
  - Keyword: Q-learning, DT
  - ExpEnv: [MuJoco](https://github.com/openai/mujoco-py), AntMaze, Adroit
    
- [Meta-DT: Offline Meta-RL as Conditional Sequence Modeling with World Model Disentanglement](https://arxiv.org/pdf/2410.11448)
  - Zhi Wang, Li Zhang, Wenhao Wu, Yuanheng Zhu, Dongbin Zhao, Chunlin Chen
  - Keyword: Offline Meta-Reinforcement Learning, Transformer, World Model Disentanglement
  - ExpEnv: MuJoCo, Meta-World

- [Decomposed Prompt Decision Transformer for Efficient Unseen Task Generalization](https://openreview.net/pdf?id=HcqnhqoXS3)
  - Hongling Zheng, Li Shen, Yong Luo, Tongliang Liu, Jialie Shen, Dacheng Tao
  - Publisher: NeurIPS 2024
  - Key: Offline Reinforcement Learning, Prompt Tuning
  - ExpEnv: [MuJoco](https://github.com/openai/mujoco-py)

### IROS 2024
- [Steering Decision Transformers via Temporal Difference Learning](https://cpsl.pratt.duke.edu/files/docs/d2t2.pdf)
  - Hao-Lun Hsu, Alper Kamil Bozkurt, Juncheng Dong, Qitong Gao, Vahid Tarokh, Miroslav Pajic
  - Publisher: IROS 2024
  - Key: Robotics, offline reinforcement learning, sequence modeling
  - ExpEnv: [D4RL](https://github.com/rail-berkeley/d4rl), Robotic Manipulation, [CARLA](https://leaderboard.carla.org/)

### ICML 2024

- [Generalization to New Sequential Decision Making Tasks with In-Context Learning](https://arxiv.org/abs/2312.03801)
  - Sharath Chandra Raparthy, Eric Hambro, Robert Kirk, Mikael Henaff, Roberta Raileanu
  - Publisher: ICML 2024
  - Key: DT, In-Context Learning
  - ExpEnv: [MiniHack](), [Procgen]()

- [HarmoDT: Harmony Multi-Task Decision Transformer for Offline Reinforcement Learning](https://arxiv.org/abs/2405.18080)
  - Shengchao Hu, Ziqing Fan, Li Shen, Ya Zhang, Yanfeng Wang, Dacheng Tao
  - Publisher: ICML 2024
  - Key: Multi-task, DT
  - ExpEnv: [MetaWorld](https://github.com/Farama-Foundation/Metaworld)

- [Q-value Regularized Transformer for Offline Reinforcement Learning](https://arxiv.org/abs/2405.17098)
  - Shengchao Hu, Ziqing Fan, Chaoqin Huang, Li Shen, Ya Zhang, Yanfeng Wang, Dacheng Tao
  - Publisher: ICML 2024
  - Key: Q-learning, DT
  - ExpEnv: [D4RL](https://github.com/rail-berkeley/d4rl)

- [Temporal Logic Specification-Conditioned Decision Transformer for Offline Safe Reinforcement Learning](https://arxiv.org/abs/2402.17217)
  - Zijian Guo, Weichao Zhou, Wenchao Li
  - Publisher: ICML 2024
  - Key: Signal Temporal Logic (STL), DT
  - ExpEnv: [DSRL](https://github.com/liuzuxin/DSRL)

- [Think Before You Act: Decision Transformers with Working Memory](https://openreview.net/forum?id=PSQ5Z920M8)
  - Jikun Kang, Romain Laroche, Xingdi Yuan, Adam Trischler, Xue Liu, Jie Fu
  - Publisher: ICML 2024
  - Key: Working Memory, DT
  - ExpEnv: [MetaWorld](https://github.com/Farama-Foundation/Metaworld), [Atari](https://github.com/openai/gym)

- [In-Context Decision Transformer: Reinforcement Learning via Hierarchical Chain-of-Thought](https://arxiv.org/abs/2405.20692)
  - Sili Huang, Jifeng Hu, Hechang Chen, Lichao Sun, Bo Yang
  - Publisher: ICML 2024
  - Key: Hierarchical Structure, DT
  - ExpEnv: [D4RL](https://github.com/rail-berkeley/d4rl)

- [Rethinking Decision Transformer via Hierarchical Reinforcement Learning](https://arxiv.org/abs/2311.00267)
  - Yi Ma, Jianye Hao, Hebin Liang, Chenjun Xiao,
  - Publisher: ICML 2024
  - Key: DT, Hierarchical Reinforcement Learning 
  - ExpEnv: [D4RL](https://github.com/rail-berkeley/d4rl)

### ICLR 2024
- [Learning Multi-Agent Communication from Graph Modeling Perspective](https://arxiv.org/abs/2405.08550)
  - Shengchao Hu, Li Shen, Ya Zhang, Dacheng Tao
  - Publisher: ICLR 2024
  - Key: Communication, Sequence Modeling
  - ExpEnv: [SMAC](https://github.com/oxwhirl/smac)

- [Decision ConvFormer: Local Filtering in MetaFormer is Sufficient for Decision Making ](https://arxiv.org/abs/2310.03022)
  - Jeonghye Kim, Suyoung Lee, Woojun Kim, Youngchul Sung
  - Key: MetaFormer, Decision ConvFormer
  - ExpEnv: [D4RL](https://github.com/rail-berkeley/d4rl), [Atari](https://github.com/openai/gym)

- [When should we prefer Decision Transformers for Offline Reinforcement Learning?](https://openreview.net/pdf?id=vpV7fOFQy4)
  - Prajjwal Bhargava, Rohan Chitnis, Alborz Geramifard, Shagun Sodhani, Amy Zhang
  - Key: offline reinforcement learning, sequence modeling, reinforcement learning
  - ExpEnv: [D4RL](https://github.com/rail-berkeley/d4rl)

- [Transformers as Decision Makers: Provable In-Context Reinforcement Learning via Supervised Pretraining](https://openreview.net/pdf?id=yN4Wv17ss3)
  - Licong Lin, Yu Bai, Song Mei
  - Key: transformers, in-context learning, reinforcement learning, learning theory
  - ExpEnv: [stochastic linear bandit]()

- [Searching for High-Value Molecules Using Reinforcement Learning and Transformers](https://openreview.net/pdf?id=nqlymMx42E)
  - Raj Ghugare, Santiago Miret, Adriana Hugessen, Mariano Phielipp, Glen Berseth
  - Key: chemistry, reinforcement learning, language models
  - ExpEnv: [docking and pytdc tasks]()


### NeurIPS 2023

- [SwiftSage: A Generative Agent with Fast and Slow Thinking for Complex Interactive Tasks](https://arxiv.org/abs/2303.03982)
  - Bill Yuchen Lin, Yicheng Fu, Karina Yang, Faeze Brahman, Shiyu Huang, Chandra Bhagavatula, Prithviraj Ammanabrolu, Yejin Choi, Xiang Ren
  - Publisher: NeurIPS 2023
  - Key: Dual-Process Theory
  - ExpEnv: [ScienceWorld]()

- [HIQL: Offline Goal-Conditioned RL with Latent States as Actions](https://arxiv.org/abs/2303.03982)
  - Seohong Park, Dibya Ghosh, Benjamin Eysenbach, Sergey Levine
  - Publisher: NeurIPS 2023
  - Key: Hierarchical Goal-Conditioned RL, Offline Reinforcement Learning, Value Function Estimation
  - ExpEnv: [D4RL](https://github.com/rail-berkeley/d4rl)

- [Structured state space models for in-context reinforcement learning](https://arxiv.org/abs/2303.03982)
  - Chris Lu, Yannick Schroecker, Albert Gu, Emilio Parisotto, Jakob Foerster, Satinder Singh, Feryal Behbahani
  - Publisher: NeurIPS 2023
  - Key: state space model, in-context RL
  - ExpEnv: [t-maze]()

- [Supervised Pretraining Can Learn In-Context Reinforcement Learning](https://arxiv.org/abs/2306.14892)
  - Jonathan N. Lee, Annie Xie, Aldo Pacchiano, Yash Chandak, Chelsea Finn, Ofir Nachum, Emma Brunskill
  - Publisher: NeurIPS 2023
  - Key: Decision Pretrained Transformer, in-context learning
  - ExpEnv: [Dark Room]()

- [Is Feedback All You Need? Leveraging Natural Language Feedback in Goal-Conditioned Reinforcement Learning](https://arxiv.org/abs/2312.04736)
  - Sabrina McCallum, Max Taylor-Davies, Stefano V. Albrecht, Alessandro Suglia
  - Publisher: NeurIPS 2023 Workshop
  - Key: DT, language feedback
  - ExpEnv: [BabyAI](https://github.com/mila-iqia/babyai/tree/iclr19)

- [STEVE-1: A Generative Model for Text-to-Behavior in Minecraft](https://arxiv.org/pdf/2306.00937)
  - Shalev Lifshitz, Keiran Paster, Harris Chan, Jimmy Ba, Sheila McIlraith
  - Publisher: NeurIPS 2023
  - Key: instruction-tuned Video Pretraining
  - ExpEnv: [Minecraft]()

- [Transformers as Statisticians: Provable In-Context Learning with In-Context Algorithm Selection](https://proceedings.neurips.cc/paper_files/paper/2023/file/b2e63e36c57e153b9015fece2352a9f9-Paper-Conference.pdf)
  - Yu Bai, Fan Chen, Huan Wang, Caiming Xiong, Song Mei
  - Publisher: NeurIPS 2023
  - Key: in-context learning, transformers, deep learning theory, learning theory
  - ExpEnv: [in-context regression problems]()

- [Elastic Decision Transformer](https://arxiv.org/abs/2307.02484)
  - Yueh-Hua Wu, Xiaolong Wang, Masashi Hamaya
  - Publisher: NeurIPS 2023
  - Key: Offline RL, stitch trajectory, Multi-Task
  - ExpEnv: [D4RL](https://github.com/rail-berkeley/d4rl)

- [Learning to Modulate pre-trained Models in RL](https://arxiv.org/abs/2306.14884)
  - Thomas Schmied, Markus Hofmarcher, Fabian Paischer, Razvan Pascanu, Sepp Hochreiter
  - Publisher: NeurIPS 2023 (Poster)
  - Key: reinforcement learning, multi-task learning, continual learning, fine-tuning
  - ExpEnv: [MetaWorld](https://github.com/Farama-Foundation/Metaworld), [DMControl](https://github.com/google-deepmind/dm_control)

### CoRL 2023

- [Transformers are Adaptable Task Planners](https://arxiv.org/abs/2207.02442)
  - Vidhi Jain, Yixin Lin, Eric Undersander, Yonatan Bisk, Akshara Rai
  - Publisher: CoRL 2023
  - Key: Task Planning, Prompt, Control, Generalization
  - Code: [official](https://anonymous.4open.science/r/temporal_task_planner-Paper148/README.md)
  - ExpEnv: Dishwasher Loading

- [Q-Transformer](https://proceedings.mlr.press/v229/chebotar23a/chebotar23a.pdf)
  - Yevgen Chebotar, Quan Vuong, Alex Irpan, Karol Hausman, Fei Xia, Yao Lu, Aviral Kumar, Tianhe Yu, Alexander Herzog, Karl Pertsch, Keerthana Gopalakrishnan, Julian Ibarz, Ofir Nachum, Sumedh Sontakke, Grecia Salazar, Huong T Tran, Jodilyn Peralta, Clayton Tan, Deeksha Manjunath, Jaspiar Singht, Brianna Zitkovich, Tomas Jackson, Kanishka Rao, Chelsea Finn, Sergey Levine
  - Publisher: CoRL 2023
  - Key: Reinforcement Learning, Offline RL, Transformers, Q-Learning, Robotic Manipulation
  - Code: [Unofficial](https://github.com/lucidrains/q-transformer)
  - ExpEnv: None

### IROS 2023

- [Hierarchical Decision Transformer](https://arxiv.org/abs/2209.10447)
  - André Correia, Luís A. Alexandre
  - Publisher: IROS 2023
  - Key: Hierarchical Learning, Imitation Learning
  - ExpEnv: [MuJoco](https://github.com/openai/mujoco-py), [D4RL](https://github.com/rail-berkeley/d4rl), RoboMimic, Maze 2D

- [PACT: Perception-Action Causal Transformer for Autoregressive Robotics Pre-Training](https://arxiv.org/abs/2209.11133)
  - Rogerio Bonatti, Sai Vemprala, Shuang Ma, Felipe Frujeri, Shuhang Chen, Ashish Kapoor
  - Publisher: IROS 2023
  - Key: Robotics, Pretrain, Multitask, Representation
  - ExpEnv: MuSHR car, Habitat

### ICML 2023

- [Constrained Decision Transformer for Offline Safe Reinforcement Learning](https://proceedings.mlr.press/v202/liu23m.html)
  - Zuxin Liu, Zijian Guo, Yihang Yao, Zhepeng Cen, Wenhao Yu, Tingnan Zhang, Ding Zhao
  - Publisher: ICML 2023
  - Key: Offline Safe RL, DT
  - ExpEnv: [Bullet-Safety-Gym](https://github.com/SvenGronauer/Bullet-Safety-Gym)

- [Q-learning Decision Transformer: Leveraging Dynamic Programming for Conditional Sequence Modelling in Offline RL](https://arxiv.org/abs/2209.03993)
  - Taku Yamagata, Ahmed Khalil, Raul Santos-Rodriguez
  - Publisher: ICML 2023
  - Key: Q-Learning
  - ExpEnv: [D4RL](https://github.com/rail-berkeley/d4rl)

### ICRA 2023

- [LATTE: LAnguage Trajectory TransformEr](https://arxiv.org/abs/2208.02918)
  - Arthur Bucker, Luis Figueredo, Sami Haddadin, Ashish Kapoor, Shuang Ma, Sai Vemprala, Rogerio Bonatti
  - Publisher: ICRA 2023
  - Key: MultiModal,  Robotics
  - Code: [official](https://github.com/arthurfenderbucker/latte-language-trajectory-transformer), [official](https://github.com/arthurfenderbucker/nl_trajectory_reshaper)
  - ExpEnv: [CoppeliaSim](https://www.coppeliarobotics.com/)


### ICLR 2023

- [In-context Reinforcement Learning with Algorithm Distillation](https://arxiv.org/abs/2210.14215)
  - Michael Laskin, Luyu Wang, Junhyuk Oh, Emilio Parisotto, Stephen Spencer, Richie Steigerwald, DJ Strouse, Steven Stenberg Hansen, Angelos Filos, Ethan Brooks, maxime gazeau, Himanshu Sahni, Satinder Singh, Volodymyr Mnih
  - Publisher: ICLR 2023
  - Key: Reinforcement Learning, Transformers, Learning to Learn, Large Language Models
  - ExpEnv: [Adversarial Bandit](), [Dark Room](), [Dark Key-to-Door](), [DMLab Watermaze]()

- [EDGI: Equivariant Diffusion for Planning with Embodied Agents](https://arxiv.org/abs/2303.12410)
  - Johann Brehmer, Joey Bose, Pim de Haan, Taco Cohen
  - Publisher: ICLR 2023 Reincarnating RL workshop
  - Key: rich geometric structure, equivariant, conditional generative modeling, representation
  - ExpEnv: None 

- [Learning to Modulate pre-trained Models in RL](https://arxiv.org/abs/2306.14884)
  - Thomas Schmied, Markus Hofmarcher, Fabian Paischer, Razvan Pascanu, Sepp Hochreiter
  - Publisher: ICLR 2023 Reincarnating RL workshop
  - Key: reinforcement learning, multi-task learning, continual learning, fine-tuning
  - ExpEnv: [MetaWorld](https://github.com/Farama-Foundation/Metaworld), [DMControl](https://github.com/google-deepmind/dm_control)

- [DeFog: Decision Transformer under Random Frame Dropping](https://arxiv.org/abs/2303.03391)
  - Kaizhe Hu*, Ray Chen Zheng*, Yang Gao, Huazhe Xu
  - Publisher: ICLR 2023
  - Key: Offline RL, POMDP, Frame-Dropping, Practical Application
  - Code: [official](https://github.com/hukz18/DeFog)
  - ExpEnv: [MuJoco](https://github.com/openai/mujoco-py), [D4RL](https://github.com/rail-berkeley/d4rl), [Atari](https://github.com/openai/gym)

### NeurIPS 2022

- [When does return-conditioned supervised learning work for offline reinforcement learning?](https://arxiv.org/abs/2206.01079)
  - David Brandfonbrener, Alberto Bietti, Jacob Buckman, Romain Laroche, Joan Bruna
  - Publisher: NeurIPS 2022
  - Key: Theoretical analysis
  - ExpEnv: [MuJoco](https://github.com/openai/mujoco-py), [D4RL](https://github.com/rail-berkeley/d4rl)

- [You Can't Count on Luck: Why Decision Transformers Fail in Stochastic Environments](https://arxiv.org/abs/2205.15967)
  - Keiran Paster, Sheila McIlraith, Jimmy Ba
  - Publisher: NeurIPS 2022
  - Key: Stochastic Environments
  - ExpEnv: Gambling, Connect Four, [2048](https://github.com/FelipeMarcelino/2048-Gym)

- [Multi-Agent Reinforcement Learning is a Sequence Modeling Problem](https://arxiv.org/abs/2205.14953)
  - Muning Wen, Jakub Grudzien Kuba, Runji Lin, Weinan Zhang, Ying Wen, Jun Wang, Yaodong Yang
  - Publisher: NeurIPS 2022
  - Key: Multi-Agent RL
  - ExpEnv: [SMAC](https://github.com/oxwhirl/smac), [MA MuJoco](https://github.com/schroederdewitt/multiagent_mujoco)

- [Bootstrapped Transformer for Offline Reinforcement Learning](https://arxiv.org/abs/2206.08569)
  - Kerong Wang, Hanye Zhao, Xufang Luo, Kan Ren, Weinan Zhang, Dongsheng Li
  - Publisher: NeurIPS 2022
  - Key:  Generation model
  - Code: [official](https://seqml.github.io/bootorl)
  - ExpEnv: [MuJoco](https://github.com/openai/mujoco-py), [D4RL](https://github.com/rail-berkeley/d4rl), [Adroit](https://github.com/aravindr93/hand_dapg)

- [Multi-Game Decision Transformers](https://arxiv.org/abs/2205.15241)
  - Kuang-Huei Lee, Ofir Nachum, Mengjiao Yang, Lisa Lee, Daniel Freeman, Winnie Xu, Sergio Guadarrama, Ian Fischer, Eric Jang, Henryk Michalewski, Igor Mordatch
  - Publisher: NeurIPS 2022
  - Key: Multi-Task,  Finetuning
  - Code: [official](https://sites.google.com/view/multi-game-transformers)
  - ExpEnv: [Atari](https://github.com/openai/gym), [REM](https://github.com/google-research/batch_rl)

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
