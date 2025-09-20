# Enhancing Interpretability in Deep Reinforcement Learning through Semantic Clustering

**Authors:** Liang Zhang, Justin Lieffers, Adarsh Pyarelal  
**Conference:** NeurIPS 2025 Main Track

This repository contains the official implementation of our research on enhancing interpretability in deep reinforcement learning through semantic clustering techniques. Our work extends the [OpenAI train-procgen](https://github.com/openai/train-procgen) framework to incorporate semantic clustering methods for improved understanding and visualization of learned policies in procedural environments.

## 📋 Abstract

This work presents a novel approach to enhancing interpretability in deep reinforcement learning by leveraging semantic clustering techniques. We demonstrate how semantic clustering can provide insights into learned policies, enabling better understanding of agent behavior and decision-making processes in complex procedural environments.

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ualiangzhang/semantic_rl.git
   cd semantic_rl
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Procgen environments:**
   Follow the installation steps in the [Procgen repository](https://github.com/openai/procgen).

### Basic Usage

**Train a semantic clustering model:**
```bash
python -m train_procgen.train_sppo --env_name <ENV_NAME> --num_levels 0 --distribution_mode easy --timesteps_per_proc 25000000 --rand_seed <RAND_SEED>
```

**Train a baseline model:**
```bash
python -m train_procgen.train_ppo --env_name <ENV_NAME> --num_levels 0 --distribution_mode easy --timesteps_per_proc 25000000 --rand_seed <RAND_SEED>
```

## 📊 Visualization and Analysis

### Performance Analysis

**Generate generalization figures for a single game:**
```bash
cd train_procgen
python single_graph.py --env_name <ENV_NAME>
# Example:
python single_graph.py --env_name coinrun
```

**Generate generalization figures for all games:**
```bash
cd train_procgen
python graph.py
```

### Semantic Clustering Visualization

**Generate embedding space visualizations:**
```bash
python -m train_procgen.enjoy_sppo --env_name <ENV_NAME> --mode 1
```

**Generate skill demonstration videos:**
```bash
python -m train_procgen.enjoy_sppo --env_name <ENV_NAME> --mode 0
```

**Interactive cluster exploration:**
```bash
python -m train_procgen.hover_clusters --env_name <ENV_NAME>
# Example:
python -m train_procgen.hover_clusters --env_name fruitbot
```

## 🎮 Supported Environments

Our implementation supports four Procgen environments:
- **CoinRun** - Platformer with coin collection
- **FruitBot** - Fruit collection game
- **Jumper** - Jumping-based platformer
- **Ninja** - Action-based ninja game

## 🎬 Semantic Clustering Demonstration

### Ninja Environment - 8 Semantic Clusters

The following videos demonstrate the 8 distinct semantic clusters learned by our model in the Ninja environment. Each cluster represents a different behavioral pattern and skill set:

#### 📹 Video Demonstrations

| Cluster | Video File | Description |
|---------|------------|-------------|
| **Cluster 0** | [cluster_0.avi](videos/ninja_skills/cluster_0.avi) | Combat strategy and movement pattern 1 |
| **Cluster 1** | [cluster_1.avi](videos/ninja_skills/cluster_1.avi) | Combat strategy and movement pattern 2 |
| **Cluster 2** | [cluster_2.avi](videos/ninja_skills/cluster_2.avi) | Combat strategy and movement pattern 3 |
| **Cluster 3** | [cluster_3.avi](videos/ninja_skills/cluster_3.avi) | Combat strategy and movement pattern 4 |
| **Cluster 4** | [cluster_4.avi](videos/ninja_skills/cluster_4.avi) | Combat strategy and movement pattern 5 |
| **Cluster 5** | [cluster_5.avi](videos/ninja_skills/cluster_5.avi) | Combat strategy and movement pattern 6 |
| **Cluster 6** | [cluster_6.avi](videos/ninja_skills/cluster_6.avi) | Combat strategy and movement pattern 7 |
| **Cluster 7** | [cluster_7.avi](videos/ninja_skills/cluster_7.avi) | Combat strategy and movement pattern 8 |

#### 🎯 How to View the Videos

1. **Download the repository:**
   ```bash
   git clone https://github.com/ualiangzhang/semantic_rl.git
   cd semantic_rl
   ```

2. **Navigate to the videos directory:**
   ```bash
   cd videos/ninja_skills/
   ```

3. **Play the videos:**
   - **Windows:** Double-click any `.avi` file to play in your default video player
   - **macOS:** Double-click any `.avi` file to play in QuickTime Player
   - **Linux:** Use VLC, MPV, or your preferred video player

#### 📊 Alternative: Generate Your Own Videos

You can also generate these videos yourself using our code:

```bash
# Generate Ninja skill cluster videos
python -m train_procgen.enjoy_sppo --env_name ninja --mode 0 --num_embeddings 8
```

**Note:** These videos showcase the distinct behavioral patterns learned by our semantic clustering approach. Each cluster demonstrates different combat strategies, movement patterns, and decision-making processes in the Ninja environment.

## 📁 Output Structure

```
baseline/                # Required RL training package
train_procgen/
├── checkpoints/         # Trained model checkpoints
├── figures/             # Generated visualizations and videos
videos/                  # video clips corresponding to the clusters in the paper             
```


## 📈 Reproducing Results

To reproduce the results from our paper:

1. **Train models** using the commands above
2. **Generate visualizations** using the provided scripts
3. **Analyze results** using the interactive tools

**Note:** Video generation may take 30-60 minutes depending on machine performance, as it ensures comprehensive exploration of all clusters.

<!-- ## 🤝 Citation

If you use this code in your research, please cite our paper: -->

<!-- ```bibtex
@article{zhang2025enhancing,
  title={Enhancing Interpretability in Deep Reinforcement Learning through Semantic Clustering},
  author={Zhang, Liang and Lieffers, Justin and Pyarelal, Adarsh},
  journal={Advances in Neural Information Processing Systems},
  year={2025}
}
``` -->

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This work builds upon the [OpenAI train-procgen](https://github.com/openai/train-procgen) framework. We thank the original authors for their excellent work on procedural generation for reinforcement learning benchmarking.