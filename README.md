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
   git clone https://github.com/your-username/semantic_rl.git
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

Our implementation supports all Procgen environments:
- **CoinRun** - Platformer with coin collection
- **FruitBot** - Fruit collection game
- **Jumper** - Jumping-based platformer
- **Ninja** - Action-based ninja game
- **StarPilot** - Space exploration game
- And more...

## 📁 Output Structure

```
train_procgen/
├── checkpoints/          # Trained model checkpoints
├── figures/             # Generated visualizations and videos
└── results/             # Evaluation results and metrics
```

## 🔬 Research Features

- **Semantic Clustering:** Advanced clustering techniques for policy interpretation
- **Interactive Visualization:** Hover-based cluster exploration
- **Skill Decomposition:** Automatic identification and visualization of learned skills
- **Generalization Analysis:** Comprehensive evaluation across different difficulty levels

## 📈 Reproducing Results

To reproduce the results from our paper:

1. **Train models** using the commands above
2. **Generate visualizations** using the provided scripts
3. **Analyze results** using the interactive tools

**Note:** Video generation may take 30-60 minutes depending on machine performance, as it ensures comprehensive exploration of all clusters.

## 🤝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{zhang2025enhancing,
  title={Enhancing Interpretability in Deep Reinforcement Learning through Semantic Clustering},
  author={Zhang, Liang and Lieffers, Justin and Pyarelal, Adarsh},
  journal={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This work builds upon the [OpenAI train-procgen](https://github.com/openai/train-procgen) framework. We thank the original authors for their excellent work on procedural generation for reinforcement learning benchmarking.

## 📞 Contact

For questions or issues, please contact:
- Liang Zhang: [email]
- Justin Lieffers: [email]  
- Adarsh Pyarelal: [email]