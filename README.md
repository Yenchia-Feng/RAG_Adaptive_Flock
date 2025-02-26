# RAG_Adaptive_Flock
Python simulation where a flock of birds adapts its movement using retrieval-augmented generation. Instead of relying solely on fixed rules, birds retrieve examples of past behaviors from a vector database to evade predators.

# Boid-Predator Simulation

This project simulates flocking behavior with predator interaction using boids algorithm and RAG (Retrieval-Augmented Generation).


## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

4. Run the simulation:
```bash
python synthetic_data_generation.py
```
```bash
python baseline_flock_animation.py
```
```bash
python adaptation.py
```

## Requirements

- Python 3.8+
- OpenAI API key
- FFmpeg (for video saving)

See `requirements.txt` for full Python package requirements.

## Project Structure

- `strategy_animation.py`: Main simulation script
- `simulation.py`: Boid and flock behavior implementation
- `synthetic_data_generation.py`: Data generation for RAG
- `flock_data/`: Directory containing training data
- `vector_stores/`: Directory containing ChromaDB collections
