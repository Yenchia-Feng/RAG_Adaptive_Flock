# Adaptive Flock RAG
**Adaptive Flock RAG** is a Python-based simulation that models the dynamic movement of a flock of birds using **Retrieval-Augmented Generation (RAG)**. In this simulation, instead of relying on predefined rules for flocking behavior, the birds learn to adapt their movements in real-time by retrieving past behavioral examples from a vector database, enabling them to avoid predators more effectively.

The core components of this simulation are:

- **Boids Algorithm:** This algorithm simulates flocking behavior, modeling the interactions between birds such as **separation, alignment, and cohesion**.

- **RAG (Retrieval-Augmented Generation):** RAG is integrated to allow the flock to dynamically retrieve and apply past movement strategies stored in a vector database, enhancing the flock’s ability to evade predators.

## Key Features

- **Adaptive Flocking Behavior:** The birds in the simulation are capable of learning and adapting their movement based on previous experiences, rather than following hardcoded rules.

- **Predator Interaction:** The simulation includes predator agents that threaten the flock, requiring them to adjust their behavior to avoid capture.

- **Boids Simulation:** Classic boid behaviors like flock cohesion, alignment, and separation are implemented, ensuring natural group dynamics.

- **RAG-Based Strategy Retrieval:** When threatened by predators, the birds search for and apply past strategies retrieved from a vector database, improving their chances of evasion.

## Synthetic Data Generation for Predator Interaction

The data generation component of this project is designed to simulate realistic flock and predator interactions. Using **GPT-4o**, synthetic_data_generation.py creates a dataset of bird movement strategies in response to predators. This data is stored in a vector database (ChromaDB) for future retrieval. The data generation process involves creating realistic movement strategies that are converted into vector embeddings using **text-embedding-3-small**.

The synthetic data includes: 

- **Bird Position and Velocity:** Each bird’s relative position and velocity are tracked, with values generated to reflect realistic movement patterns in the flock.

- **Force Components:** External forces acting on the birds, such as the repulsion force from predators, are modeled to create realistic responses.

- **Group Behavior:** The data ensures that the birds maintain some cohesion while adapting to the predator, with nearby birds exhibiting similar velocities and behaviors.

## Adaptive Flock using RAG

In this simulation, **RAG (Retrieval-Augmented Generation)** is used to enhance the boids' ability to adapt their behavior in response to the predator's movements. The core idea behind RAG is to combine retrieval from a **pre-existing knowledge base** with **real-time generation** to make informed decisions.

Here’s how RAG is incorporated in the simulation:

**1. Boid-Predator Interaction:**

A random predator is loaded to interact with the boid simulation. As each boid in the flock moves, its position and velocity are continuously updated. To avoid the predator, each boid needs to adapt its movement strategy based on how close it is to the predator and how its velocity aligns with that of the predator.

**2. Calculating Relative Positions and Velocities:**

Instead of looking at the absolute positions and velocities, the relative positions and velocities of each boid to the predator are calculated. This helps understand how far each boid is from the predator and how aligned (or opposed) its velocity is relative to the predator’s velocity.

**3. Embedding the Data:**

Every 50th frame, the simulation aggregates these relative positions and velocities for all the boids into a query string (e.g., a comma-separated list of position and velocity values). This query string is then transformed into an embedding using OpenAI Embeddings.
The embedding captures the semantic meaning of the relative positions and velocities, allowing the simulation to query a vector database (Chroma) for similar past scenarios.

**4. Querying the Vector Database:**

The simulation queries the vector database (flock_data) using this embedding. The database stores pre-calculated movement strategies (force vectors) that were generated in previous runs under similar conditions.
The query retrieves the most relevant movement strategies based on the current boid and predator situation. These retrieved strategies are typically force vectors, representing how each boid should adjust its movement to avoid the predator.

**5. Adapting Behavior:**

Once the database returns the most relevant movement strategies, the force vectors are extracted from the database results. These vectors represent the adjustments the boids should make to avoid the predator based on their relative positions and velocities.
These force vectors are then applied to update the boids' movement for the next frame.



## Setup

1. Clone the repository:
```bash
git clone https://github.com/Yenchia-Feng/RAG_Adaptive_Flock.git
cd RAG_Adaptive_Flock
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

**Note:** in synthetic_data_generation.py, the default number of data generation is 3. You can change this by setting `num_generations` to a different number.

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

- `adaptation.py`: Main simulation script
- `simulation.py`: Boid and flock behavior implementation
- `synthetic_data_generation.py`: Data generation for RAG
- `baseline_flock_animation.py`: Animates baseline flock behavior with predator
- `flock_data/`: Directory containing training data
- `vector_stores/`: Directory containing ChromaDB collections
