import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from simulation import Boid, Flock
import os
import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import random
import csv
from dotenv import load_dotenv

load_dotenv()

def main():
    # Setup
    width, height = 800, 600
    flock = Flock(Boid.num_boids, width, height)
    
    # Create figure and axis
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 8))  # Slightly wider to accommodate legend
    ax = fig.add_subplot(111)

    # Load predator data
    predator_lines, metadata = load_random_predator()
    print(predator_lines[1])
 
    # Check flock_data folder and load files into collection if needed
    data_path = r"C:\Users\yench\Documents\Flock_RAG\flock_data"
    vector_store_path = r"C:\Users\yench\Documents\Flock_RAG\vector_stores"
    
    # Connect to collections
    try:
        # Connect to flock_store (vectors)
        store_client = chromadb.PersistentClient(path=vector_store_path)
        flock_store = store_client.get_collection("flock_store")
        
        # Connect to flock_data (documents)
        data_client = chromadb.PersistentClient(path=data_path)
        flock_data = data_client.get_collection("flock_data")
        
        print(f"Connected to collections. Found {flock_data.count()} documents")
        
    except Exception as e:
        print(f"Error accessing collections: {e}")
        print("Please run synthetic_data_generation.py first to create and populate the collections")
        return

    # Track which frames we've processed and store last force vectors
    processed_frames = set()
    last_force_vectors = np.zeros((30, 2))  # Initialize with zero forces
    
    def animate(frame):
        ax.clear()
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        
        nonlocal last_force_vectors  # Allow updating the last_force_vectors
        
        # Skip if we've already processed this frame
        if frame in processed_frames:
            return ax.artists + ax.collections
        processed_frames.add(frame)
        
        # Get positions and velocities of boids
        positions = np.array([b.position for b in flock.boids])
        velocities = np.array([b.velocity for b in flock.boids])

        # Get predator data for position and velocity
        predator_pos = np.array([float(predator_lines[frame].split(',')[1]), 
                                float(predator_lines[frame].split(',')[2])])
        predator_vel = np.array([float(predator_lines[frame].split(',')[3]), 
                                float(predator_lines[frame].split(',')[4])])

        # Calculate relative positions and velocities
        relative_positions = positions - predator_pos
        relative_velocities = velocities - predator_vel

        # Only calculate new forces every 50 frames
        if frame % 50 == 0:
            try:
                # Format query exactly like stored data
                flat_data = []
                for pos, vel in zip(relative_positions, relative_velocities):
                    flat_data.extend([pos[0], pos[1], vel[0], vel[1]])
                
                query_text = ",".join(f"{x:.6f}" for x in flat_data)
                print(f"\nQuerying collections (frame {frame})")
                
                embeddings = OpenAIEmbeddings()
                query_embedding = embeddings.embed_query(query_text)
                
                # Query using the embedding
                results = flock_data.query(
                    query_embeddings=[query_embedding],
                    n_results=5,
                    include=['documents', 'distances', 'metadatas']
                )
                
                print(f"Found {len(results['documents'][0])} matches")
                
                # Get the best matching document
                if results['documents'][0]:
                    best_match = results['documents'][0][0]
                    parts = best_match.split(',')
                    
                    # Extract force components from the document
                    if len(parts) >= 2:
                        base_fx, base_fy = float(parts[-2]), float(parts[-1])
                        print(f"Base forces from best match: fx={base_fx:.2f}, fy={base_fy:.2f}")
                        
                        # Generate 30 force vectors based on the base forces
                        new_forces = np.zeros((30, 2))  # Initialize array for all boids
                        for i in range(30):  # For each boid
                            # Calculate individual force based on relative position to predator
                            rel_pos = relative_positions[i]
                            rel_vel = relative_velocities[i]
                            
                            # Adjust base forces based on relative position and velocity
                            dist_to_predator = np.linalg.norm(rel_pos)
                            vel_alignment = np.dot(rel_vel, predator_vel) / (np.linalg.norm(rel_vel) * np.linalg.norm(predator_vel))
                            
                            # Scale forces based on distance and velocity alignment
                            fx = base_fx * (1 + 0.2 * np.random.randn()) * (1 / (1 + 0.1 * dist_to_predator))
                            fy = base_fy * (1 + 0.2 * np.random.randn()) * (1 / (1 + 0.1 * dist_to_predator))
                            
                            # Add slight variation for each boid
                            if vel_alignment > 0:
                                fx *= (1 - 0.5 * vel_alignment)
                                fy *= (1 - 0.5 * vel_alignment)
                            
                            new_forces[i] = [fx, fy]
                        
                        # Update last_force_vectors with the new forces
                        last_force_vectors = new_forces
                        
                        print("\nGenerated force vectors for all boids:")
                        for i in range(30):
                            print(f"Boid {i}: fx={last_force_vectors[i,0]:.2f}, fy={last_force_vectors[i,1]:.2f}")

            except Exception as e:
                print(f"Query error: {e}")
                import traceback
                print(traceback.format_exc())

        # Verify force vector shape before updating
        if last_force_vectors.shape != (30, 2):
            print(f"Warning: Force vectors shape {last_force_vectors.shape} incorrect, resetting to zeros")
            last_force_vectors = np.zeros((30, 2))

        # Update flock with current force vectors
        flock.update(0.1, last_force_vectors)

        # Plot boids and forces with labels
        boids = ax.scatter(positions[:, 0], positions[:, 1], 
                         c='white', s=15, label='Boids')
        velocity_arrows = ax.quiver(positions[:, 0], positions[:, 1],
                                  velocities[:, 0], velocities[:, 1],
                                  color='blue', scale=200, width=0.003,
                                  label='Boid Velocities')

        # Plot predator
        predator = ax.scatter(float(predator_pos[0]), float(predator_pos[1]), 
                            c='red', s=100, marker='*', label='Predator')
        predator_arrow = ax.quiver(float(predator_pos[0]), float(predator_pos[1]),
                                 float(predator_vel[0]), float(predator_vel[1]), 
                                 color='red', scale=200, width=0.005,
                                 label='Predator Velocity')

        # Plot force vectors
        force_arrows = ax.quiver(positions[:, 0], positions[:, 1],
                               last_force_vectors[:, 0], last_force_vectors[:, 1],
                               color='yellow', scale=10, width=0.002,
                               alpha=0.5, label='Adaptation Forces')

        # Add frame counter
        ax.text(10, height-20, f'Frame: {frame}', color='white')
        if frame % 100 == 0:
            ax.text(10, height-40, 'Updating forces', color='yellow')
        
        # Simple legend in top right corner
        legend = ax.legend(loc='upper right',
                          frameon=True,
                          facecolor='black',
                          edgecolor='white')
        
        return [boids, velocity_arrows, predator, predator_arrow, force_arrows, legend]
    
    # Create animation with explicit frame range
    num_frames = len(predator_lines)
    print(f"Creating animation with {num_frames} frames")
    
    anim = FuncAnimation(
        fig, animate,
        frames=range(num_frames),
        interval=1,
        blit=True,
        repeat=False
    )
    
    plt.show()

def load_random_predator():
    """Load a random predator trajectory from the predator_data directory"""
    try:
        # Get path to predator data
        predator_path = "predator_data"
        predator_files = [f for f in os.listdir(predator_path) if f.endswith('.csv')]
        selected_file = random.choice(predator_files)
        
        # Read the file
        file_path = os.path.join(predator_path, selected_file)
        with open(file_path, 'r') as f:
            predator_data = f.read()
        
        # Split into lines and remove header
        lines = [line for line in predator_data.split('\n') 
                if line.strip() and not line.startswith('timestep')]
        
        # Get first position
        first_pos = lines[0].split(',')
        print(f"First position: timestep={first_pos[0]}, x={first_pos[1]}, y={first_pos[2]}, vx={first_pos[3]}, vy={first_pos[4]}")
        
        return lines, {"filename": selected_file}
        
    except Exception as e:
        print(f"Error loading predator data: {e}")
        return None, None

if __name__ == "__main__":
    main()
    predator_lines, metadata = load_random_predator()
    
    if predator_lines:
        # Example: get position at index 0
        pos = predator_lines[0].split(',')
        print(f"\nInitial predator position:")
        print(f"x: {pos[1]}")
        print(f"y: {pos[2]}")
        print(f"velocity: ({pos[3]}, {pos[4]})")
