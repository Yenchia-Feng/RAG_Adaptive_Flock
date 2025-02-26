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


def main():
    # Setup
    width, height = 800, 600
    flock = Flock(Boid.num_boids, width, height)
    
    # Create CSV file for storing boid data
    csv_filename = 'boid_data.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['boid_id', 'timestep', 'position_x', 'position_y', 'velocity_x', 'velocity_y'])
        
        # Save initial positions (timestep 0)
        for i, boid in enumerate(flock.boids):
            writer.writerow([
                i,  # boid_id
                0,  # timestep
                boid.position[0],  # position_x
                boid.position[1],  # position_y
                boid.velocity[0],  # velocity_x
                boid.velocity[1]   # velocity_y
            ])
        
        # Simulate and save 500 timesteps
        for timestep in range(1, 500):
            flock.update(0.1)  # No need to pass external_force here anymore
            for i, boid in enumerate(flock.boids):
                writer.writerow([
                    i,  # boid_id
                    timestep,  # timestep
                    boid.position[0],  # position_x
                    boid.position[1],  # position_y
                    boid.velocity[0],  # velocity_x
                    boid.velocity[1]   # velocity_y
                ])
    
    print(f"Saved {500} timesteps of boid data to {csv_filename}")
    
    # Create figure and axis for visualization
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    # Load predator data
    predator_lines, metadata = load_random_predator()
 
    def animate(frame):

        print(['frame', frame])

        ax.clear()
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        
        # Update flock
        flock.update(0.1)
        
        # Get positions and velocities of boids
        positions = np.array([b.position for b in flock.boids])
        velocities = np.array([b.velocity for b in flock.boids])

        #print([frame, positions])

        # Get predator data for position and velocity
        predator_pos = np.array([predator_lines[frame].split(',')[1], predator_lines[frame].split(',')[2]])
        predator_vel = np.array([predator_lines[frame].split(',')[3], predator_lines[frame].split(',')[4]])
        
        # Plot boids
        ax.scatter(positions[:, 0], positions[:, 1], c='white', s=15, label='Boids')
        ax.quiver(positions[:, 0], positions[:, 1],
                 velocities[:, 0], velocities[:, 1],
                 color='blue', scale=200, width=0.003, label='Boid Velocities')

        # Plot predator
        ax.scatter(float(predator_pos[0]), float(predator_pos[1]), c='red', s=100, marker='*', label='Predator')
        ax.quiver(float(predator_pos[0]), float(predator_pos[1]),
                 float(predator_vel[0]), float(predator_vel[1]), 
                 color='red', scale=200, width=0.005, label='Predator Velocity')
        ax.legend(loc='upper right')
        
        return ax.artists + ax.collections
    
     # Create animation with explicit frame range
    num_frames = len(predator_lines)
    print(f"Creating animation with {num_frames} frames")
    
    anim = FuncAnimation(
        fig, animate,
        frames=range(num_frames),  # Explicitly use range
        interval=1,
        blit=True,
        repeat=False
    )
    # Fix the warning by specifying save_count and cache_frame_data
    #anim = FuncAnimation(fig, animate,
    #                    frames=None,
    #                    interval=1,
    #                    blit=True,
    #                    cache_frame_data=False)  # Disable frame caching
    
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
