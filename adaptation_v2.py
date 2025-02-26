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
        
        # Simulate and save 1000 timesteps
        for timestep in range(1, 1001):
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
    
    print(f"Saved {1000} timesteps of boid data to {csv_filename}")
    
    # Create figure and axis for visualization
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    # Load predator data
    predator_lines, metadata = load_random_predator()
    
    def animate(frame):
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
        ax.scatter(positions[:, 0], positions[:, 1], c='white', s=15)
        ax.quiver(positions[:, 0], positions[:, 1],
                 velocities[:, 0], velocities[:, 1],
                 color='blue', scale=200, width=0.003)

        # Plot predator
        ax.scatter(float(predator_pos[0]), float(predator_pos[1]), c='red', s=100, marker='*', label='Predator')
        ax.quiver(float(predator_pos[0]), float(predator_pos[1]),
                 float(predator_vel[0]), float(predator_vel[1]), 
                 color='red', scale=200, width=0.005)
        ax.legend()
        

        
        return ax.artists + ax.collections
    
    # Fix the warning by specifying save_count and cache_frame_data
    anim = FuncAnimation(fig, animate,
                        frames=None,
                        interval=1,
                        blit=True,
                        cache_frame_data=False)  # Disable frame caching
    
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

def visualize_simulation(flock_store, predator_store):
    """Visualize the simulation using simulation.py with retrieved force data"""
    # Setup
    width, height = 800, 600
    num_boids = 30
    
    # Create figure with two subplots
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    plt.subplots_adjust(wspace=0.3)
    
    # Initialize two separate flocks
    flock1 = Flock(num_boids, width, height)
    flock2 = Flock(num_boids, width, height)
    
    # Get data from stores
    flock_results = flock_store.get()
    predator_results = predator_store.get()
    predator_lines = [line for line in predator_results['documents'][0].split('\n') 
                     if line.strip() and not line.startswith('timestep')]
    
    print(f"Number of predator positions: {len(predator_lines)}")
    
    # Create plots for both animations
    scatter1 = ax1.scatter([], [], c='white', s=30, label='Boids 1')
    scatter2 = ax2.scatter([], [], c='white', s=30, label='Boids 2')
    
    pred_scatter1 = ax1.scatter([], [], c='red', s=100, marker='*', label='Predator')
    pred_scatter2 = ax2.scatter([], [], c='red', s=100, marker='*', label='Predator')
    
    quiver1 = ax1.quiver([], [], [], [], color='blue', scale=50, width=0.003)
    quiver2 = ax2.quiver([], [], [], [], color='blue', scale=50, width=0.003)
    
    pred_quiver1 = ax1.quiver([], [], [], [], color='red', scale=50, width=0.005)
    pred_quiver2 = ax2.quiver([], [], [], [], color='red', scale=50, width=0.005)
    
    text1 = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, color='white')
    text2 = ax2.text(0.02, 0.98, '', transform=ax2.transAxes, color='white')
    
    # Initialize both plots
    for ax in [ax1, ax2]:
        ax.set_xlim(-width/2, width/2)
        ax.set_ylim(-height/2, height/2)
        ax.grid(True, alpha=0.2)
        ax.set_xticks(np.arange(-width/2, width/2 + 1, 100))
        ax.set_yticks(np.arange(-height/2, height/2 + 1, 100))
        ax.tick_params(colors='white')
        ax.set_facecolor('black')
        ax.legend(loc='upper right')
    
    ax1.set_title('Flock Instance 1', color='white', pad=10)
    ax2.set_title('Flock Instance 2', color='white', pad=10)
    
    def animate(frame):
        # Update frame counters
        text1.set_text(f'Frame: {frame}')
        text2.set_text(f'Frame: {frame}')
        
        # Get predator data
        pred_data = predator_lines[frame].split(',')
        pred_x, pred_y = float(pred_data[1]), float(pred_data[2])
        pred_vx, pred_vy = float(pred_data[3]), float(pred_data[4])
        predator_pos = np.array([pred_x, pred_y])
        
        # Update predators in both plots
        for scatter, quiver in [(pred_scatter1, pred_quiver1), (pred_scatter2, pred_quiver2)]:
            scatter.set_offsets([[pred_x, pred_y]])
            quiver.set_offsets([[pred_x, pred_y]])
            quiver.set_UVC([pred_vx], [pred_vy])
        
        # Update flock 1
        for boid in flock1.boids:
            boid.update(flock1.boids, predator_pos, np.zeros(2))
        positions1 = np.array([b.position for b in flock1.boids])
        velocities1 = np.array([b.velocity for b in flock1.boids])
        
        # Update flock 2
        for boid in flock2.boids:
            boid.update(flock2.boids, predator_pos, np.zeros(2))
        positions2 = np.array([b.position for b in flock2.boids])
        velocities2 = np.array([b.velocity for b in flock2.boids])
        
        # Update visualization for both flocks
        scatter1.set_offsets(positions1)
        quiver1.set_offsets(positions1)
        quiver1.set_UVC(velocities1[:, 0], velocities1[:, 1])
        
        scatter2.set_offsets(positions2)
        quiver2.set_offsets(positions2)
        quiver2.set_UVC(velocities2[:, 0], velocities2[:, 1])
        
        return (scatter1, scatter2, pred_scatter1, pred_scatter2,
                quiver1, quiver2, pred_quiver1, pred_quiver2,
                text1, text2)
    
    # Create single animation that updates both plots
    anim = FuncAnimation(
        fig, animate,
        frames=range(len(predator_lines)),  # Use exact number of frames
        interval=50,
        blit=True,
        repeat=False  # Don't loop the animation
    )
    
    plt.show()

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
