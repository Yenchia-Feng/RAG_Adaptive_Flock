import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import uniform, random
import csv

class Boid:
    # Configuration
    num_boids = 30 
    min_speed = 6
    max_speed = 11
    max_force = 3
    max_turn = 1
    perception = 100
    crowding = 35
    boid_radius = 20    # Physical size of each boid
    edge_distance_pct = 10
    separation_weight = 2
    alignment_weight = 1
    cohesion_weight = 1 
    
    def __init__(self, width=800, height=600):
        # Store window dimensions
        self.width = width
        self.height = height
        
        # Initialize position and velocity
        self.position = np.array([
            uniform(0, width),
            uniform(0, height)
        ], dtype=np.float64)
        
        self.velocity = np.array([
            uniform(-1, 1) * self.max_speed,
            uniform(-1, 1) * self.max_speed
        ], dtype=np.float64)
    
    def set_max_force(self, force):
        """Limit the magnitude of a force vector"""
        magnitude = np.linalg.norm(force)
        if magnitude > self.max_force:
            return (force / magnitude) * self.max_force
        return force
    
    def separation(self, neighbors):
        """Prevent overlapping and maintain comfortable spacing between boids"""
        steering = np.zeros(2, dtype=np.float64)
        for boid in neighbors:
            diff = self.position - boid.position
            dist = np.linalg.norm(diff)
            # Prevent division by zero
            if dist < 0.1:
                dist = 0.1
            # Physical collision prevention (hard minimum)
            physical_min = self.boid_radius * 2
            # Comfortable spacing distance (soft minimum)
            comfort_min = self.boid_radius * 4
            if dist < comfort_min:
                # Calculate repulsion force based on how close the boids are
                if dist < physical_min:
                    # Very strong repulsion when physically too close
                    repulsion = (physical_min - dist) / dist * 8.0  # Stronger multiplier for physical separation
                else:
                    # Moderate repulsion for comfort zone
                    repulsion = (comfort_min - dist) / dist * 3.0  # Moderate multiplier for comfort
                steering += (diff / dist) * repulsion
        return self.set_max_force(steering)
    
    def alignment(self, boids):
        steering = np.zeros(2, dtype=np.float64)
        for boid in boids:
            steering += boid.velocity
        steering = steering / len(boids)
        steering -= self.velocity
        steering = self.set_max_force(steering)
        return steering 
    
    def cohesion(self, boids):
        steering = np.zeros(2, dtype=np.float64)
        for boid in boids:
            steering += boid.position
        steering = steering / len(boids)
        steering -= self.position
        steering = self.set_max_force(steering)
        return steering 
    
    def avoid_boundaries(self, boids):
        """avoid boundaries"""
        steering = np.zeros(2, dtype=np.float64)
        margin = 50  # Distance from edge to start avoiding
        # Horizontal boundaries
        if self.position[0] < margin:
            steering[0] = self.max_force
        elif self.position[0] > self.width - margin:
            steering[0] = -self.max_force
            
        # Vertical boundaries
        if self.position[1] < margin:
            steering[1] = self.max_force
        elif self.position[1] > self.height - margin:
            steering[1] = -self.max_force      
        return self.set_max_force(steering)
    
    def get_neighbors(self, boids):
        return [boid for boid in boids if boid != self and 
                np.linalg.norm(self.position - boid.position) < self.perception]
    
    def update(self, dt, boids, external_force=np.zeros(2)):
        # Calculate steering force
        steering = np.zeros(2, dtype=np.float64)
        
        # Add strong boundary avoidance
        steering += self.avoid_boundaries(boids) * 1.0  # Pass boids to avoid_boundaries
        
        # Get neighbors and apply flocking rules
        neighbors = self.get_neighbors(boids)
        if neighbors:
            separation = self.separation(neighbors) * Boid.separation_weight
            alignment = self.alignment(neighbors) * Boid.alignment_weight
            cohesion = self.cohesion(neighbors) * Boid.cohesion_weight
            steering += (separation + alignment + cohesion) / (Boid.separation_weight + Boid.alignment_weight + Boid.cohesion_weight)
        
        # Update velocity and position with some random variation
        random_variation = np.random.normal(0, 0.1, 2)  # introduce a random force
        self.velocity += (steering * dt) + random_variation + external_force * dt
        
        
        # Limit speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed
        elif speed < self.min_speed:
            self.velocity = (self.velocity / speed) * self.min_speed
            
        # Update position
        new_position = self.position + self.velocity * dt
        
        # Constrain to boundaries
        new_position[0] = np.clip(new_position[0], 0, self.width)
        new_position[1] = np.clip(new_position[1], 0, self.height)
        
        # If we hit a boundary, reverse the velocity component
        if new_position[0] <= 0 or new_position[0] >= self.width:
            self.velocity[0] *= -1
        if new_position[1] <= 0 or new_position[1] >= self.height:
            self.velocity[1] *= -1
        
        self.position = new_position

class Flock:
    def __init__(self, n_boids, width=800, height=600):
        # Choose a random starting area for the flock
        center_x = uniform(width * 0.2, width * 0.8)
        center_y = uniform(height * 0.2, height * 0.8)
        
        # Choose a random initial direction for the flock
        initial_angle = uniform(0, 2 * np.pi)
        initial_direction = np.array([
            np.cos(initial_angle),
            np.sin(initial_angle)
        ])
        
        # Choose a base speed for the flock
        base_speed = uniform(Boid.min_speed, Boid.max_speed)
        
        self.boids = []
        for _ in range(n_boids):
            # Random position near the center
            position = np.array([
                center_x + uniform(-200, 200),  # Spread within 100x100 area
                center_y + uniform(-200, 200)
            ])
            
            # Random velocity similar to the base direction
            angle_variation = uniform(-1.5, 1.5)  # ±0.5 radians variation
            speed_variation = uniform(-3, 3)      # ±1 speed variation
            
            velocity = initial_direction * (base_speed + speed_variation)
            velocity = np.array([
                velocity[0] * np.cos(angle_variation) - velocity[1] * np.sin(angle_variation),
                velocity[0] * np.sin(angle_variation) + velocity[1] * np.cos(angle_variation)
            ])
            
            # Create boid with specific initial conditions
            boid = Boid(width, height)
            boid.position = position
            boid.velocity = velocity
            self.boids.append(boid)
    
    def update(self, dt, external_force=np.zeros((30, 2))):
        """
        Update the flock's boids positions and velocities
        Args:
            dt: time step
            external_force: numpy array of shape (30, 2) containing force vectors for each boid
        """
        for i, boid in enumerate(self.boids):
            boid.update(dt, self.boids, external_force[i])

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
    
    def animate(frame):
        ax.clear()
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        
        # Update flock
        flock.update(0.1)
        
        # Get positions and velocities
        positions = np.array([b.position for b in flock.boids])
        velocities = np.array([b.velocity for b in flock.boids])
        
        # Plot boids
        ax.scatter(positions[:, 0], positions[:, 1], c='white', s=15)
        ax.quiver(positions[:, 0], positions[:, 1],
                 velocities[:, 0], velocities[:, 1],
                 color='blue', scale=200, width=0.003)
        
        return ax.artists + ax.collections
    
    # Fix the warning by specifying save_count and cache_frame_data
    anim = FuncAnimation(fig, animate,
                        frames=None,
                        interval=1,
                        blit=True,
                        cache_frame_data=False)  # Disable frame caching
    
    plt.show()

if __name__ == "__main__":
    main() 