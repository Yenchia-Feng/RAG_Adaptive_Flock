from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import datetime
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import numpy as np
import random
import chromadb

# Load environment variables
load_dotenv()

# Initialize the client with explicit API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Example context documents for RAG
INI_CONTEXTS = [
    """
    Bird Flocking Initial Conditions:
    - Flock typically starts in loose cluster formation
    - Birds maintain minimum separation distance (20-35 units)
    - Initial velocities aligned within ±30 degrees
    - Speed range: 6-11 units per timestep
    - Position spread: 200-400 units from center of the flock
    """,
    """
    Predator-Flock Initial Setup:
    - Predator typically approaches from edge of visible area
    - Flock center offset 300-500 units from predator
    """
]

STRATEGY_CONTEXTS = [
    """
    Predator Avoidance Forces:
    - Repulsion force inversely proportional to distance
    - Maximum force magnitude: 6.0 units
    - Force direction points away from predator
    - Nearby birds experience similar force magnitudes
    - Force decreases beyond 150-200 unit radius
    """,
    """
    Group Cohesion During Evasion:
    - Birds maintain partial group structure
    - Stronger forces for birds closer to predator
    - Coordinated escape trajectories
    - Balance between escape and group cohesion
    - Force vectors consider neighbor positions
    """
]

def generate_flock_predator_data() -> str:
    """Generate CSV data for flock and predator interaction using GPT-4o"""
    
    prompt = f"""
    Based on these contexts about bird flocking and predator avoidance:
    {INI_CONTEXTS[0]}
    {INI_CONTEXTS[1]}
    {STRATEGY_CONTEXTS[0]}
    {STRATEGY_CONTEXTS[1]}
    
    Generate exactly 30 rows of CSV data for birds responding to a predator.
    Each row MUST follow this exact format with no extra spaces or lines:
    bird_id,dx,dy,dvx,dvy,fx,fy
    
    Requirements:
    - Exactly 30 rows (no empty lines)
    - bird_id must be sequential from 0 to 29
    - dx,dy: position offset from predator (-800 to 800, -600 to 600)
    - dvx,dvy: relative velocity (-11 to 11)
    - fx,fy: external force components (each component between -6.0 and 6.0)
    
    The flock should exhibit realistic group behavior:
    1. Birds closer to predator should have stronger avoidance forces
    2. Birds should maintain some cohesion (not scatter completely)
    3. Nearby birds should have similar velocities
    4. Forces should point away from predator position
    
    Return ONLY a CSV string with 30 rows (one per bird) and no header.
    Each row format: bird_id,dx,dy,dvx,dvy,fx,fy
    Example format:
    0,-200,150,5,-3,2.1,-1.5
    1,-180,140,4,-2,1.8,-1.2
    ...and so on for exactly 30 rows
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a movement strategy expert. Generate ONLY CSV data with exactly 30 rows and no additional text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        
        # Clean the response: remove empty lines and extra whitespace
        csv_data = '\n'.join(
            line.strip() for line in response.choices[0].message.content.strip().split('\n')
            if line.strip()
        )
        
        # Validate CSV format
        try:
            # Parse CSV string to verify format
            rows = [row.split(',') for row in csv_data.split('\n')]
            
            # Filter out any non-data rows (e.g., headers or empty lines)
            rows = [row for row in rows if len(row) == 7 and row[0].strip().isdigit()]
            
            if len(rows) != 30:
                raise ValueError(f"Expected 30 rows, got {len(rows)}")
            
            # Sort rows by bird_id to ensure correct order
            rows.sort(key=lambda x: int(x[0]))
            
            # Validate each row
            for i, row in enumerate(rows):
                # Verify bird_id is sequential
                if int(row[0]) != i:
                    raise ValueError(f"Invalid bird_id sequence: expected {i}, got {row[0]}")
                
                # Verify all values are numeric and within bounds
                dx, dy = float(row[1]), float(row[2])
                dvx, dvy = float(row[3]), float(row[4])
                fx, fy = float(row[5]), float(row[6])
                
                if not (-800 <= dx <= 800 and -600 <= dy <= 600):
                    raise ValueError(f"Position out of bounds: {dx},{dy}")
                if not (-11 <= dvx <= 11 and -11 <= dvy <= 11):
                    raise ValueError(f"Velocity out of bounds: {dvx},{dvy}")
                # Check each force component separately instead of magnitude
                if not (-6.0 <= fx <= 6.0 and -6.0 <= fy <= 6.0):
                    raise ValueError(f"Force component out of bounds: {fx},{fy}")
            
            # Reconstruct validated CSV string
            return '\n'.join(','.join(row) for row in rows)
            
        except (ValueError, IndexError) as e:
            print(f"CSV validation error: {e}")
        return None
            
    except Exception as e:
        print(f"Error generating flock data: {e}")
        return None

def save_flock_data(filename: str = 'flock_predator_data.csv'):
    """Generate and save flock-predator interaction data to CSV file"""
    csv_data = generate_flock_predator_data()
    if csv_data:
        with open(filename, 'w') as f:
            # Add header
            f.write("bird_id,dx,dy,dvx,dvy,fx,fy\n")
            # Add data
            f.write(csv_data)
        print(f"Successfully saved flock data to {filename}")
    else:
        print("Failed to generate flock data")

def generate_predator_path(num_steps: int = 500, dt: float = 0.1) -> str:
    """Generate smooth random walk path for predator movement
    
    Args:
        num_steps: Number of timesteps to simulate
        dt: Time step size
        
    Returns:
        CSV string with format: timestep,dx,dy,dvx,dvy
    """
    # Initialize arrays for position and velocity
    dx = np.zeros(num_steps)
    dy = np.zeros(num_steps)
    dvx = np.zeros(num_steps) 
    dvy = np.zeros(num_steps)
    
    # Initial conditions - start at random position with random velocity
    dx[0] = random.uniform(0, 800)  # Screen width
    dy[0] = random.uniform(0, 600)  # Screen height
    dvx[0] = random.uniform(-10, 10)
    dvy[0] = random.uniform(-10, 10)
    
    # Generate smooth random walk by updating velocity with random accelerations
    for t in range(1, num_steps):
        # Random acceleration with smoothing
        ax = random.uniform(-5, 5)
        ay = random.uniform(-5, 5)
        
        # Update velocity
        dvx[t] = dvx[t-1] + ax * dt
        dvy[t] = dvy[t-1] + ay * dt
        
        # Limit maximum velocity
        speed = np.sqrt(dvx[t]**2 + dvy[t]**2)
        if speed > 20:
            dvx[t] *= 20/speed
            dvy[t] *= 20/speed
            
        # Update position
        dx[t] = dx[t-1] + dvx[t] * dt
        dy[t] = dy[t-1] + dvy[t] * dt
        
        # Keep predator within screen bounds
        if dx[t] < 0:
            dx[t] = 0
            dvx[t] *= -0.5  # Bounce off boundary
        elif dx[t] > 800:
            dx[t] = 800
            dvx[t] *= -0.5
            
        if dy[t] < 0:
            dy[t] = 0
            dvy[t] *= -0.5
        elif dy[t] > 600:
            dy[t] = 600
            dvy[t] *= -0.5
    
    # Format as CSV string
    csv_rows = []
    for t in range(num_steps):
        row = f"{t},{dx[t]:.1f},{dy[t]:.1f},{dvx[t]:.1f},{dvy[t]:.1f}"
        csv_rows.append(row)
    
    return "\n".join(csv_rows)

def save_predator_path(filename: str = 'predator_path.csv'):
    """Generate and save predator path data to CSV file"""
    csv_data = generate_predator_path()
    if csv_data:  
        with open(filename, 'w') as f:
            # Add header
            f.write("timestep,dx,dy,dvx,dvy\n")
            # Add data
            f.write(csv_data)
        print(f"Successfully saved predator path to {filename}")
    else:
        print("Failed to generate predator path")


def main(num_generations: int):
    """Generate multiple flock-predator interaction datasets and store their embeddings"""
    print(f"Generating {num_generations} flock-predator interaction datasets...")
    
    # Create specific directories for different types of data
    base_path = os.path.join("vector_stores")
    flock_store_path = os.path.join(base_path, "flock_store")
    predator_store_path = os.path.join(base_path, "predator_store")
    data_path = "flock_data"
    predator_path = "predator_data"
    
    # Create all necessary directories
    for path in [flock_store_path, predator_store_path, data_path, predator_path]:
        os.makedirs(path, exist_ok=True)
    
    # Initialize the OpenAI embedding model
    embedding_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"), model='text-embedding-3-small')
    
    for i in range(num_generations):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\nGenerating dataset {i+1}/{num_generations}")
        
        # Create Chroma clients with unique collection names for this generation
        flock_client = chromadb.PersistentClient(path=flock_store_path)
        predator_client = chromadb.PersistentClient(path=predator_store_path)
        
        flock_collection_name = f"flock_collection_{timestamp}"
        predator_collection_name = f"predator_collection_{timestamp}"
        
        print(f"Creating collections:")
        print(f"- Flock: {flock_collection_name}")
        print(f"- Predator: {predator_collection_name}")
        
        # Create new collections for this generation
        # flock_collection = flock_client.create_collection(flock_collection_name)
        try:
            flock_collection = flock_client.create_collection(flock_collection_name)
        except chromadb.errors.UniqueConstraintError:
            print(f"Collection {flock_collection_name} already exists. Skipping creation.")

        predator_collection = predator_client.create_collection(predator_collection_name)
        
        # Create Chroma instances
        flock_store = Chroma(
            client=flock_client,
            collection_name=flock_collection_name,
            embedding_function=embedding_model
        )
        
        predator_store = Chroma(
            client=predator_client,
            collection_name=predator_collection_name,
            embedding_function=embedding_model
        )
        
        # Generate data
        try:
            # Generate flock data
            flock_csv_data = generate_flock_predator_data()
            if flock_csv_data is None:
                print(f"Skipping dataset {i+1} due to flock generation error")
                continue
                
            # Save flock data to file
            flock_filename = os.path.join(data_path, f"flock_predator_data_{timestamp}.csv")
            with open(flock_filename, 'w') as f:
                f.write("bird_id,dx,dy,dvx,dvy,fx,fy\n")
                f.write(flock_csv_data)
            print(f"Saved flock data to {flock_filename}")
            
            # Generate predator data
            predator_csv_data = generate_predator_path()
            predator_filename = os.path.join(predator_path, f"predator_path_{timestamp}.csv")
            with open(predator_filename, 'w') as f:
                f.write("timestep,dx,dy,dvx,dvy\n")
                f.write(predator_csv_data)
            print(f"Saved predator data to {predator_filename}")
            
            # Create Document objects
            flock_document = Document(
                page_content=flock_csv_data,
                metadata={
                    "filename": flock_filename,
                    "timestamp": timestamp,
                    "generation": i,
                    "type": "flock"
                }
            )
            
            predator_document = Document(
                page_content=predator_csv_data,
                metadata={
                    "filename": predator_filename,
                    "timestamp": timestamp,
                    "generation": i,
                    "type": "predator"
                }
            )
            
            # Add documents to stores
            flock_store.add_documents(
                documents=[flock_document],
                ids=[f"flock_{timestamp}"]
            )
            
            predator_store.add_documents(
                documents=[predator_document],
                ids=[f"predator_{timestamp}"]
            )
            
            print(f"Successfully stored dataset {i+1} in vector stores")
            
            # Verify storage
            print(f"Verifying storage:")
            print(f"- Flock collection count: {flock_collection.count()}")
            print(f"- Predator collection count: {predator_collection.count()}")
            
        except Exception as e:
            print(f"Error generating dataset {i+1}: {e}")
            continue
    
    # Final verification
    print("\nFinal verification:")
    print("Listing all collections:")
    try:
        flock_client = chromadb.PersistentClient(path=flock_store_path)
        predator_client = chromadb.PersistentClient(path=predator_store_path)
        
        # Get collection names
        flock_collection_names = flock_client.list_collections()
        predator_collection_names = predator_client.list_collections()
        
        print(f"Found {len(flock_collection_names)} flock collections:")
        for collection_name in flock_collection_names:
            # Get collection to access its properties
            collection = flock_client.get_collection(name=collection_name)
            print(f"- {collection_name} (count: {collection.count()})")
        
        print(f"\nFound {len(predator_collection_names)} predator collections:")
        for collection_name in predator_collection_names:
            # Get collection to access its properties
            collection = predator_client.get_collection(name=collection_name)
            print(f"- {collection_name} (count: {collection.count()})")
            
    except Exception as e:
        print(f"Error during final verification: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main(num_generations=50)
    print("\nData generation complete")
