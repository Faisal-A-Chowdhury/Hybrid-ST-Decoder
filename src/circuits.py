import torch
import stim

def get_data_surface(d: int = 5, r: int = 5, p: float = 0.005, train_size: int = 50000):
    """Generate Surface Code data using Stim simulator (as in the paper)"""
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=r,
        distance=d,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )
    
    sampler = circuit.compile_detector_sampler()
    detections, flips = sampler.sample(train_size, separate_observables=True)
    
    # Convert to -1/+1 format (standard in the paper)
    detections = (detections.astype(int) * 2 - 1)
    detections = torch.tensor(detections, dtype=torch.float32)
    flips = torch.tensor(flips.astype(int).flatten(), dtype=torch.float32)
    
    num_input = detections.shape[1]
    
    return detections, flips, num_input, circuit


def transform_to_3d(detections: torch.Tensor, d: int, r: int):
    """Convert 2D detections to 3D spatio-temporal tensor [B, 1, X, Y, T]"""
    B = detections.shape[0]
    # Approximate grid size for rotated surface code
    X = Y = 2 * d
    T = r + 1
    
    tensor = torch.zeros((B, 1, X, Y, T), dtype=torch.float32, device=detections.device)
    
    # Fill available detectors (simplified but functional mapping)
    seq_len = min(detections.shape[1], X * Y * T)
    flat_tensor = tensor.view(B, -1)
    flat_tensor[:, :seq_len] = detections[:, :seq_len]
    
    return tensor


def get_circuit_surface(d: int = 5, r: int = 5, p: float = 0.005):
    """Return raw Stim circuit"""
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=r,
        distance=d,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )


# Quick Test
if __name__ == "__main__":
    print("Testing Surface Code Data Generation...")
    detections, flips, num_input, circuit = get_data_surface(d=5, r=5, p=0.005, train_size=1000)
    
    print(f"Data Generated Successfully!")
    print(f"Detections Shape : {detections.shape}")
    print(f"Flips Shape      : {flips.shape}")
    print(f"Number of Detectors: {num_input}")
    
    X_3d = transform_to_3d(detections[:8], d=5, r=5)
    print(f"3D Tensor Shape  : {X_3d.shape}")
