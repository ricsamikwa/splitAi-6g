import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
fc = 3.5e9  # Carrier frequency (Hz)
c = 3e8     # Speed of light (m/s)
lambda_c = c / fc
bandwidth = 20e6  # 20 MHz
noise_figure = 9  # dB
thermal_noise = -174 + 10 * np.log10(bandwidth)  # dBm
total_noise = thermal_noise + noise_figure  # dBm

# UE speeds in m/s
speeds = {
    'Fixed': 0,
    'Pedestrian': 3,
    'Vehicular': 8.33  # 30 km/h
}

# Simulation duration and sampling
duration = 10  # seconds
sample_rate = 100  # Hz
num_samples = duration * sample_rate
t = np.linspace(0, duration, num_samples)

# gNB position
gnb_pos = np.array([0, 0, 10])  # 10 meters height

# Function to compute distance and path loss
def compute_distance(speed, t):
    ue_pos = np.array([t * speed, np.zeros_like(t), np.zeros_like(t)])  # UE moves along x-axis
    distances = np.linalg.norm(ue_pos - gnb_pos[:, None], axis=0)
    return distances

def compute_path_loss(distances):
    # Free-space path loss with Rayleigh fading (TDL approximation)
    path_loss_db = 20 * np.log10(distances) + 20 * np.log10(fc) - 147.55
    fading_db = np.random.normal(0, 3, size=distances.shape)  # Rayleigh fading approximation
    total_loss_db = path_loss_db + fading_db
    return total_loss_db

def compute_throughput(loss_db):
    # Received power in dBm (assuming transmit power = 23 dBm)
    rx_power_dbm = 23 - loss_db
    sinr_db = rx_power_dbm - total_noise
    sinr_linear = 10 ** (sinr_db / 10)
    capacity = bandwidth * np.log2(1 + sinr_linear)  # Shannon capacity in bps
    return capacity / 1e6  # Convert to Mbps

# Compute throughput for each scenario
plt.figure(figsize=(10, 6))
for label, speed in speeds.items():
    distances = compute_distance(speed, t)
    loss_db = compute_path_loss(distances)
    throughput = compute_throughput(loss_db)
    plt.plot(t, throughput, label=f'{label} UE ({speed:.2f} m/s)')

# Plot settings
plt.xlabel('Time (s)')
plt.ylabel('Throughput (Mbps)')
plt.title('Wireless Link Throughput vs. Time for Different UE Speeds (TDL Channel)')
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.savefig("throughput_comparison_tdl.png")
plt.show()