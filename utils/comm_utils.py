def calculate_comm_time(data_size_bytes, bandwidth_MBps):
    return data_size_bytes / (bandwidth_MBps * 1e6)

def calculate_comm_energy(data_size_bytes, energy_cost_per_byte):
    return data_size_bytes * energy_cost_per_byte
