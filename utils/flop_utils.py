def calculate_inference_time(flops, cpu_freq, flops_per_cycle):
    return flops / (cpu_freq * 1e9 * flops_per_cycle)
