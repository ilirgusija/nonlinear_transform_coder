from model import Quantizer 
import torch
import numpy as np

def compare_quantizers(data, num_levels_list):
    for n_levels in num_levels_list:
        # Lloyd-Max quantization
        lm_levels, lm_boundaries = lloyd_max_quantize(data, n_levels)
        lm_quantized_data = quantize_data_with_lm(data, lm_levels, lm_boundaries)

        # Load the trained Quantizer model
        quantizer = Quantizer(data, n_levels)
        quantizer.load_state_dict(torch.load(f'quantizer_params_{n_levels}.pt'))

        # Quantize data using the trained model
        trained_quantized_data = quantizer(data).detach()

        # Compare results
        lm_mse = ((data - lm_quantized_data) ** 2).mean()
        trained_mse = ((data - trained_quantized_data) ** 2).mean()
        print(f"Levels: {n_levels} | Lloyd-Max MSE: {lm_mse:.4f} | Trained MSE: {trained_mse:.4f}")
        

def lloyd_max_quantization(data, n_levels, max_iter=100, tol=1e-5):
    # Initialization
    min_data, max_data = np.min(data), np.max(data)
    levels = np.linspace(min_data, max_data, n_levels)  # Initial guess for levels
    decision_boundaries = np.zeros(n_levels + 1)

    for _ in range(max_iter):
        # Update decision boundaries
        decision_boundaries[0], decision_boundaries[-1] = min_data, max_data
        decision_boundaries[1:-1] = 0.5 * (levels[:-1] + levels[1:])

        # Update levels
        new_levels = np.array([np.mean(data[(data >= decision_boundaries[i]) & (data < decision_boundaries[i+1])])
                               for i in range(n_levels)])

        # Check for convergence
        if np.linalg.norm(new_levels - levels) < tol:
            break

        levels = new_levels

    return levels, decision_boundaries

def quantize_data_with_lm(data, levels, boundaries):
    quantized_data = np.zeros_like(data)
    for i in range(len(levels)):
        indices = np.where((data >= boundaries[i]) & (data < boundaries[i + 1]))
        quantized_data[indices] = levels[i]
    return quantized_data

# Compare the quantizers
def main():
    M = 10000
    data = torch.randn(M, 1)
    compare_quantizers(data, num_levels_list=[1,2,4,8,16])

if __name__ == "__main__":
    main()