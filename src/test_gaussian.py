from model import Quantizer_Gaussian
import torch
import numpy as np

def compare_quantizers(data, num_levels_list):
    data_np = data.numpy() if isinstance(data, torch.Tensor) else data  # Convert to NumPy array if needed

    for n_levels in num_levels_list:
        # Lloyd-Max quantization
        lm_levels, lm_boundaries = lloyd_max_quantization(data_np, n_levels)
        lm_quantized_data = quantize_data_with_lm(data_np, lm_levels, lm_boundaries)

        # Load the trained Quantizer model
        quantizer = Quantizer_Gaussian(data, n_levels)  # Assuming data is appropriate for initializing Quantizer
        quantizer.load_state_dict(torch.load(f'../params/quantizer_params_{n_levels}.pt'))

        # Quantize data using the trained model
        trained_quantized_data = quantizer(data).detach().numpy()  # Convert to NumPy array for MSE calculation

        # Calculate MSE
        lm_mse = np.mean((data_np - lm_quantized_data) ** 2)
        trained_mse = np.mean((data_np - trained_quantized_data) ** 2)
        print(f"Levels: {n_levels} | Lloyd-Max MSE: {lm_mse:.4f} | Trained MSE: {trained_mse:.4f}")
        

def lloyd_max_quantization(data, n_levels, max_iter=100, tol=1e-5):
    min_data, max_data = np.min(data), np.max(data)
    levels = np.linspace(min_data, max_data, n_levels)
    decision_boundaries = np.zeros(n_levels + 1)

    for _ in range(max_iter):
        decision_boundaries[0], decision_boundaries[-1] = min_data, max_data
        decision_boundaries[1:-1] = 0.5 * (levels[:-1] + levels[1:])

        new_levels = []
        for i in range(n_levels):
            data_slice = data[(data >= decision_boundaries[i]) & (data < decision_boundaries[i+1])]
            if data_slice.size > 0:
                new_levels.append(np.mean(data_slice))
            else:
                new_levels.append(levels[i])  # Keep the level unchanged if the slice is empty

        new_levels = np.array(new_levels)
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