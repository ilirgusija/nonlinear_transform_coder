from model import Quantizer_Gaussian
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal

# Rate Loss Function
def rate(quantized_values, std=1.0):
    normal_dist = Normal(0, std)
    # Calculating negative log-likelihood as a proxy for rate
    nll = -normal_dist.log_prob(quantized_values)
    return torch.mean(nll)

def compare_quantizers(data, num_levels_list, num_runs=10):
    data_np = data.numpy() if isinstance(data, torch.Tensor) else data  # Convert to NumPy array if needed

    for n_levels in num_levels_list:
        lm_mse=0
        trained_mse=0
        for _ in range(num_runs):
            # Lloyd-Max quantization
            # verbose = False if n_levels != 8 else True
            verbose = False
            lm_levels, lm_boundaries = lloyd_max_quantization(data_np, n_levels, verbose=verbose)
            lm_quantized_data = quantize_data_with_lm(data_np, lm_levels, lm_boundaries)

            # Load the trained Quantizer model
            quantizer = Quantizer_Gaussian(data, n_levels)  # Assuming data is appropriate for initializing Quantizer
            quantizer.load_state_dict(torch.load(f'../params/quantizer_params_{n_levels}.pth'))
            quantizer.eval()
            # Quantize data using the trained model
            with torch.no_grad():
                trained_output, compressed_data = quantizer(data)
            
            trained_quantized_data = compressed_data.detach().numpy()  # Convert to NumPy array for MSE calculation

            # Calculate MSE
            lm_mse += np.mean((data_np - lm_quantized_data) ** 2)
            trained_mse += np.mean((data_np - trained_output) ** 2)
            
            # Calculate Rate
            lm_rate+=rate(lm_quantized_data)
            trained_mse+=rate(trained_quantized_data)
            
        print(f"Levels: {n_levels} | Lloyd-Max Distortion: {lm_mse/float(num_runs):.4f} | Trained Distortion: {trained_mse/float(num_runs):.4f}")
    
def plot_rate_distortion(data, lambda_, num_runs=10):
    data_np = data.numpy() if isinstance(data, torch.Tensor) else data  # Convert to NumPy array if needed
    rates_model= []
    distortions_model = []
    for l_ in lambda_:
        trained_mse=0
        trained_rate=0
        for i in range(num_runs):
            # Load the trained Quantizer model
            quantizer = Quantizer_Gaussian(data)  # Assuming data is appropriate for initializing Quantizer
            quantizer.load_state_dict(torch.load(f'../params/quantizer_gauss_params_{i}.pth'))
            quantizer.eval()
            
            # Quantize data using the trained model
            with torch.no_grad():
                trained_output, compressed_data = quantizer(data)
            
            trained_quantized_data = compressed_data.detach().numpy()  # Convert to NumPy array for MSE calculation

            # Calculate distortion and rate
            trained_mse += np.mean((data_np - trained_output) ** 2)
            trained_mse+=rate(trained_quantized_data)
        
        avg_rate = trained_rate/float(num_runs)
        avg_distortion = trained_mse/float(num_runs)
        rates_model.append(avg_rate)
        distortions_model.append(avg_distortion)
            
        print(f"Lambda: {l_} | Trained Rate: {avg_rate:.4f} | Trained Distortion: {avg_distortion:.4f}")    
        
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(distortions_model, rates_model, label='Model')
    plt.ylabel('Bit Rate (bits)')
    plt.xlabel('Distortion (MSE)')
    plt.title('Rate-Distortion Comparison')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(distortions_model, rates_model, label='Model')
    plt.ylabel('Bit Rate (bits)')
    plt.xlabel('Distortion (MSE)')
    plt.title('Rate-Distortion Scatter Plot')
    plt.legend()

    plt.tight_layout()
    plt.savefig('../plots/rate_distortion_gauss.png')
    plt.show()
    
def lloyd_max_quantization(data, n_levels, max_iter=100, tol=1e-5, verbose=False, save_dir="../output"):
    min_data, max_data = np.min(data), np.max(data)
    levels = np.linspace(min_data, max_data, n_levels)
    decision_boundaries = np.zeros(n_levels + 1)

    for iteration in range(max_iter):
        decision_boundaries[0], decision_boundaries[-1] = min_data, max_data
        decision_boundaries[1:-1] = 0.5 * (levels[:-1] + levels[1:])

        new_levels = []
        for i in range(n_levels):
            data_slice = data[(data >= decision_boundaries[i]) & (data < decision_boundaries[i+1])]
            if data_slice.size > 0:
                new_levels.append(np.mean(data_slice))
            else:
                new_levels.append(levels[i])  # Keep the level unchanged if the slice is empty
        
        if verbose:
            visualize_quantization(data, levels, decision_boundaries, iteration, save_dir)

        new_levels = np.array(new_levels)
        if np.linalg.norm(new_levels - levels) < tol:
            break

        levels = new_levels
    if verbose:
        create_animation(save_dir, '../output/quantization_animation.gif', frame_duration=0.5)
        cleanup_frames()

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
    data = torch.randn(M, M)
    lambda_ = [0.01, 0.05, 0.1, 0.5, 1, 2, 4, 6, 8, 10]
    # compare_quantizers(data)
    plot_rate_distortion(data, lambda_)

if __name__ == "__main__":
    main()