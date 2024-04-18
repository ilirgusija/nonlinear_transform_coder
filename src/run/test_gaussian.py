from model import Quantizer_Gaussian
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import device_manager, calc_distortion, gen_gaussian_data, calc_empirical_rate

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

def compare_quantizers(data, num_levels_list, num_runs=10):
    data_np = data.numpy() if isinstance(data, torch.Tensor) else data  # Convert to NumPy array if needed

    for n_levels in num_levels_list:
        lm_mse=0
        trained_dist=0
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
            trained_dist += np.mean((data_np - trained_output) ** 2)
            
            # Calculate Rate
            lm_rate+=calc_rate(lm_quantized_data)
            trained_dist+=calc_rate(trained_quantized_data)
            
        print(f"Levels: {n_levels} | Lloyd-Max Distortion: {lm_mse/float(num_runs):.4f} | Trained Distortion: {trained_dist/float(num_runs):.4f}")

##################################################################################################################

def calc_rate_distortion(model, data, device, num_runs=10):
    model.eval()
    
    for _ in range(num_runs):
        dist=0
        rate=0
        for batch in data:
            inputs = batch[0].to(device) 
            # Quantize data using the trained model
            with torch.no_grad():
                trained_output, compressed_data = model(inputs)

            # Calculate distortion and rate
            dist += calc_distortion(inputs, trained_output)
            rate += calc_empirical_rate(compressed_data)
            
    avg_rate = rate/float(num_runs*len(data))
    avg_distortion = dist/float(num_runs*len(data))
        
    return avg_rate, avg_distortion
    
def plot_rate_disortion(distortions_model, rates_model):
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

# Compare the quantizers
def main():
    M = 10000
    data = gen_gaussian_data(M, 1, 16)
    lambda_ = [0.01, 0.05, 0.1, 0.5, 1, 2, 4, 8, 16]
    rates_model= []
    distortions_model = []
    quantizer = Quantizer_Gaussian(N_input=1, N_bottleneck=10, N_output=1)
    quantizer, device = device_manager(quantizer)   
    for i, l_ in enumerate(lambda_):
        # Load the trained Quantizer model
        quantizer.load_state_dict(torch.load(f'../params/quantizer_gauss_params_{i}.pth'))
        quantizer.to(device)
        
        avg_rate, avg_distortion = calc_rate_distortion(quantizer, data, device)
        print(f"Lambda: {l_} | Index: {i} | Trained Rate: {avg_rate:.4f} | Trained Distortion: {avg_distortion:.4f}")    
        
        rates_model.append(avg_rate)
        distortions_model.append(avg_distortion)
            
    distortions_model = [l.cpu().detach().numpy() for l in distortions_model]
    rates_model = [l.cpu().detach().numpy() for l in rates_model]
    plot_rate_disortion(distortions_model, rates_model)
    
if __name__ == "__main__":
    main()