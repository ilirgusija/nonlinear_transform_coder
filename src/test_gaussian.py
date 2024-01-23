from model import Quantizer_Gaussian
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio

def cleanup_frames(directory="../output"):
    for file_name in os.listdir(directory):
        if file_name.endswith('.png'):
            os.remove(os.path.join(directory, file_name))
   
def visualize_quantization(data, levels, decision_boundaries, iteration, save_dir):
    plt.figure(figsize=(10, 6))
    plt.title(f"Lloyd-Max Quantization - Iteration {iteration}")
    plt.scatter(data, np.zeros_like(data), label='Data Points', alpha=0.5)
    plt.scatter(levels, np.zeros_like(levels), color='red', label='Quantization Levels')
    for boundary in decision_boundaries:
        plt.axvline(x=boundary, color='green', linestyle='--')
    plt.legend()

    # Save the plot as an image
    if iteration < 10:
        iteration = "0" + str(iteration)
    filename = os.path.join(save_dir, f'iteration_{iteration}.png')
    plt.savefig(filename)
    plt.close()
    
def create_animation(save_dir, output_file, frame_duration):
    images = []
    for file_name in sorted(os.listdir(save_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(save_dir, file_name)
            images.append(imageio.imread(file_path))
    
    # Create a GIF
    imageio.mimsave(output_file, images, duration=frame_duration)
        
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
            trained_quantized_data = quantizer(data).detach().numpy()  # Convert to NumPy array for MSE calculation

            # Calculate MSE
            lm_mse += np.mean((data_np - lm_quantized_data) ** 2)
            trained_mse += np.mean((data_np - trained_quantized_data) ** 2)
        print(f"Levels: {n_levels} | Lloyd-Max Distortion: {lm_mse/float(num_runs):.4f} | Trained Distortion: {trained_mse/float(num_runs):.4f}")
        
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
    data = torch.randn(M, 1)
    compare_quantizers(data, num_levels_list=[1,2,4,8,16])

if __name__ == "__main__":
    main()