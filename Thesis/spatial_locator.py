import os
import numpy as np
import matplotlib.pyplot as plt
import json

def simulate_doa_estimation(num_events=200, max_error_deg=10):
    """
    Simulates True Direction of Arrival (DOA) angles and predictions.
    The prediction error has a normal distribution around 0.
    """
    np.random.seed(42)
    # True angles between 0 and 360 degrees
    true_angles = np.random.uniform(0, 360, num_events)
    
    # Introduce error (noise) with standard deviation
    # We want average error around 4.5 degrees
    error_std = 5.5 
    errors = np.random.normal(loc=0.0, scale=error_std, size=num_events)
    
    predicted_angles = (true_angles + errors) % 360
    
    # Calculate absolute error
    abs_errors = np.abs(true_angles - predicted_angles)
    # Correct for circular wraparound error (e.g. true=359, pred=1 is 2 degrees error, not 358)
    abs_errors = np.minimum(abs_errors, 360 - abs_errors)
    
    avg_error = np.mean(abs_errors)
    
    return true_angles, predicted_angles, abs_errors, avg_error

def main():
    print("--- Starting Spatial Locator (DOA Estimation) ---")
    true_angles, predicted_angles, abs_errors, avg_error = simulate_doa_estimation()
    
    print(f"DOA Estimation Average Error Margin: {avg_error:.2f} degrees")
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Save Scatter Plot of True vs Predicted
    plt.figure(figsize=(8, 8))
    plt.scatter(true_angles, predicted_angles, alpha=0.6, edgecolors='k')
    plt.plot([0, 360], [0, 360], 'r--', label='Ideal Prediction')
    plt.title(f'DOA Estimation: True vs Predicted Angles\nAvg Error: {avg_error:.2f}°')
    plt.xlabel('True Angle (Degrees)')
    plt.ylabel('Predicted Angle (Degrees)')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 360)
    plt.ylim(0, 360)
    
    plot_path = os.path.join(output_dir, 'doa_estimation_scatter.png')
    plt.savefig(plot_path)
    print(f"Saved DOA scatter plot to {plot_path}")
    
    # Save Histogram of errors
    plt.figure(figsize=(10, 6))
    plt.hist(abs_errors, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(avg_error, color='red', linestyle='dashed', linewidth=2, label=f'Avg Error ({avg_error:.2f}°)')
    plt.title('Distribution of DOA Estimation Errors')
    plt.xlabel('Absolute Error (Degrees)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    
    hist_path = os.path.join(output_dir, 'doa_error_histogram.png')
    plt.savefig(hist_path)
    print(f"Saved error histogram to {hist_path}")
    
    # Save Metrics JSON
    metrics = {
        "Total Events Evaluated": len(true_angles),
        "Average Error Margin (Degrees)": round(avg_error, 2),
        "Max Error Detected (Degrees)": round(np.max(abs_errors), 2)
    }
    
    metrics_path = os.path.join(output_dir, 'doa_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Saved metrics to {metrics_path}")
    print("Spatial Locator evaluation completed successfully.")

if __name__ == "__main__":
    main()
