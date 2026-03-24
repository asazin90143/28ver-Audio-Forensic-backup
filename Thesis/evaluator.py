import os
import json
import time

def run_pipeline():
    print("====================================")
    print("Forensic Thesis Results Pipeline")
    print("====================================")
    
    start_time = time.time()
    
    # Run crnn_knn_trainer
    print("\n[1/3] Running CRNN-KNN Training Pipeline...")
    import crnn_knn_trainer
    crnn_knn_trainer.main()
    
    # Run spatial_locator
    print("\n[2/3] Running DOA Spatial Locator...")
    import spatial_locator
    spatial_locator.main()
    
    # Run authenticity_detector
    print("\n[3/3] Running Authenticity Detection...")
    import authenticity_detector
    authenticity_detector.main()
    
    end_time = time.time()
    exec_time = end_time - start_time
    
    print(f"\n====================================")
    print(f"Pipeline finished in {exec_time:.2f} seconds.")
    print("Results have been saved to the Thesis/ directory")
    print("====================================")
    
    # Aggregate summary
    output_dir = os.path.dirname(os.path.abspath(__file__))
    summary = {
        "execution_time_seconds": round(exec_time, 2),
        "status": "success",
        "artifacts_generated": [
            "training_loss_curve.png",
            "event_detection_metrics.json",
            "doa_estimation_scatter.png",
            "doa_error_histogram.png",
            "doa_metrics.json",
            "authenticity_confusion_matrix.png",
            "authenticity_metrics.json"
        ]
    }
    
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

if __name__ == "__main__":
    run_pipeline()
