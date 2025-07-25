#!/usr/bin/env python3
"""
Visualize world model video predictions
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pathlib


def load_and_visualize_predictions(npz_path, output_dir=None, save_frames=True):
    """Load and visualize video predictions from npz file."""
    
    if output_dir is None:
        output_dir = npz_path.parent / 'prediction_frames'
    else:
        output_dir = pathlib.Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading predictions from: {npz_path}")
    
    # Load the predictions
    data = np.load(npz_path)
    
    # Get the first prediction sequence
    prediction_keys = list(data.keys())
    print(f"Found {len(prediction_keys)} prediction sequences")
    
    # Use first sequence
    video_data = data[prediction_keys[0]]
    print(f"Video data shape: {video_data.shape}")
    
    # video_data shape: [batch, time, height, width*3, channels]
    # The width is 3x because it contains: truth | model | error
    batch_size, time_steps, height, width_combined, channels = video_data.shape
    width = width_combined // 3

    print(f"Batch size: {batch_size}")
    print(f"Time steps: {time_steps}")
    print(f"Frame size: {height}x{width}")
    print(f"Channels: {channels}")

    # The data seems to be concatenated vertically, not horizontally
    # Let's check if height is actually height*3
    if height > width:
        print("Detected vertical concatenation - fixing dimensions...")
        # Reshape: height is actually height*3, width is the real width
        real_height = height // 3
        real_width = width_combined

        # Reshape the data
        video_data = video_data.reshape(batch_size, time_steps, 3, real_height, real_width, channels)

        # Split into truth, model, error
        truth = video_data[:, :, 0, :, :, :]  # First third
        model = video_data[:, :, 1, :, :, :]  # Second third
        error = video_data[:, :, 2, :, :, :]  # Third third

        print(f"Fixed frame size: {real_height}x{real_width}")
    else:
        # Original horizontal concatenation
        truth = video_data[:, :, :, :width, :]
        model = video_data[:, :, :, width:2*width, :]
        error = video_data[:, :, :, 2*width:, :]
    
    # Use first batch
    truth = truth[0]  # [time, height, width, channels]
    model = model[0]
    error = error[0]
    
    print(f"Processing {time_steps} frames...")
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('World Model Video Predictions', fontsize=16)
    
    axes[0].set_title('Ground Truth')
    axes[1].set_title('Model Prediction')
    axes[2].set_title('Prediction Error')
    
    for ax in axes:
        ax.axis('off')
    
    # Save individual frames if requested
    if save_frames:
        frames_dir = output_dir / 'individual_frames'
        frames_dir.mkdir(exist_ok=True)
        
        for t in range(time_steps):
            # Create frame comparison with proper aspect ratio
            fig_frame, axes_frame = plt.subplots(1, 3, figsize=(12, 4))
            fig_frame.suptitle(f'Frame {t+1}/{time_steps}', fontsize=14)
            
            axes_frame[0].imshow(truth[t])
            axes_frame[0].set_title('Ground Truth')
            axes_frame[0].axis('off')
            
            axes_frame[1].imshow(model[t])
            axes_frame[1].set_title('Model Prediction')
            axes_frame[1].axis('off')
            
            axes_frame[2].imshow(error[t])
            axes_frame[2].set_title('Prediction Error')
            axes_frame[2].axis('off')
            
            plt.tight_layout()
            frame_path = frames_dir / f'frame_{t:03d}.png'
            plt.savefig(frame_path, dpi=150, bbox_inches='tight')
            plt.close(fig_frame)
            
            if t % 5 == 0:
                print(f"Saved frame {t+1}/{time_steps}")
    
    # Create summary visualization with multiple time steps
    n_display = min(6, time_steps)  # Reduce number for better layout
    step_size = max(1, time_steps // n_display)
    display_times = list(range(0, time_steps, step_size))[:n_display]

    # Better figure sizing based on aspect ratio
    fig_width = n_display * 3  # 3 inches per time step
    fig_height = 9  # 3 inches per row
    fig_summary, axes_summary = plt.subplots(3, n_display, figsize=(fig_width, fig_height))
    fig_summary.suptitle('World Model Predictions Over Time', fontsize=16)
    
    if n_display == 1:
        axes_summary = axes_summary.reshape(3, 1)
    
    for i, t in enumerate(display_times):
        axes_summary[0, i].imshow(truth[t])
        axes_summary[0, i].set_title(f'Truth t={t}')
        axes_summary[0, i].axis('off')
        
        axes_summary[1, i].imshow(model[t])
        axes_summary[1, i].set_title(f'Model t={t}')
        axes_summary[1, i].axis('off')
        
        axes_summary[2, i].imshow(error[t])
        axes_summary[2, i].set_title(f'Error t={t}')
        axes_summary[2, i].axis('off')
    
    # Add row labels
    axes_summary[0, 0].set_ylabel('Ground Truth', rotation=90, size='large')
    axes_summary[1, 0].set_ylabel('Model Prediction', rotation=90, size='large')
    axes_summary[2, 0].set_ylabel('Prediction Error', rotation=90, size='large')
    
    plt.tight_layout()
    
    # Save summary
    summary_path = output_dir / 'prediction_summary.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"Summary visualization saved to: {summary_path}")
    
    # Show the plot
    plt.show()
    
    return {
        'truth': truth,
        'model': model, 
        'error': error,
        'summary_path': summary_path,
        'frames_dir': frames_dir if save_frames else None
    }


def main():
    parser = argparse.ArgumentParser(description='Visualize world model video predictions')
    parser.add_argument('--npz-path', type=str, required=True,
                       help='Path to video_predictions.npz file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for visualizations')
    parser.add_argument('--no-frames', action='store_true',
                       help='Skip saving individual frames')
    
    args = parser.parse_args()
    
    npz_path = pathlib.Path(args.npz_path)
    if not npz_path.exists():
        print(f"Error: File not found: {npz_path}")
        return
    
    results = load_and_visualize_predictions(
        npz_path, 
        args.output_dir, 
        save_frames=not args.no_frames
    )
    
    print(f"\nVisualization complete!")
    print(f"Summary saved to: {results['summary_path']}")
    if results['frames_dir']:
        print(f"Individual frames saved to: {results['frames_dir']}")


if __name__ == "__main__":
    main()
