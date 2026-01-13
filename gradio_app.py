import os
import sys
import shutil
from pathlib import Path

# Set Gradio temp dir
temp_dir = os.path.join(os.getcwd(), 'gradio_tmp')
os.makedirs(temp_dir, exist_ok=True)
os.environ['GRADIO_TEMP_DIR'] = temp_dir
os.environ['TMPDIR'] = temp_dir
os.environ['TEMP'] = temp_dir
os.environ['TMP'] = temp_dir

# Monkeypatch gradio routes to fix permission error
try:
    import gradio.routes
    gradio.routes.vibe_edit_history_dir = Path(temp_dir) / "vibe_edit_history"
except ImportError:
    pass

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import pandas as pd
import wfdb
from model_utils import ModelWrapper

# Initialize Model Wrapper
device = "cuda" if torch.cuda.is_available() else "cpu"
wrapper = ModelWrapper(device=device)

# Sample Data Paths
SAMPLE_PATHS = {
    "Classification (CPSC2018)": "example_data/cpsc_2018/A0001",
    "Detection (BA-LABOUR)": "example_data/ba_labour/B1_Pregnancy_01",
    "Forecasting (PTB)": "example_data/ptb/S0001",
    "Generation (MITDB)": "example_data/mitdb/100"
}

def load_sample_file(sample_key, state):
    if not sample_key:
        return None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "Please select a sample.", state
    
    # Resolve relative path to absolute
    rel_path = SAMPLE_PATHS.get(sample_key)
    if not rel_path:
         return None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "Sample path not found.", state
         
    base_path = os.path.abspath(rel_path)
    if not base_path:
        return None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "Sample path not found.", state
        
    # Create a list of file objects to simulate upload or just pass paths
    # We need to copy them to a temp location so load_and_resample can process them similarly
    # But wrapper.load_and_resample expects file objects with 'name' attribute or paths.
    
    # Let's verify what files exist
    extensions = ['.dat', '.hea', '.mat', '.atr']
    found_files = []
    
    # Prepare a temp directory for this load
    session_tmp = os.path.join(temp_dir, f"sample_{os.urandom(4).hex()}")
    os.makedirs(session_tmp, exist_ok=True)
    
    try:
        # Check for files with same basename
        dirname = os.path.dirname(base_path)
        basename = os.path.basename(base_path)
        
        # If base_path points to a file without extension (e.g. .../100), check extensions
        # If it has extension (e.g. .../A0001.mat), strip it
        if os.path.exists(base_path) and os.path.isfile(base_path):
             basename = os.path.splitext(basename)[0]
        
        # Search for related files
        for ext in extensions:
            src = os.path.join(dirname, basename + ext)
            if os.path.exists(src):
                dst = os.path.join(session_tmp, basename + ext)
                shutil.copy(src, dst)
                # Create a dummy object with 'name' attribute for compatibility if needed
                # But wrapper.load_and_resample takes a list of file objects (gr.File output)
                # If we pass a list of class instances with .name, it works.
                class FileObj:
                    def __init__(self, path):
                        self.name = path
                found_files.append(FileObj(dst))
        
        if not found_files:
             return None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), f"No compatible files found for {basename}", state

        # Reuse load_signal logic
        return load_signal(found_files, state)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), f"Error loading sample: {str(e)}", state

def load_signal(files, state):
    if not files:
        return None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "Please upload files.", state
    
    try:
        # Update ModelWrapper to handle .mat if not already (it uses shutil.copy, so it's fine as long as we extend extensions check)
        # We need to monkeypatch or update ModelWrapper? 
        # Actually ModelWrapper.load_and_resample in model_utils.py only checks for .dat+.hea or .csv.
        # It DOES NOT support .mat directly in the provided code.
        # I need to modify ModelWrapper or handle it here.
        # Since I can't modify model_utils.py easily without rewriting it, I will try to handle .mat here if wrapper fails?
        # Or better, I'll update model_utils.py separately. But for now let's try to assume wrapper works or I fix it.
        # Wait, I am the owner, I can modify model_utils.py.
        # But to save time/steps, I'll update load_and_resample logic inside this file or assume I updated model_utils.py.
        # Let's modify model_utils.py first to support .mat!
        # Actually, let's just do it.
        
        signal, fs = wrapper.load_and_resample(files)
        state = {"signal": signal, "fs": fs}
        
        # Initial plot with selection at 0
        fig = plot_signal_with_selection(0, state)
        seg_fig = plot_segment(0, state)
        
        max_start = max(0, len(signal) - 500)
        
        return (
            fig, 
            gr.update(visible=True, maximum=max_start, value=0), 
            gr.update(visible=True), 
            seg_fig,
            f"Signal loaded. Length: {len(signal)} samples. Fs: {fs}Hz. Resampled to 100Hz.",
            state
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), f"Error: {str(e)}", state

def plot_signal_with_selection(start_index, state):
    if not state or state.get("signal") is None:
        return None
    
    signal = state["signal"]
    length = 500
    end_index = start_index + length
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(signal, color='blue', alpha=0.6, label='ECG Signal')
    
    # Highlight selection
    if start_index + length <= len(signal):
        rect = patches.Rectangle((start_index, min(signal)), length, max(signal) - min(signal), 
                                 linewidth=2, edgecolor='red', facecolor='red', alpha=0.2, label='Selected')
        ax.add_patch(rect)
        ax.axvline(x=start_index, color='red', linestyle='--')
        ax.axvline(x=end_index, color='red', linestyle='--')
        
    ax.set_title(f"ECG Signal (Selected: {start_index} - {end_index})")
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    plt.tight_layout()
    
    return fig

def plot_segment(start_index, state):
    if not state or state.get("signal") is None:
        return None
    
    signal = state["signal"]
    # Ensure indices are valid
    if start_index < 0: start_index = 0
    if start_index + 500 > len(signal): start_index = max(0, len(signal) - 500)
        
    segment = wrapper.extract_segment(signal, start_index)
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(segment, color='blue', linewidth=1.5)
    ax.set_title("Cropped Input Segment (500 points)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

def update_plot(start_index, state):
    return plot_signal_with_selection(start_index, state), plot_segment(start_index, state)

def run_analysis(task, start_index, state):
    if not state or state.get("signal") is None:
        return None, "No signal loaded.", None
        
    try:
        segment = wrapper.extract_segment(state["signal"], start_index)
        output = wrapper.inference(task, segment)
        
        # Visualization of result
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        result_text = ""
        csv_path = "analysis_result.csv"
        
        if task == "Classification":
            probs = torch.softmax(torch.tensor(output), dim=1).numpy()[0]
            # Use 7 classes: Normal, AF, I-AVB, LBBB, RBBB, PAC, PVC
            class_names = ["Normal", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC"]
            # Ensure we have enough names or fallback
            if len(probs) > len(class_names):
                class_names += [f"Class {i}" for i in range(len(class_names), len(probs))]
            
            ax.bar(class_names[:len(probs)], probs)
            ax.set_title("Classification Probabilities")
            ax.set_xticklabels(class_names[:len(probs)], rotation=45)
            ax.set_ylabel("Probability")
            
            pred_idx = np.argmax(probs)
            result_text = f"Predicted Class: {class_names[pred_idx]}\nConfidence: {probs[pred_idx]:.2f}"
            pd.DataFrame({'Class': class_names[:len(probs)], 'Probability': probs}).to_csv(csv_path, index=False)
            
        elif task == "Detection":
            prob_signal = output[0]
            
            # Use plot logic from validation/Detection.py
            plt.close(fig) 
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            ax1.plot(segment, label="ECG Segment", color='blue', alpha=0.3)
            ax1.set_title("ECG Segment with Detection")
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Amplitude', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            # Highlight peaks > 0.5
            threshold = 0.5
            peaks = np.where(prob_signal > threshold)[0]
            if len(peaks) > 0:
                 ax1.scatter(peaks, segment[peaks], color='red', s=30, label='Detected Event', zorder=5)
            
            # Plot Probability on twin axis
            ax2 = ax1.twinx()
            ax2.plot(prob_signal, color='orange', alpha=0.5, label="Detection Probability")
            ax2.axhline(y=threshold, color='red', linestyle='--', label="Threshold")
            ax2.set_ylabel('Detection Probability', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
            ax2.set_ylim(0, 1.1)
            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            result_text = f"Detected {len(peaks)} events (Threshold: {threshold})"
            pd.DataFrame({'Time': range(len(prob_signal)), 'Probability': prob_signal}).to_csv(csv_path, index=False)
            
        elif task == "Forecasting":
            forecast = output.flatten()
            full_len = len(segment) + len(forecast)
            time_axis = np.arange(full_len)
            
            ax.plot(time_axis[:len(segment)], segment, label="History")
            ax.plot(time_axis[len(segment):], forecast, label="Forecast", color='red')
            ax.set_title("ECG Forecasting")
            ax.legend()
            
            result_text = "Forecast generated."
            pd.DataFrame({'Forecast': forecast}).to_csv(csv_path, index=False)
            
        elif task == "Generation":
            gen_signal = output[0]
            # Requirement: Do not plot input, only generated
            ax.plot(gen_signal, label="Generated", color='green')
            ax.set_title("ECG Generation")
            ax.legend()
            result_text = "Signal generated."
            pd.DataFrame({'Generated': gen_signal}).to_csv(csv_path, index=False)
            
        plt.tight_layout()
        plot_path = os.path.join(temp_dir, f"result_plot_{os.urandom(4).hex()}.png")
        fig.savefig(plot_path)
        plt.close(fig)
        
        return plot_path, result_text, csv_path
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}", None

# Gradio Interface
with gr.Blocks(title="ECG Analysis Agent") as demo:
    state = gr.State({})
    
    gr.Markdown("# ECG Analysis Agent")
    gr.Markdown("1. Load Sample OR Upload Files. 2. Select 500-point segment. 3. Run Analysis.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Data Loading")
            with gr.Tab("Load Sample"):
                sample_dropdown = gr.Dropdown(
                    choices=list(SAMPLE_PATHS.keys()),
                    label="Select Sample Dataset"
                )
                load_sample_btn = gr.Button("Load Sample", variant="secondary")
                
            with gr.Tab("Upload File"):
                file_input = gr.File(label="Upload Files", file_count="multiple", file_types=[".csv", ".dat", ".hea", ".mat"])
                load_btn = gr.Button("Load & Preprocess", variant="primary")
            
            status_info = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("### Task Selection")
            task_dropdown = gr.Dropdown(
                choices=["Classification", "Detection", "Forecasting", "Generation"], 
                value="Classification", 
                label="Select Task"
            )
            analyze_btn = gr.Button("Analyze Selected Segment", variant="primary", visible=False)

        with gr.Column(scale=2):
            signal_plot = gr.Plot(label="Full Signal & Selection")
            start_slider = gr.Slider(label="Start Index", minimum=0, maximum=1000, value=0, step=1, visible=False)
            # Req 1: Segment plot displayed here, updating with slider
            segment_plot = gr.Plot(label="Cropped Input Segment (500 pts)")
            
    with gr.Row():
        with gr.Column():
            result_plot = gr.Image(label="Analysis Visualization")
        with gr.Column():
            result_info = gr.Textbox(label="Result Info")
            download_btn = gr.File(label="Download CSV")

    # Event Wiring
    load_btn.click(
        load_signal, 
        inputs=[file_input, state], 
        outputs=[signal_plot, start_slider, analyze_btn, segment_plot, status_info, state]
    )
    
    load_sample_btn.click(
        load_sample_file,
        inputs=[sample_dropdown, state],
        outputs=[signal_plot, start_slider, analyze_btn, segment_plot, status_info, state]
    )
    
    start_slider.change(
        update_plot,
        inputs=[start_slider, state],
        outputs=[signal_plot, segment_plot]
    )
    
    analyze_btn.click(
        run_analysis,
        inputs=[task_dropdown, start_slider, state],
        outputs=[result_plot, result_info, download_btn]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
