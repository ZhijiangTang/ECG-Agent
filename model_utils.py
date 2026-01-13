import sys
import os
import shutil
import torch
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample
import matplotlib.pyplot as plt
import io

# Add current directory to sys.path
sys.path.append(os.getcwd())

from models.PSSM import PSSM
from models.head import ClassificationFinetune, DetectionFinetune, ForecastFinetune, GenerationFinetune

class ModelWrapper:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.models = {}
        self.configs = {
            "Classification": {
                "path": "model_weights_backup/classification_best_model.pth",
                "d_model": 16,
                "class": ClassificationFinetune,
                "num_class": 7,
                "args": {"d_model": 16, "patch_len": 1, "oa_mlp_d_model": 500, "hid_mlp": "[1024]", "model_name": "PSSM", "device": device}
            },
            "Detection": {
                "path": "model_weights_backup/detection_best_model.pth",
                "d_model": 16,
                "class": DetectionFinetune,
                "args": {"d_model": 16, "patch_len": 1, "oo_mlp_d_model": 1, "hid_mlp": "[1024]", "model_name": "PSSM", "device": device}
            },
            "Forecasting": {
                "path": "model_weights_backup/forecast_best_model.pth",
                "d_model": 16,
                "class": ForecastFinetune,
                "args": {"d_model": 16, "patch_len": 1, "oa_mlp_d_model": 500, "hid_mlp": "[1024]", "predict_length": 100, "model_name": "PSSM", "device": device}
            },
            "Generation": {
                "path": "model_weights_backup/generation_best_model.pth",
                "d_model": 16,
                "class": GenerationFinetune,
                "args": {"d_model": 16, "patch_len": 1, "oo_mlp_d_model": 1, "hid_mlp": "[1024]", "model_name": "PSSM", "device": device}
            }
        }

    def load_model(self, task):
        if task in self.models:
            return self.models[task]
        
        print(f"Loading model for {task}...")
        cfg = self.configs[task]
        args = cfg["args"].copy()
        args["model_root_path"] = os.path.dirname(os.path.dirname(cfg["path"]))
        args["run_name"] = os.path.basename(os.path.dirname(cfg["path"]))
        args["model_save_name"] = "best_model.pth"
        
        base_model = PSSM(args=args, input_channels=1, d_model=cfg["d_model"])
        
        if task == "Classification":
            model = cfg["class"](args, base_model=base_model, num_class=cfg["num_class"])
        else:
            model = cfg["class"](args, base_model=base_model)
            
        model.to(self.device)
        try:
            model.load_finetune_model()
        except Exception as e:
            print(f"Error loading weights via load_finetune_model: {e}")
            # Fallback: load directly
            sd = torch.load(cfg["path"], map_location=self.device)
            model.load_state_dict(sd, strict=False)
            
        model.eval()
        self.models[task] = model
        print(f"Model for {task} loaded.")
        return model

    def load_and_resample(self, file_objs, target_fs=100):
        """
        Load signal from file(s), validate .dat/.hea pairing, and resample to target_fs.
        Returns: signal (np.array), fs (float)
        """
        if not file_objs:
            raise ValueError("No files uploaded.")
            
        # Organize files by name and extension
        files_map = {}
        temp_dir = os.path.join(os.getcwd(), 'gradio_tmp', 'upload_process')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Copy files to a temp directory to ensure they are together
        for f in file_objs:
            # f is a NamedString or similar from Gradio, f.name is path
            # But we need original filename to pair .dat and .hea
            # Gradio gives file path in temp. But we don't know original name easily if passed as object?
            # Actually Gradio `file_count="multiple"` passes a list of file paths (strings) or file objects depending on type.
            # If type="filepath", it's a list of paths. The paths usually preserve extension but name might be hashed.
            # Wait, Gradio creates temp files. If user uploads 'record.dat' and 'record.hea', 
            # Gradio might save them as '.../record.dat' and '.../record.hea' if uploaded together?
            # Or distinct temp paths.
            # We need to rely on the fact that we can inspect the original filename if possible, 
            # OR we try to group by stem if they are uploaded together.
            # Assuming we just get paths.
            
            # Let's try to copy them to our temp dir.
            # We will use the basename of the temp file.
            shutil.copy(f.name, temp_dir)
            files_map[f.name] = os.path.join(temp_dir, os.path.basename(f.name))

        # Scan the temp dir
        uploaded_files = os.listdir(temp_dir)
        csv_files = [f for f in uploaded_files if f.endswith('.csv')]
        
        signal = None
        fs = None
        
        if csv_files:
            # Prioritize CSV if present (assuming single file or user choice)
            # If multiple CSVs, take first.
            df = pd.read_csv(os.path.join(temp_dir, csv_files[0]))
            signal = df.iloc[:, 0].values
            # CSV usually doesn't have fs. Warn and assume 100 or 500?
            # User requirement: "When original sampling frequency is not available, provide default and log warning"
            print("Warning: CSV file detected. Assuming default sampling rate of 100Hz.")
            fs = 100 # Default
            # If default is 100, no resampling needed if we target 100.
            
        else:
            # Check for valid WFDB pairs (.hea + .dat/.mat)
            valid_pair_found = False
            for f in uploaded_files:
                if f.endswith('.hea'):
                    basename = os.path.splitext(f)[0]
                    # Check for data file (.dat or .mat)
                    if f"{basename}.dat" in uploaded_files or f"{basename}.mat" in uploaded_files:
                        record_path = os.path.join(temp_dir, basename)
                        try:
                            record = wfdb.rdsamp(record_path)
                            # Handle multi-channel: take first channel
                            if record[0].ndim > 1:
                                signal = record[0][:, 0]
                            else:
                                signal = record[0]
                            fs = record[1]['fs']
                            valid_pair_found = True
                            break 
                        except Exception as e:
                            print(f"Failed to read record {basename}: {e}")
                            continue
            
            if not valid_pair_found:
                # Clean up
                shutil.rmtree(temp_dir)
                raise ValueError("Unsupported file format. Please upload .csv or paired WFDB files (.dat/.mat + .hea).")
        
        # Clean up temp dir
        shutil.rmtree(temp_dir)
        
        if signal is None:
             raise ValueError("Failed to load signal.")

        # Resample to target_fs
        if fs != target_fs:
            num_samples = int(len(signal) * target_fs / fs)
            signal = resample(signal, num_samples)
            fs = target_fs
            
        # Normalize
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            
        return signal, fs

    def extract_segment(self, signal, start_index, length=500):
        if start_index < 0 or start_index + length > len(signal):
             raise ValueError("Invalid crop range.")
        return signal[start_index : start_index + length]

    def inference(self, task, signal_segment):
        model = self.load_model(task)
        # Input tensor: (1, 500)
        input_tensor = torch.tensor(signal_segment, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = model(input_tensor)
            
        return output.cpu().numpy()
