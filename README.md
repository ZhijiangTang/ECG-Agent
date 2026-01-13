# ECG Agent

This repository provides tools for [[💼ECG analysis]](https://github.com/ZhijiangTang/ECG-Benchmark), including an interactive Gradio web interface and a unified test suite for verifying model performance across multiple tasks (Classification, Detection, Forecasting, Generation).

## Prerequisites

Ensure you have the required Python packages installed:

```bash
pip install gradio wfdb torch pandas numpy matplotlib scipy scikit-learn omegaconf hydra-core
```

## Hardware Requirements

*   **Inference**: CPU or any CUDA-capable GPU (VRAM > 2GB).
*   **Training**: NVIDIA GPU with VRAM > 6GB recommended (tested on RTX 3090).

## Roadmap

*   [ ] Introduce visualization interface for model training process.

## 1. Interactive Web Interface (Gradio)

The Gradio app provides a user-friendly interface to visualize ECG signals, crop segments, and run analysis tasks.

### How to Run

1.  Start the application:
    ```bash
    python gradio_app.py
    ```

2.  Access the interface:
    *   Open your web browser and go to: `http://localhost:7860` (or the URL provided in the terminal).

### Usage Steps

1.  **Upload**: Drag and drop your `.dat` and `.hea` files (must be paired) into the upload box.
2.  **Load**: Click "Load & Preprocess". The signal will be resampled to 100Hz and displayed.
3.  **Select**: Use the "Start Index" slider to choose a 500-point window. The selected red region will update on the chart.
4.  **Analyze**: Select a task (e.g., Classification) from the dropdown and click "Analyze Selected Segment".
5.  **View Results**: The result visualization and text info will appear on the right. You can also download the result CSV.

## 2. Unified Test Suite

The unified test script verifies the functionality of all 4 tasks using a standard test file and generates a report.

### How to Run

Execute the test script:
```bash
python unified_test.py
```

### Output

*   **Console**: Logs of the loading and inference process.
*   **Report**: A `test_report.md` file summarizing the results.
*   **Visualization**: A `visualization.png` image showing the signal processing steps.

## 3. Model Weights

The model weights used by these applications are stored in `model_weights_backup/`. 
Refer to [MANIFEST.md](model_weights_backup/MANIFEST.md) for details on the source checkpoints and configurations.

## 🧑‍🤝‍🧑 Brothers link     
*   [[💼 ECG-Benchmark]](https://github.com/ZhijiangTang/ECG-Benchmark)
*   [[🤖 ECG-Agent]](https://github.com/ZhijiangTang/ECG-Agent)
*   [[📖 A Comprehensive Benchmark for Electrocardiogram Time-Series]](https://arxiv.org/abs/2507.14206)

## 📝 Citation

If you find this repo helpful, please cite our paper:

```bibtex
@inproceedings{tang2025comprehensive,
  title={A Comprehensive Benchmark for Electrocardiogram Time-Series},
  author={Tang, Zhijiang and Qi, Jiaxin and Zheng, Yuhua and Huang, Jianqiang},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={6490--6499},
  year={2025}
}
```
