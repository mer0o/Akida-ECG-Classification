# Neural Network Quantization and Akida Conversion Pipeline

A comprehensive pipeline for training, quantizing, and converting neural networks to utilise Brainchip's Akida neuromorphic processor to classify ECG Signals.

## Project Structure

```
project_root/
├── Dockerfile           # Container configuration for reproducible environment
├── requirements.txt     # Python dependencies
├── utils.py             # Common utility functions
├── cnn_trainer.py       # CNN model creation and training
├── quantization_handler.py  # Model quantization pipeline
├── akida_converter.py   # Akida conversion and evaluation
├── config.py            # Central configuration parameters
├── __init__.py          # Python package initialization
├── tf_init.py           # TensorFlow initialization and setup
├── models/              # Directory for saved models
│   ├── cnn/             # CNN model artifacts
│   ├── quantized/       # Quantized model versions
├── datasets/            # Datasets storage, ADD THE X_train2.npy HERE BEFORE STARTING.
├── tests/               # Unit and integration tests
└── legacy_combined/     # Initial combined code before separation and refactoring
```

## Setup
1. Manually Add X_train2.npy to the datasetsfolder Before beginning

2. Build the Docker container:
```bash
docker build --platform linux/amd64 -t akida_env .
```

3. Run the container:
```bash
docker run -it --platform linux/amd64 -v $(pwd):/app akida_env
```

## Pipeline Steps

1. **Train CNN Model**
   ```python
   python cnn_trainer.py
   ```
   - Creates and trains a basic CNN model
   - Saves model to `models/cnn/cnn_model.h5`

2. **Quantize Model**
   ```python
   python quantization_handler.py
   ```
   - Converts model to 8-bit quantized format
   - Performs post-quantization fine-tuning
   - Saves quantized model to `models/quantized/quantized_model.h5`

3. **Convert to Akida**
   ```python
   python akida_converter.py
   ```
   - Converts quantized model to Akida format
   - Performs compatibility checks
   - Evaluates conversion accuracy

## Key Features

- Reproducible environment using Docker
- Modular well separated pipeline
- Configurable parameters in central config file
- Comprehensive evaluation metrics
- Data preprocessing utilities
- Visualization tools for model performance

## Requirements (If ran without Docker)

- Python 3.11.0
- TensorFlow 2.15.0
- Akida 2.11.0
- Quantizeml 0.13.1
- Additional dependencies in requirements.txt


## File Descriptions

- **utils.py**: Common functions for data handling, seed setting, and evaluation
- **cnn_trainer.py**: CNN model architecture and training pipeline
- **quantization_handler.py**: Handles model quantization and fine-tuning
- **akida_converter.py**: Manages Akida conversion and evaluation
- **config.py**: Central configuration for model parameters and training settings

## Trials and experiments
| TRIAL                          | EXPECTATIONS                                               | RESULTS                                                                 |
|--------------------------------|-----------------------------------------------------------|-------------------------------------------------------------------------|
| Adding Batch Norm              | To normalised data, should improve accuracy              | ~~accuracy improved slightly.~~ 99.10 → 99.15. Didn’t work and threw the following [error](#). |
| Preprocessing Data             | Appears to be counter intuitive but should normalise inputs | The accuracy after normalisation drops, but gets better accuracy at the end. |
| No Calibration                 | Worse performance. To check if it works                  | Worse performance raw (0.22, 0.12) tuned (0.9709, 0.9708).             |
| Setting seed 42 in final_code  | Should yield stable results                              | Slightly changing, original f1(0.69). Final (0.78, 0.70), same as after quantisation (fitting made no difference). |


## Notes

- Docker container runs on Debian. To use ubuntu container change base image, and download miniconda first.
- Model architecture is optimized for Akida compatibility
- Performance metrics include accuracy, F1 score, and confusion matrices.
- The system can be ran on ARM based processors, however tensorflow is not yet compatible (due to AVX2 instructions). Therefore it is recommended to use the container just for Akida.

## Contributing & Testing

Please ensure any contributions:
1. Follow the existing code structure
2. Include appropriate documentation
3. Create a separate branch with clear naming to keep structure.
