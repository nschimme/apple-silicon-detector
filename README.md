# Cross-Platform Detector for Frigate

An optimized object detection client for Frigate that leverages high-performance inference using ONNX Runtime. Provides seamless integration with Frigate's ZMQ detector plugin across macOS (Apple Silicon), Linux (Nvidia/AMD/Intel), and more.

## Features

- **ZMQ IPC/TCP Communication**: Implements the REQ/REP protocol over IPC or TCP endpoints
- **ONNX Runtime Integration**: Runs inference using ONNX models with optimized execution providers
- **Cross-Platform Support**:
  - **macOS**: Optimized for Apple Silicon using CoreML
  - **Linux (Nvidia)**: High-performance inference using CUDA and TensorRT
  - **Linux (AMD)**: GPU acceleration via ROCm and MIGraphX
  - **Linux (Intel/NPUs)**: Optimized for Intel CPUs, GPUs, and NPUs via OpenVINO
  - **Linux (AMD Ryzen AI)**: Optimized for Ryzen AI NPUs (Phoenix, Strix) via Vitis-AI
- **Smart Auto-Detection**: Automatically selects the best available execution provider for your hardware
- **Error Handling**: Robust error handling with fallback to zero results
- **Flexible Deployment**: Supports native execution or Docker containers

## Quick Start

### macOS
1. **Option A: macOS App**
   - Download the latest `FrigateDetector.app.zip` from the Releases page.
   - Unzip it and open `FrigateDetector.app` (first run: right‑click → Open).
2. **Option B: Native**
   ```bash
   make install
   make run
   ```

### Linux (Docker)
The easiest way to run the detector on Linux is via our specialized Docker images:

- **Nvidia GPUs**:
  ```bash
  make docker-build-cuda
  docker run --rm --gpus all frigate-detector:cuda
  ```
- **Intel / Generic NPUs**:
  ```bash
  make docker-build-openvino
  docker run --rm frigate-detector:openvino
  ```
- **AMD GPUs**:
  ```bash
  make docker-build-rocm
  docker run --rm --device=/dev/kfd --device=/dev/dri frigate-detector:rocm
  ```
- **AMD Ryzen AI NPUs**:
  ```bash
  make docker-build-vitisai
  docker run --rm --device=/dev/accel/accel0 frigate-detector:vitisai
  ```
  *Note: Ryzen AI support on Linux requires kernel 6.10+ and the `amdxdna` driver installed on the host.*

The detector will automatically use the model provided by Frigate and start communicating. See [the Frigate documentation](https://deploy-preview-19787--frigate-docs.netlify.app/configuration/object_detectors#apple-silicon-detector) for setup instructions.

## What's Included

- **Model Loading**: Uses whatever model Frigate configures via its automatic model loading
- **Hardware Optimization**: Automatically selects the best provider (CoreML, TensorRT, CUDA, ROCm, OpenVINO, or CPU)
- **Frigate Integration**: Drop-in replacement for Frigate's built-in detectors
- **Multiple Model Support**: YOLOv9, RF-DETR, D-FINE, and custom ONNX models

## Advanced Configuration

### Custom Endpoints
```bash
make run ENDPOINT="tcp://*:5555"
```
Or in Docker:
```bash
docker run -p 5555:5555 frigate-detector:cuda --endpoint tcp://*:5555
```

### List Available Providers
Check which hardware accelerators are detected:
```bash
make run PROVIDERS="--list-providers"
```

### Manual Provider Selection
```bash
make run PROVIDERS="CUDAExecutionProvider CPUExecutionProvider"
```

### Programmatic Usage

```python
from detector.zmq_onnx_client import ZmqOnnxClient

# Create client instance (AUTO detects best providers)
client = ZmqOnnxClient(
    endpoint="tcp://*:5555",
    model_path="/path/to/your/model.onnx"
)

# Start the server
client.start_server()
```

## Performance

- **M3/M4 Optimization**: Leverages Apple's Neural Engine (~8ms for YOLOv9-t)
- **Nvidia TensorRT**: Optimized for low-latency inference on RTX/Tesla GPUs
- **OpenVINO**: Balanced performance across Intel hardware and NPUs
- **Ryzen AI**: Specialized acceleration for AMD's XDNA architecture
- **Async Processing**: Non-blocking ZMQ communication

## Troubleshooting

### Common Issues
- **GPU Not Detected**: Ensure the appropriate drivers (Nvidia/ROCm) are installed on the host and exposed to the container.
- **Permission Denied**: For IPC endpoints, ensure proper permissions on `/tmp/cache/`.
- **ZMQ Bind Failed**: Ensure the endpoint is not already in use.

### Debug Mode
Enable verbose logging:
```bash
make run VERBOSE=1
```

## License

This project is provided as-is for integration with Frigate and ONNX Runtime inference.
