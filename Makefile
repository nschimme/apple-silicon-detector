VENV := venv
PYTHON := python3.11
PIP := $(VENV)/bin/pip3
PY := $(VENV)/bin/python3

# Docker configuration
DOCKER_IMAGE_NAME := frigate-detector
DOCKER_TAG := latest

.PHONY: help venv install reinstall clean run docker-build-cuda docker-build-openvino docker-build-rocm docker-run

help:
	@echo "Targets:"
	@echo "  venv       - Create local virtual environment in $(VENV)/"
	@echo "  install    - Create venv (if needed) and install dependencies"
	@echo "  run        - Run the ZMQ ONNX client"
	@echo ""
	@echo "Docker Targets:"
	@echo "  docker-build-cuda     - Build Docker image for Nvidia GPUs"
	@echo "  docker-build-openvino - Build Docker image for Intel/Generic NPUs"
	@echo "  docker-build-rocm     - Build Docker image for AMD GPUs"
	@echo "  docker-build-vitisai  - Build Docker image for AMD Ryzen AI NPUs"
	@echo "  docker-run            - Run the Docker container (requires DOCKER_IMAGE_NAME and DOCKER_TAG)"
	@echo ""
	@echo "Examples:"
	@echo "  make install"
	@echo "  make run MODEL=/path/to/model.onnx"
	@echo "  make docker-build-cuda"
	@echo "  docker run --rm --gpus all frigate-detector:cuda"

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run: venv
	$(PY) detector/zmq_onnx_client.py $(if $(MODEL),--model $(MODEL),) $(if $(ENDPOINT),--endpoint $(ENDPOINT),) $(if $(PROVIDERS),--providers $(PROVIDERS),) $(if $(VERBOSE),-v,)

reinstall: clean install

clean:
	rm -rf $(VENV)

# Docker builds
docker-build-cuda:
	docker build -t $(DOCKER_IMAGE_NAME):cuda -f docker/Dockerfile.cuda .

docker-build-openvino:
	docker build -t $(DOCKER_IMAGE_NAME):openvino -f docker/Dockerfile.openvino .

docker-build-rocm:
	docker build -t $(DOCKER_IMAGE_NAME):rocm -f docker/Dockerfile.rocm .

docker-build-vitisai:
	docker build -t $(DOCKER_IMAGE_NAME):vitisai -f docker/Dockerfile.vitisai .

docker-run:
	docker run --rm -it $(DOCKER_IMAGE_NAME):$(DOCKER_TAG)

# Verification
verify-openvino:
	pip uninstall -y onnxruntime || true
	sed -i 's/onnxruntime==/onnxruntime-openvino==/' requirements.txt
	pip install -r requirements.txt
	python3 test/test_e2e.py
