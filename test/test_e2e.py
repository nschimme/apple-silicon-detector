#!/usr/bin/env python3
"""
End-to-end test for the detector.
Starts the detector server and sends a test request.
"""

import subprocess
import time
import sys
import os
import numpy as np
import zmq
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_e2e_test():
    endpoint = "tcp://127.0.0.1:5555"

    # Start the detector in the background
    logger.info("Starting detector server...")
    detector_proc = subprocess.Popen(
        [sys.executable, "detector/zmq_onnx_client.py", "--endpoint", endpoint, "--verbose"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    time.sleep(3)  # Give it time to start

    try:
        # Check if process is still running
        if detector_proc.poll() is not None:
            stdout, _ = detector_proc.communicate()
            logger.error(f"Detector failed to start:\n{stdout}")
            return False

        # Connect to the detector
        logger.info(f"Connecting to detector at {endpoint}...")
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 10000)
        socket.connect(endpoint)

        # 1. Test Model Availability Check
        logger.info("Sending model availability request...")
        header = {"model_request": True, "model_name": "test_model.onnx"}
        socket.send_json(header)
        response = socket.recv_json()
        logger.info(f"Availability response: {response}")

        # 2. Test Inference (should return zero results since no model loaded yet)
        logger.info("Sending inference request (expecting zero results)...")
        test_tensor = np.zeros((320, 320, 3), dtype=np.uint8)
        header = {
            "shape": list(test_tensor.shape),
            "dtype": str(test_tensor.dtype.name),
            "model_type": "yolo-generic"
        }
        socket.send_multipart([json.dumps(header).encode("utf-8"), test_tensor.tobytes()])

        response_frames = socket.recv_multipart()
        resp_header = json.loads(response_frames[0].decode())
        resp_data = np.frombuffer(response_frames[1], dtype=np.float32).reshape(resp_header["shape"])

        logger.info(f"Inference response header: {resp_header}")
        logger.info(f"Inference response data shape: {resp_data.shape}")

        if resp_data.shape == (20, 6):
            logger.info("E2E Test Passed!")
            return True
        else:
            logger.error(f"Unexpected response shape: {resp_data.shape}")
            return False

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return False
    finally:
        logger.info("Stopping detector server...")
        detector_proc.terminate()
        try:
            detector_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            detector_proc.kill()

if __name__ == "__main__":
    success = run_e2e_test()
    sys.exit(0 if success else 1)
