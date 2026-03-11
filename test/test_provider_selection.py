import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Add the detector directory to sys.path to import modules inside it
detector_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "detector")
sys.path.append(detector_dir)

from zmq_onnx_client import get_best_providers, ZmqOnnxClient

class TestProviderSelection(unittest.TestCase):

    @patch('onnxruntime.get_available_providers')
    def test_get_best_providers_mac(self, mock_get_providers):
        # Simulate macOS with CoreML available
        mock_get_providers.return_value = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        providers = get_best_providers()
        self.assertEqual(providers[0], 'CoreMLExecutionProvider')
        self.assertIn('CPUExecutionProvider', providers)

    @patch('onnxruntime.get_available_providers')
    def test_get_best_providers_nvidia(self, mock_get_providers):
        # Simulate Linux with Nvidia CUDA/TensorRT available
        mock_get_providers.return_value = ['TensorRTExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        providers = get_best_providers()
        self.assertEqual(providers[0], 'TensorRTExecutionProvider')
        self.assertEqual(providers[1], 'CUDAExecutionProvider')

    @patch('onnxruntime.get_available_providers')
    def test_get_best_providers_amd(self, mock_get_providers):
        # Simulate Linux with AMD ROCm available
        mock_get_providers.return_value = ['ROCMExecutionProvider', 'CPUExecutionProvider']
        providers = get_best_providers()
        self.assertEqual(providers[0], 'ROCMExecutionProvider')

    @patch('onnxruntime.get_available_providers')
    def test_get_best_providers_intel(self, mock_get_providers):
        # Simulate Linux with OpenVINO available
        mock_get_providers.return_value = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
        providers = get_best_providers()
        self.assertEqual(providers[0], 'OpenVINOExecutionProvider')

    @patch('onnxruntime.get_available_providers')
    def test_get_best_providers_cpu_only(self, mock_get_providers):
        # Simulate CPU only system
        mock_get_providers.return_value = ['CPUExecutionProvider']
        providers = get_best_providers()
        self.assertEqual(len(providers), 1)
        self.assertEqual(providers[0], 'CPUExecutionProvider')

    @patch('onnxruntime.InferenceSession')
    @patch('onnxruntime.get_available_providers')
    def test_zmq_client_auto_providers(self, mock_get_providers, mock_session):
        mock_get_providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        # Mock ZMQ context to avoid actual socket creation
        with patch('zmq.Context'):
            client = ZmqOnnxClient(model_path="test_model.onnx")
            self.assertEqual(client.providers, ['CUDAExecutionProvider', 'CPUExecutionProvider'])

if __name__ == '__main__':
    unittest.main()
