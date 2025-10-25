"""
Vision CNN Workload - Shows layer pipelining.

Semantic optimization:
- Detect convolutional stages
- Pipeline across multiple GPUs
- Overlap computation and communication

Expected result:
- No Semantics: 80ms (sequential layers)
- Full Genie: 60ms (pipelined)
- Speedup: 1.33x
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional
import time


class VisionCNNWorkload:
    """
    ResNet-50 inference - Shows layer pipelining.

    Semantic optimization:
    - Detect convolutional stages
    - Pipeline across multiple GPUs
    - Overlap computation and communication
    """

    def __init__(self, model_name: str = "microsoft/resnet-50", batch_size: int = 8):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.processor = None

    def load_model(self):
        """Load model from HuggingFace."""
        try:
            from transformers import AutoImageProcessor, ResNetForImageClassification

            print(f"Loading {self.model_name}...")
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = ResNetForImageClassification.from_pretrained(self.model_name)
            self.model.eval()

            print(f"✓ Loaded {self.model_name}")
            return True

        except ImportError:
            print("❌ transformers not available - using mock model")
            self.model = self._create_mock_resnet()
            self.processor = self._create_mock_processor()
            return False

    def _create_mock_resnet(self):
        """Create a simple mock ResNet model."""
        class MockResNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

                # Multiple conv layers for pipelining demo
                self.layer1 = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                )
                self.layer2 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                )
                self.layer3 = nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                )
                self.layer4 = nn.Sequential(
                    nn.Conv2d(256, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                )

                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(512, 1000)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)

                return type('Output', (), {'logits': x})()

        return MockResNet()

    def _create_mock_processor(self):
        """Create a simple mock processor."""
        class MockProcessor:
            def __call__(self, image, return_tensors='pt'):
                # Mock processing - handle both None and actual image tensors
                if image is None:
                    # Fallback for None input
                    return {'pixel_values': torch.randn(1, 3, 224, 224)}
                else:
                    # Process the provided image tensor
                    if image.dim() == 3:  # Single image (C, H, W)
                        return {'pixel_values': image.unsqueeze(0).float() / 255.0}
                    else:  # Batch of images (B, C, H, W)
                        return {'pixel_values': image.float() / 255.0}

        return MockProcessor()

    def get_sample_inputs(self) -> List[torch.Tensor]:
        """Get sample inputs for the workload."""
        if self.processor is None:
            self.load_model()

        # Create mock image input - use proper mock image instead of None
        if hasattr(self.processor, '__call__'):
            # Create a mock image tensor (random RGB image)
            mock_image = torch.randn(1, 3, 224, 224) * 255
            mock_image = torch.clamp(mock_image, 0, 255).byte()

            inputs = self.processor(mock_image, return_tensors='pt')
            # Expand to batch size
            pixel_values = inputs['pixel_values']
            if pixel_values.dim() == 3:
                pixel_values = pixel_values.unsqueeze(0)
            pixel_values = pixel_values.repeat(self.batch_size, 1, 1, 1)
            return [pixel_values]
        else:
            # Fallback
            return [torch.randn(self.batch_size, 3, 224, 224)]

    def run(self, baseline_config: Dict) -> Dict[str, Any]:
        """
        Run vision CNN workload.

        Key metrics:
        - Total inference latency
        - Throughput (images/second)
        - Memory usage
        """
        if self.model is None:
            self.load_model()

        # Get inputs
        inputs = self.get_sample_inputs()
        pixel_values = inputs[0]

        print(f"CNN input: {pixel_values.shape} (batch_size={pixel_values.shape[0]})")

        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = self.model(pixel_values)

        end_time = time.perf_counter()

        return {
            'batch_size': pixel_values.shape[0],
            'input_shape': list(pixel_values.shape),
            'output_classes': outputs.logits.shape[1],
            'total_latency_ms': (end_time - start_time) * 1000,
            'throughput_images_per_sec': pixel_values.shape[0] / (end_time - start_time),
            'baseline': baseline_config.get('baseline', 'unknown'),
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Get workload metadata."""
        return {
            'workload': 'vision_cnn',
            'model': self.model_name,
            'batch_size': self.batch_size,
            'description': 'Vision CNN inference (ResNet-50)',
            'key_optimization': 'layer_pipelining',
            'expected_speedup': '1.2x vs no semantics',
            'parallel_dimension': 'batch_size'
        }
