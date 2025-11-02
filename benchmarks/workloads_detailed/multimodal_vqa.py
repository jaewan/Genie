"""
Multimodal VQA Workload - Shows parallel branch execution.

Semantic optimization:
- Detect independent vision and text branches
- Execute in parallel on different GPUs
- Only synchronize at fusion point

Expected result:
- No Semantics: 150ms (sequential)
- Full Genie: 100ms (parallel)
- Speedup: 1.5x
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional
import time


class MultimodalVQAWorkload:
    """
    CLIP-based VQA - Shows parallel branch execution.

    Semantic optimization:
    - Detect independent vision and text branches
    - Execute in parallel on different GPUs
    - Only synchronize at fusion point
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", batch_size: int = 4):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.processor = None

    def load_model(self):
        """Load model from HuggingFace."""
        try:
            from transformers import CLIPProcessor, CLIPModel

            print(f"Loading {self.model_name}...")
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            
            # CRITICAL FIX: Load without device_map, use explicit placement
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.model.eval()

            # CRITICAL FIX: Ensure model is on cuda:0
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                self.model.to(device)

            print(f"✓ Loaded {self.model_name}")
            return True

        except ImportError:
            print("❌ transformers not available - using mock model")
            self.model = self._create_mock_clip()
            self.processor = self._create_mock_processor()
            
            # CRITICAL FIX: Keep mock model on CPU to avoid GPU OOM
            # Real benchmarks will use real models on GPU
            # self.model stays on CPU for validation
            
            return False

    def _create_mock_clip(self):
        """Create a simple mock CLIP model."""
        class MockCLIP(nn.Module):
            def __init__(self):
                super().__init__()
                # Vision encoder
                self.vision_model = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=16, stride=16),  # Patch embedding
                    nn.Flatten(),
                    nn.Linear(64 * 14 * 14, 512),  # Vision features
                )

                # Text encoder
                self.text_model = nn.Sequential(
                    nn.Embedding(49408, 512),  # Text embeddings
                    nn.Linear(512, 512),  # Text features
                )

                # Fusion
                self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

            def forward(self, input_ids=None, pixel_values=None, **kwargs):
                vision_features = None
                text_features = None

                if pixel_values is not None:
                    vision_features = self.vision_model(pixel_values)

                if input_ids is not None:
                    text_embeddings = self.text_model[0](input_ids)
                    text_features = self.text_model[1](text_embeddings)

                if vision_features is not None and text_features is not None:
                    # Compute similarity
                    vision_features = vision_features / vision_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    # Cosine similarity
                    logits_per_image = vision_features @ text_features.t() * self.logit_scale.exp()
                    logits_per_text = logits_per_image.t()

                    return type('Output', (), {
                        'logits_per_image': logits_per_image,
                        'logits_per_text': logits_per_text
                    })()

                return type('Output', (), {'logits': torch.randn(1, 1000)})()

        return MockCLIP()

    def _create_mock_processor(self):
        """Create a simple mock processor."""
        class MockProcessor:
            def __init__(self):
                self.tokenizer = self._create_mock_tokenizer()

            def _create_mock_tokenizer(self):
                class MockTokenizer:
                    def __call__(self, text, return_tensors='pt', **kwargs):
                        # Mock tokenization
                        return {'input_ids': torch.randint(0, 49408, (1, 77))}

                return MockTokenizer()

            def __call__(self, text=None, images=None, return_tensors='pt', **kwargs):
                result = {}

                if images is not None:
                    # Mock image processing - handle both single image and list of images
                    if isinstance(images, list):
                        # Process multiple images
                        batch_size = len(images)
                        result['pixel_values'] = torch.randn(batch_size, 3, 224, 224)
                    else:
                        # Single image
                        result['pixel_values'] = torch.randn(1, 3, 224, 224)

                if text is not None:
                    # Handle both single text and list of texts
                    if isinstance(text, list):
                        result.update(self.tokenizer(text[0], return_tensors=return_tensors, **kwargs))
                        # Expand for batch
                        if 'input_ids' in result:
                            result['input_ids'] = result['input_ids'].repeat(len(text), 1)
                    else:
                        result.update(self.tokenizer(text, return_tensors=return_tensors, **kwargs))

                return result

        return MockProcessor()

    def get_sample_inputs(self) -> List[torch.Tensor]:
        """Get sample inputs for the workload."""
        if self.processor is None:
            self.load_model()

        # Create multimodal inputs
        texts = ["a photo of a cat", "a photo of a dog"] * (self.batch_size // 2)

        # Create mock images instead of None
        mock_images = [torch.randn(3, 224, 224) * 255 for _ in range(len(texts))]
        mock_images = [torch.clamp(img, 0, 255).byte() for img in mock_images]

        inputs = self.processor(text=texts, images=mock_images, return_tensors='pt')

        # CRITICAL FIX: Ensure inputs are on the correct device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        input_ids = inputs['input_ids'].to(device)
        pixel_values = inputs['pixel_values'].to(device)

        return [input_ids, pixel_values]

    def run(self, baseline_config: Dict) -> Dict[str, Any]:
        """
        Run multimodal VQA workload.

        Key metrics:
        - Total inference latency
        - Vision vs text processing time
        - Similarity scores
        """
        if self.model is None:
            self.load_model()

        # CRITICAL FIX: Move model to correct device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        # Get inputs
        inputs = self.get_sample_inputs()
        input_ids = inputs[0]
        pixel_values = inputs[1]

        print(f"VQA input: text={input_ids.shape}, vision={pixel_values.shape}")

        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, pixel_values=pixel_values)

        end_time = time.perf_counter()

        # CRITICAL FIX: Handle CLIP output which returns CLIPOutput namedtuple
        # Extract the tensor from the output
        try:
            if hasattr(outputs, 'logits_per_image'):
                logits_shape = outputs.logits_per_image.shape
            else:
                logits_shape = None
        except:
            logits_shape = None

        return {
            'batch_size': input_ids.shape[0],
            'text_length': input_ids.shape[1],
            'vision_shape': list(pixel_values.shape),
            'similarity_matrix_shape': logits_shape,
            'total_latency_ms': (end_time - start_time) * 1000,
            'throughput_samples_per_sec': input_ids.shape[0] / (end_time - start_time),
            'baseline': baseline_config.get('baseline', 'unknown'),
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Get workload metadata."""
        return {
            'workload': 'multimodal_vqa',
            'model': self.model_name,
            'batch_size': self.batch_size,
            'description': 'Multimodal VQA (CLIP-based)',
            'key_optimization': 'parallel_branches',
            'expected_speedup': '1.5x vs no semantics',
            'parallel_branches': ['vision_encoder', 'text_encoder']
        }
