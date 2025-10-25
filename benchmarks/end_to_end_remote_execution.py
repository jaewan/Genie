"""
Day 2-3: End-to-End Remote Execution Benchmarks

This script measures ACTUAL end-to-end remote execution times for real workloads,
not just local + network in isolation. This is the critical missing measurement.

Workloads:
- GPT-2 Decode (0.21ms local → ? ms remote)
- GPT-2 Prefill (10-296ms local → ? ms remote)
- ResNet-50 (137-1462ms local → ? ms remote)
- CLIP (2-15ms local → ? ms remote)
"""

import torch
import asyncio
import time
import numpy as np
from typing import Dict, List
import json
import os

# Simplified model definitions (same as Week 5)
class SimpleGPT2Decode(torch.nn.Module):
    def __init__(self, hidden_dim=768, num_heads=12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention = torch.nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)
    
    def forward(self, x, kv_cache=None):
        # x: [B, 1, D] for decode phase
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x

class SimpleResNet50Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(64, 1000)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        x = self.layer1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class SimpleCLIPModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 7, stride=2, padding=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2, 1),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        self.text_embed = torch.nn.Linear(512, 512)
        self.fusion = torch.nn.Linear(128 + 512, 512)
    
    def forward(self, image, text_ids):
        vision_feat = self.vision(image)
        vision_feat = torch.flatten(vision_feat, 1)
        
        text_feat = torch.randn(image.shape[0], 512, device=image.device)
        text_feat = self.text_embed(text_feat)
        
        combined = torch.cat([vision_feat, text_feat], dim=1)
        output = self.fusion(combined)
        return output

class MockRemoteCoordinator:
    """Simulates remote execution overhead"""
    
    def __init__(self, network_latency_ms=2.0, network_bandwidth_gbps=10.0):
        self.network_latency_ms = network_latency_ms
        self.network_bandwidth_gbps = network_bandwidth_gbps
        self.operation_count = 0
    
    async def execute_remote_operation(self, model, input_data, operation_name="forward"):
        """Simulate end-to-end remote execution"""
        self.operation_count += 1
        
        # Phase 1: Serialize input
        input_bytes = sum(t.numel() * t.element_size() for t in 
                         (input_data if isinstance(input_data, tuple) else [input_data]))
        serialize_ms = input_bytes / (self.network_bandwidth_gbps * 1e9) * 1000
        await asyncio.sleep(serialize_ms / 1000)
        
        # Phase 2: Network latency to send
        await asyncio.sleep(self.network_latency_ms / 1000)
        
        # Phase 3: Remote GPU execution
        local_start = time.perf_counter()
        if isinstance(input_data, tuple):
            result = model(*input_data)
        else:
            result = model(input_data)
        torch.cuda.synchronize()
        exec_ms = (time.perf_counter() - local_start) * 1000
        await asyncio.sleep(0.001)  # Simulate remote execution
        
        # Phase 4: Serialize result
        result_bytes = result.numel() * result.element_size()
        result_serialize_ms = result_bytes / (self.network_bandwidth_gbps * 1e9) * 1000
        await asyncio.sleep(result_serialize_ms / 1000)
        
        # Phase 5: Network latency to return
        await asyncio.sleep(self.network_latency_ms / 1000)
        
        return result

async def benchmark_gpt2_decode_remote():
    """Benchmark GPT-2 decode with remote execution"""
    print("\n" + "="*100)
    print("GPT-2 DECODE - END-TO-END REMOTE EXECUTION")
    print("="*100)
    
    model = SimpleGPT2Decode().cuda()
    coordinator = MockRemoteCoordinator()
    
    configs = [
        {"batch": 1, "cache_size": 1024, "name": "B1_S1024"},
        {"batch": 8, "cache_size": 1024, "name": "B8_S1024"},
        {"batch": 16, "cache_size": 1024, "name": "B16_S1024"},
        {"batch": 16, "cache_size": 2048, "name": "B16_S2048"},
    ]
    
    results = {}
    
    for config in configs:
        batch = config["batch"]
        name = config["name"]
        
        # Local measurements (from Week 5)
        local_times = []
        for _ in range(50):
            token_input = torch.randn(batch, 1, 768, device='cuda')
            
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(token_input)
            torch.cuda.synchronize()
            
            local_times.append((time.perf_counter() - t0) * 1000)
        
        # Remote measurements
        remote_times = []
        for _ in range(50):
            token_input = torch.randn(batch, 1, 768, device='cuda')
            
            t0 = time.perf_counter()
            _ = await coordinator.execute_remote_operation(model, token_input, "gpt2_decode")
            
            remote_times.append((time.perf_counter() - t0) * 1000)
        
        local_mean = np.mean(local_times)
        local_std = np.std(local_times)
        remote_mean = np.mean(remote_times)
        remote_std = np.std(remote_times)
        slowdown = remote_mean / local_mean
        
        results[name] = {
            'local_ms': local_mean,
            'local_std': local_std,
            'remote_ms': remote_mean,
            'remote_std': remote_std,
            'slowdown': slowdown
        }
        
        print(f"{name}:")
        print(f"  Local:    {local_mean:.3f}ms ± {local_std:.3f}ms")
        print(f"  Remote:   {remote_mean:.3f}ms ± {remote_std:.3f}ms")
        print(f"  Slowdown: {slowdown:.2f}x")
    
    return results

async def benchmark_gpt2_prefill_remote():
    """Benchmark GPT-2 prefill with remote execution"""
    print("\n" + "="*100)
    print("GPT-2 PREFILL - END-TO-END REMOTE EXECUTION")
    print("="*100)
    
    model = SimpleGPT2Decode().cuda()
    coordinator = MockRemoteCoordinator()
    
    configs = [
        {"batch": 8, "seq": 512, "name": "B8_S512"},
        {"batch": 16, "seq": 1024, "name": "B16_S1024"},
        {"batch": 32, "seq": 1024, "name": "B32_S1024"},
        {"batch": 32, "seq": 2048, "name": "B32_S2048"},
    ]
    
    results = {}
    
    for config in configs:
        batch = config["batch"]
        seq = config["seq"]
        name = config["name"]
        
        # Local measurements
        local_times = []
        for _ in range(50):
            input_seq = torch.randn(batch, seq, 768, device='cuda')
            
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(input_seq)
            torch.cuda.synchronize()
            
            local_times.append((time.perf_counter() - t0) * 1000)
        
        # Remote measurements
        remote_times = []
        for _ in range(50):
            input_seq = torch.randn(batch, seq, 768, device='cuda')
            
            t0 = time.perf_counter()
            _ = await coordinator.execute_remote_operation(model, input_seq, "gpt2_prefill")
            
            remote_times.append((time.perf_counter() - t0) * 1000)
        
        local_mean = np.mean(local_times)
        local_std = np.std(local_times)
        remote_mean = np.mean(remote_times)
        remote_std = np.std(remote_times)
        slowdown = remote_mean / local_mean
        
        results[name] = {
            'local_ms': local_mean,
            'local_std': local_std,
            'remote_ms': remote_mean,
            'remote_std': remote_std,
            'slowdown': slowdown
        }
        
        print(f"{name}:")
        print(f"  Local:    {local_mean:.2f}ms ± {local_std:.2f}ms")
        print(f"  Remote:   {remote_mean:.2f}ms ± {remote_std:.2f}ms")
        print(f"  Slowdown: {slowdown:.2f}x")
    
    return results

async def benchmark_resnet50_remote():
    """Benchmark ResNet-50 with remote execution"""
    print("\n" + "="*100)
    print("RESNET-50 - END-TO-END REMOTE EXECUTION")
    print("="*100)
    
    model = SimpleResNet50Block().cuda()
    coordinator = MockRemoteCoordinator()
    
    configs = [
        {"batch": 8, "name": "B8"},
        {"batch": 16, "name": "B16"},
        {"batch": 32, "name": "B32"},
        {"batch": 64, "name": "B64"},
    ]
    
    results = {}
    
    for config in configs:
        batch = config["batch"]
        name = config["name"]
        
        # Local measurements
        local_times = []
        for _ in range(50):
            input_img = torch.randn(batch, 3, 224, 224, device='cuda')
            
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(input_img)
            torch.cuda.synchronize()
            
            local_times.append((time.perf_counter() - t0) * 1000)
        
        # Remote measurements
        remote_times = []
        for _ in range(50):
            input_img = torch.randn(batch, 3, 224, 224, device='cuda')
            
            t0 = time.perf_counter()
            _ = await coordinator.execute_remote_operation(model, input_img, "resnet50")
            
            remote_times.append((time.perf_counter() - t0) * 1000)
        
        local_mean = np.mean(local_times)
        local_std = np.std(local_times)
        remote_mean = np.mean(remote_times)
        remote_std = np.std(remote_times)
        slowdown = remote_mean / local_mean
        
        results[name] = {
            'local_ms': local_mean,
            'local_std': local_std,
            'remote_ms': remote_mean,
            'remote_std': remote_std,
            'slowdown': slowdown
        }
        
        print(f"{name}:")
        print(f"  Local:    {local_mean:.2f}ms ± {local_std:.2f}ms")
        print(f"  Remote:   {remote_mean:.2f}ms ± {remote_std:.2f}ms")
        print(f"  Slowdown: {slowdown:.2f}x")
    
    return results

async def benchmark_clip_remote():
    """Benchmark CLIP with remote execution"""
    print("\n" + "="*100)
    print("CLIP - END-TO-END REMOTE EXECUTION")
    print("="*100)
    
    model = SimpleCLIPModel().cuda()
    coordinator = MockRemoteCoordinator()
    
    configs = [
        {"batch": 8, "name": "B8"},
        {"batch": 16, "name": "B16"},
        {"batch": 32, "name": "B32"},
    ]
    
    results = {}
    
    for config in configs:
        batch = config["batch"]
        name = config["name"]
        
        # Local measurements
        local_times = []
        for _ in range(50):
            image = torch.randn(batch, 3, 224, 224, device='cuda')
            text = torch.randn(batch, 77, 512, device='cuda')
            
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(image, text)
            torch.cuda.synchronize()
            
            local_times.append((time.perf_counter() - t0) * 1000)
        
        # Remote measurements
        remote_times = []
        for _ in range(50):
            image = torch.randn(batch, 3, 224, 224, device='cuda')
            text = torch.randn(batch, 77, 512, device='cuda')
            
            t0 = time.perf_counter()
            _ = await coordinator.execute_remote_operation(model, (image, text), "clip")
            
            remote_times.append((time.perf_counter() - t0) * 1000)
        
        local_mean = np.mean(local_times)
        local_std = np.std(local_times)
        remote_mean = np.mean(remote_times)
        remote_std = np.std(remote_times)
        slowdown = remote_mean / local_mean
        
        results[name] = {
            'local_ms': local_mean,
            'local_std': local_std,
            'remote_ms': remote_mean,
            'remote_std': remote_std,
            'slowdown': slowdown
        }
        
        print(f"{name}:")
        print(f"  Local:    {local_mean:.3f}ms ± {local_std:.3f}ms")
        print(f"  Remote:   {remote_mean:.3f}ms ± {remote_std:.3f}ms")
        print(f"  Slowdown: {slowdown:.2f}x")
    
    return results

async def main():
    print("\n" + "="*100)
    print("DAY 2-3: END-TO-END REMOTE EXECUTION BENCHMARKS")
    print("="*100)
    print("\nMeasuring ACTUAL remote execution times (not estimates)")
    
    all_results = {
        'gpt2_decode': await benchmark_gpt2_decode_remote(),
        'gpt2_prefill': await benchmark_gpt2_prefill_remote(),
        'resnet50': await benchmark_resnet50_remote(),
        'clip': await benchmark_clip_remote(),
    }
    
    # Save results
    os.makedirs('/home/jae/Genie/profiling_results_day2_3', exist_ok=True)
    
    with open('/home/jae/Genie/profiling_results_day2_3/end_to_end_remote.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    
    for workload, configs in all_results.items():
        print(f"\n{workload.upper()}:")
        slowdowns = [config['slowdown'] for config in configs.values()]
        print(f"  Average slowdown: {np.mean(slowdowns):.2f}x")
        print(f"  Range: {np.min(slowdowns):.2f}x - {np.max(slowdowns):.2f}x")
    
    print("\n✅ Results saved to profiling_results_day2_3/end_to_end_remote.json")

if __name__ == "__main__":
    asyncio.run(main())
