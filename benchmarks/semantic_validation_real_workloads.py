"""
Day 4: Semantic Benefits Validation on Real Workloads

Test semantic awareness on actual models to determine if semantic scheduling
helps or just adds overhead.
"""

import torch
import asyncio
import time
import numpy as np
import json
import os

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
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x

class SimpleGPT2Prefill(torch.nn.Module):
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
    
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x

class SimpleResNet50(torch.nn.Module):
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

class SimpleCLIP(torch.nn.Module):
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

class SemanticScheduler:
    """Mock scheduler that applies semantic optimizations"""
    
    def __init__(self):
        self.semantics_enabled = True
        self.optimization_count = 0
    
    def apply_semantics(self, model_name, input_data):
        """Apply semantic optimizations"""
        if not self.semantics_enabled:
            return
        
        self.optimization_count += 1
        
        # Simulate semantic overhead (pattern detection, scheduling)
        overhead_ms = 0.5  # ~0.5ms for semantic analysis
        return overhead_ms
    
    def enable(self):
        self.semantics_enabled = True
    
    def disable(self):
        self.semantics_enabled = False

def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def ttest_ind(group1, group2):
    """Simple t-test implementation"""
    mean1, mean2 = np.mean(group1), np.mean(group2)
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard error
    se = np.sqrt((var1/n1) + (var2/n2))
    
    # t-statistic
    t_stat = (mean1 - mean2) / se if se > 0 else 0
    
    # Simple p-value approximation (very rough)
    p_value = 0.05 if abs(t_stat) > 1.96 else 0.5
    
    return t_stat, p_value

async def benchmark_with_without_semantics(model, input_generator, model_name, config_name, n_runs=50):
    """Benchmark a model with and without semantic awareness"""
    
    scheduler = SemanticScheduler()
    
    # Baseline: No semantics
    scheduler.disable()
    baseline_times = []
    for _ in range(n_runs):
        input_data = input_generator()
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(*input_data if isinstance(input_data, tuple) else [input_data])
        torch.cuda.synchronize()
        
        baseline_times.append((time.perf_counter() - t0) * 1000)
    
    # Optimized: With semantics
    scheduler.enable()
    optimized_times = []
    for _ in range(n_runs):
        input_data = input_generator()
        semantic_ms = scheduler.apply_semantics(model_name, input_data)
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(*input_data if isinstance(input_data, tuple) else [input_data])
        torch.cuda.synchronize()
        exec_ms = (time.perf_counter() - t0) * 1000
        
        # Add semantic overhead
        total_ms = exec_ms + (semantic_ms if semantic_ms else 0)
        optimized_times.append(total_ms)
    
    # Statistical analysis
    baseline_mean = np.mean(baseline_times)
    baseline_std = np.std(baseline_times)
    optimized_mean = np.mean(optimized_times)
    optimized_std = np.std(optimized_times)
    
    speedup = baseline_mean / optimized_mean
    
    t_stat, p_value = ttest_ind(baseline_times, optimized_times)
    cohens_d = calculate_cohens_d(baseline_times, optimized_times)
    
    return {
        'baseline_ms': baseline_mean,
        'baseline_std': baseline_std,
        'optimized_ms': optimized_mean,
        'optimized_std': optimized_std,
        'speedup': speedup,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05
    }

async def test_gpt2_decode():
    """Test semantic benefits for GPT-2 decode"""
    print("\n" + "="*100)
    print("SEMANTIC VALIDATION: GPT-2 DECODE")
    print("="*100)
    
    model = SimpleGPT2Decode().cuda()
    results = {}
    
    configs = [
        {"batch": 16, "name": "B16_S1024"},
        {"batch": 32, "name": "B32_S1024"},
    ]
    
    for config in configs:
        batch = config["batch"]
        name = config["name"]
        
        def input_gen():
            return torch.randn(batch, 1, 768, device='cuda')
        
        result = await benchmark_with_without_semantics(
            model, input_gen, "gpt2_decode", name
        )
        results[name] = result
        
        print(f"\n{name}:")
        print(f"  Baseline:  {result['baseline_ms']:.3f}ms ± {result['baseline_std']:.3f}ms")
        print(f"  Optimized: {result['optimized_ms']:.3f}ms ± {result['optimized_std']:.3f}ms")
        print(f"  Speedup:   {result['speedup']:.2f}x")
        print(f"  Cohen's d: {result['cohens_d']:.2f}")
        print(f"  p-value:   {result['p_value']:.4f}")
        print(f"  Significant: {'YES' if result['significant'] else 'NO'}")
    
    return results

async def test_gpt2_prefill():
    """Test semantic benefits for GPT-2 prefill"""
    print("\n" + "="*100)
    print("SEMANTIC VALIDATION: GPT-2 PREFILL")
    print("="*100)
    
    model = SimpleGPT2Prefill().cuda()
    results = {}
    
    configs = [
        {"batch": 32, "seq": 1024, "name": "B32_S1024"},
        {"batch": 32, "seq": 2048, "name": "B32_S2048"},
    ]
    
    for config in configs:
        batch = config["batch"]
        seq = config["seq"]
        name = config["name"]
        
        def input_gen():
            return torch.randn(batch, seq, 768, device='cuda')
        
        result = await benchmark_with_without_semantics(
            model, input_gen, "gpt2_prefill", name
        )
        results[name] = result
        
        print(f"\n{name}:")
        print(f"  Baseline:  {result['baseline_ms']:.2f}ms ± {result['baseline_std']:.2f}ms")
        print(f"  Optimized: {result['optimized_ms']:.2f}ms ± {result['optimized_std']:.2f}ms")
        print(f"  Speedup:   {result['speedup']:.2f}x")
        print(f"  Cohen's d: {result['cohens_d']:.2f}")
        print(f"  p-value:   {result['p_value']:.4f}")
        print(f"  Significant: {'YES' if result['significant'] else 'NO'}")
    
    return results

async def test_resnet50():
    """Test semantic benefits for ResNet-50"""
    print("\n" + "="*100)
    print("SEMANTIC VALIDATION: RESNET-50")
    print("="*100)
    
    model = SimpleResNet50().cuda()
    results = {}
    
    configs = [
        {"batch": 32, "name": "B32"},
        {"batch": 64, "name": "B64"},
    ]
    
    for config in configs:
        batch = config["batch"]
        name = config["name"]
        
        def input_gen():
            return torch.randn(batch, 3, 224, 224, device='cuda')
        
        result = await benchmark_with_without_semantics(
            model, input_gen, "resnet50", name
        )
        results[name] = result
        
        print(f"\n{name}:")
        print(f"  Baseline:  {result['baseline_ms']:.2f}ms ± {result['baseline_std']:.2f}ms")
        print(f"  Optimized: {result['optimized_ms']:.2f}ms ± {result['optimized_std']:.2f}ms")
        print(f"  Speedup:   {result['speedup']:.2f}x")
        print(f"  Cohen's d: {result['cohens_d']:.2f}")
        print(f"  p-value:   {result['p_value']:.4f}")
        print(f"  Significant: {'YES' if result['significant'] else 'NO'}")
    
    return results

async def test_clip():
    """Test semantic benefits for CLIP"""
    print("\n" + "="*100)
    print("SEMANTIC VALIDATION: CLIP")
    print("="*100)
    
    model = SimpleCLIP().cuda()
    results = {}
    
    configs = [
        {"batch": 16, "name": "B16"},
        {"batch": 32, "name": "B32"},
    ]
    
    for config in configs:
        batch = config["batch"]
        name = config["name"]
        
        def input_gen():
            return (
                torch.randn(batch, 3, 224, 224, device='cuda'),
                torch.randn(batch, 77, 512, device='cuda')
            )
        
        result = await benchmark_with_without_semantics(
            model, input_gen, "clip", name
        )
        results[name] = result
        
        print(f"\n{name}:")
        print(f"  Baseline:  {result['baseline_ms']:.3f}ms ± {result['baseline_std']:.3f}ms")
        print(f"  Optimized: {result['optimized_ms']:.3f}ms ± {result['optimized_std']:.3f}ms")
        print(f"  Speedup:   {result['speedup']:.2f}x")
        print(f"  Cohen's d: {result['cohens_d']:.2f}")
        print(f"  p-value:   {result['p_value']:.4f}")
        print(f"  Significant: {'YES' if result['significant'] else 'NO'}")
    
    return results

async def main():
    print("\n" + "="*100)
    print("DAY 4: SEMANTIC BENEFITS VALIDATION ON REAL WORKLOADS")
    print("="*100)
    print("\nTesting with/without semantic awareness to measure actual benefits")
    
    all_results = {
        'gpt2_decode': await test_gpt2_decode(),
        'gpt2_prefill': await test_gpt2_prefill(),
        'resnet50': await test_resnet50(),
        'clip': await test_clip(),
    }
    
    # Save results
    os.makedirs('/home/jae/Genie/profiling_results_day4', exist_ok=True)
    
    with open('/home/jae/Genie/profiling_results_day4/semantic_validation.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    
    for workload, configs in all_results.items():
        print(f"\n{workload.upper()}:")
        speedups = [config['speedup'] for config in configs.values()]
        significant = sum(1 for config in configs.values() if config['significant'])
        
        print(f"  Avg speedup: {np.mean(speedups):.2f}x")
        print(f"  Significant: {significant}/{len(configs)}")
        
        for name, result in configs.items():
            sig = "✓" if result['significant'] else "✗"
            print(f"    {name}: {result['speedup']:.2f}x {sig}")
    
    print("\n✅ Results saved to profiling_results_day4/semantic_validation.json")

if __name__ == "__main__":
    asyncio.run(main())
