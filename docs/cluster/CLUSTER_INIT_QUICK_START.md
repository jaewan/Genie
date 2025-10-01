# Cluster Initialization - Quick Start for Developers

**For**: Junior developers implementing the cluster initialization feature  
**Prerequisites**: Read `CLUSTER_INIT_SUMMARY.md` first

---

## 🚀 Getting Started

### Step 1: Setup Development Environment

```bash
# Clone repo
cd /home/jaewan/Genie

# Install in editable mode
pip install -e .

# Install test dependencies
pip install pytest pytest-asyncio pytest-cov pytest-timeout

# Verify installation
python -c "import genie; print('OK')"
```

### Step 2: Read Documentation

**Must read** (in order):
1. `CLUSTER_INIT_SUMMARY.md` - Overview and architecture
2. `CLUSTER_INIT_IMPLEMENTATION_PLAN.md` - Detailed Phase 1 & 2
3. `CLUSTER_INIT_IMPLEMENTATION_PLAN_PART2.md` - Detailed Phase 3 & 4

**Background reading**:
- `.kiro/HotNets25.tex` - Research paper
- `docs/implementation/05-runtime-transport.md` - Existing transport layer

### Step 3: Start with Phase 1, Task 1.1

**File**: `genie/cluster/node_info.py`

Follow the step-by-step instructions in the implementation plan.

---

## 📝 Development Workflow

### For Each Task

1. **Create the file** (if new)
   ```bash
   mkdir -p genie/cluster
   touch genie/cluster/node_info.py
   ```

2. **Copy the code** from the implementation plan
   - Use the exact code provided
   - Add your name to docstring if you modify

3. **Write tests**
   ```bash
   touch tests/test_cluster_node_info.py
   ```
   - Copy test code from plan
   - Run: `pytest tests/test_cluster_node_info.py -v`

4. **Commit your work**
   ```bash
   git add genie/cluster/node_info.py tests/test_cluster_node_info.py
   git commit -m "Implement Task 1.1: Cluster node info structures"
   ```

5. **Move to next task** ✓

---

## 🧪 Testing Commands

### Run Specific Test File
```bash
pytest tests/test_cluster_node_info.py -v
```

### Run All Cluster Tests
```bash
pytest tests/test_cluster*.py -v
```

### Run with Coverage
```bash
pytest tests/ -v --cov=genie.cluster --cov-report=html
open htmlcov/index.html  # View coverage report
```

### Run Integration Tests (requires server)
```bash
export GENIE_TEST_SERVER='localhost'
pytest tests/integration/ -v -m integration
```

---

## 📁 File Structure

```
Genie/
├── genie/
│   ├── cluster/                    # NEW: Cluster management
│   │   ├── __init__.py            # Task 1.1
│   │   ├── node_info.py           # Task 1.1
│   │   ├── init.py                # Task 1.2
│   │   ├── backend_info.py        # Task 2.2
│   │   ├── events.py              # Task 3.2
│   │   ├── monitoring.py          # Task 3.2
│   │   ├── health.py              # Task 3.3
│   │   └── dashboard.py           # Task 3.4
│   └── runtime/
│       └── network_discovery.py   # NEW: Task 2.1
│
├── tests/
│   ├── test_cluster_node_info.py  # Task 1.1
│   ├── test_cluster_init.py       # Task 1.2
│   ├── test_network_discovery.py  # Task 2.1
│   ├── test_resource_monitor.py   # Task 3.2
│   └── integration/               # Task 4.1
│       ├── test_full_discovery.py
│       └── test_end_to_end.py
│
├── examples/                       # NEW: Task 4.3
│   ├── basic_client.py
│   ├── gpu_server.py
│   └── monitoring_dashboard.py
│
├── benchmarks/                     # NEW: Task 4.4
│   └── bench_init.py
│
└── docs/
    ├── CLUSTER_INIT_SUMMARY.md              # Overview
    ├── CLUSTER_INIT_IMPLEMENTATION_PLAN.md  # Part 1
    ├── CLUSTER_INIT_IMPLEMENTATION_PLAN_PART2.md  # Part 2
    ├── CLUSTER_INIT_QUICK_START.md         # This file
    ├── USER_GUIDE.md                        # Task 4.2
    └── ENVIRONMENT_VARIABLES.md             # Task 1.3
```

---

## 🐛 Debugging Tips

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'genie.cluster'`

**Solution**:
```bash
# Make sure __init__.py exists
touch genie/cluster/__init__.py

# Reinstall in editable mode
pip install -e .
```

### Test Failures

**Problem**: Tests failing with connection errors

**Solution**:
```python
# Check if you need a test server
import os
if not os.getenv('GENIE_TEST_SERVER'):
    pytest.skip("Set GENIE_TEST_SERVER to run this test")
```

### Async Errors

**Problem**: `RuntimeError: Event loop is closed`

**Solution**:
```python
# Use pytest-asyncio
@pytest.mark.asyncio
async def test_something():
    await my_async_function()
```

### Type Errors

**Problem**: MyPy complaining about types

**Solution**:
```python
from typing import Optional, Dict, Any

def my_func(x: Optional[str] = None) -> Dict[str, Any]:
    return {}
```

---

## ✅ Task Checklist

### Phase 1: Core Infrastructure
- [ ] Task 1.1: Node info structures
  - [ ] Create `genie/cluster/__init__.py`
  - [ ] Create `genie/cluster/node_info.py`
  - [ ] Write `tests/test_cluster_node_info.py`
  - [ ] All tests passing
  
- [ ] Task 1.2: Basic init API
  - [ ] Create `genie/cluster/init.py`
  - [ ] Write `tests/test_cluster_init.py`
  - [ ] All tests passing
  
- [ ] Task 1.3: Environment variables
  - [ ] Create `docs/ENVIRONMENT_VARIABLES.md`
  - [ ] Add env var tests
  
- [ ] Task 1.4: Unit tests
  - [ ] All Phase 1 tests passing
  - [ ] Coverage >80%

### Phase 2: Network Discovery
- [ ] Task 2.1: Discovery service
  - [ ] Create `genie/runtime/network_discovery.py`
  - [ ] Write `tests/test_network_discovery.py`
  - [ ] All tests passing
  
- [ ] Task 2.2: Backend selection
  - [ ] Create `genie/cluster/backend_info.py`
  - [ ] Add selection tests
  
- [ ] Task 2.3: Transport integration
  - [ ] Verify init.py integration
  - [ ] Test with real transport
  
- [ ] Task 2.4: Discovery tests
  - [ ] Integration tests
  - [ ] Coverage >80%

### Phase 3: Resource Monitoring
- [ ] Task 3.1: Enhanced NodeInfo ✓ (done in Task 1.1)
  
- [ ] Task 3.2: GPU monitoring
  - [ ] Create `genie/cluster/monitoring.py`
  - [ ] Create `genie/cluster/events.py`
  - [ ] Write `tests/test_resource_monitor.py`
  
- [ ] Task 3.3: Health checks
  - [ ] Create `genie/cluster/health.py`
  - [ ] Add health check tests
  
- [ ] Task 3.4: Dashboard
  - [ ] Create `genie/cluster/dashboard.py`
  - [ ] Test dashboard manually

### Phase 4: Integration & Docs
- [ ] Task 4.1: Integration tests
  - [ ] Create `tests/integration/test_end_to_end.py`
  - [ ] Create `tests/integration/test_multi_node.py`
  - [ ] All integration tests passing
  
- [ ] Task 4.2: Documentation
  - [ ] Create `docs/USER_GUIDE.md`
  - [ ] Update `docs/implementation/README.md`
  - [ ] Update `docs/implementation/01-architecture-overview.md`
  
- [ ] Task 4.3: Example scripts
  - [ ] Create `examples/basic_client.py`
  - [ ] Create `examples/gpu_server.py`
  - [ ] Create `examples/monitoring_dashboard.py`
  - [ ] Test all examples
  
- [ ] Task 4.4: Benchmarks
  - [ ] Create `benchmarks/bench_init.py`
  - [ ] Run and document results

---

## 🎯 Daily Workflow

### Morning (9am - 12pm)
1. Pick next task from checklist
2. Read task details in implementation plan
3. Create files and copy code
4. Write tests
5. Make tests pass

### Afternoon (1pm - 5pm)
1. Refine implementation
2. Add edge case tests
3. Update documentation
4. Commit and push
5. Move to next task

### Before End of Day
1. Run all tests: `pytest tests/ -v`
2. Check coverage: `pytest --cov=genie.cluster`
3. Commit work: `git commit -am "Progress on Task X.Y"`
4. Update checklist above
5. Document any blockers

---

## 💬 Getting Help

### Stuck on a Task?

1. **Re-read the implementation plan** - answer is probably there
2. **Check existing code** - look at similar functionality
3. **Run the tests** - they show expected behavior
4. **Ask a question** - in Slack or daily standup

### Questions to Ask

❌ Bad: "How do I do Task 1.1?"
✅ Good: "I'm implementing NodeInfo in Task 1.1. The test expects is_healthy() to return False after 60s, but my implementation returns True. What am I missing?"

❌ Bad: "Tests are failing"
✅ Good: "test_node_health_check in test_cluster_node_info.py fails with AssertionError on line 45. I've checked that last_heartbeat is set correctly. Could it be a timing issue?"

### Resources

- **Slack**: #genie-dev
- **Email**: genie-team@example.com
- **Office Hours**: Wed 2-4pm
- **Code Review**: Create PR after each phase

---

## 🎓 Learning Resources

### Python Async/Await
- https://realpython.com/async-io-python/
- Focus on: `async def`, `await`, `asyncio.create_task()`

### PyTorch Distributed
- https://pytorch.org/tutorials/beginner/dist_overview.html
- See how they do `init_process_group()`

### Testing with Pytest
- https://docs.pytest.org/en/stable/
- Focus on: fixtures, markers, parametrize

### Type Hints
- https://docs.python.org/3/library/typing.html
- Focus on: Optional, Dict, List, Any

---

## 🏁 Done with Implementation?

### Final Steps

1. **Run full test suite**
   ```bash
   pytest tests/ -v --cov=genie.cluster --cov-report=html
   ```

2. **Check all tasks completed** (see checklist above)

3. **Write summary**
   - What you implemented
   - Test coverage achieved
   - Any deviations from plan
   - Known issues

4. **Create pull request**
   ```bash
   git push origin feature/cluster-init
   # Create PR on GitHub
   ```

5. **Request code review** from senior developer

6. **Celebrate!** 🎉

---

## 📚 Appendix: Common Code Patterns

### Creating a New Dataclass

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class MyClass:
    required_field: str
    optional_field: Optional[int] = None
    list_field: List[str] = field(default_factory=list)
```

### Async Function with Error Handling

```python
async def my_async_function():
    try:
        result = await some_async_call()
        return result
    except asyncio.TimeoutError:
        logger.error("Timeout!")
        return None
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
```

### Writing a Test

```python
import pytest

@pytest.mark.asyncio
async def test_my_function():
    """Test my_function does what it should"""
    # Arrange
    input_data = "test"
    
    # Act
    result = await my_function(input_data)
    
    # Assert
    assert result is not None
    assert result == "expected"
```

### Singleton Pattern

```python
class MyClass:
    _instance = None
    
    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
```

---

**Good luck with the implementation! You've got this! 💪**

*Questions? Check the implementation plan or ask in #genie-dev*

