# Root Cause Analysis: exp5_1 Coordinator Initialization Issues

## Problem Summary

exp5_1 fails with error:
```
RuntimeError: Djinn coordinator unavailable after initialization. 
Initialized: False, Coordinator set: False, Server: localhost:5556
```

## Root Causes Identified

### 1. **Race Condition in Initialization State Management**

**Location**: `Evaluation/common/experiment_runner.py:178-203`

**Issue**: When `ensure_initialized()` is called synchronously before `asyncio.run()`, it initializes Djinn in one event loop. When we enter a new event loop with `asyncio.run()`, the code detects `initialized=True` but `coordinator=None` (possibly due to event loop cleanup or state inconsistency), then tries to re-initialize.

**Problem Flow**:
1. `ensure_initialized()` called synchronously → creates event loop A, initializes coordinator
2. `asyncio.run()` called → creates new event loop B
3. Code checks `_runtime_state.initialized` → True, but `coordinator` might be None or inaccessible
4. Code sets `_runtime_state.initialized = False` and calls `_genie_init_async_impl()`
5. `_genie_init_async_impl()` checks `if _runtime_state.initialized:` at line 250 → might see True if there's a race condition
6. Returns early without actually initializing

**Evidence**:
- Error message shows "Initialization marked complete but coordinator missing, re-initializing..."
- Then shows "Initialized: False, Coordinator set: False"
- This indicates re-initialization failed or was skipped

### 2. **Direct Call to `_genie_init_async_impl()` Bypasses Proper Initialization Flow**

**Location**: `Evaluation/common/experiment_runner.py:193, 200`

**Issue**: The code calls `_genie_init_async_impl()` directly instead of using `init_async()`. While `_genie_init_async_impl()` is the implementation, `init_async()` has additional logic to:
- Check if initialization is already in progress
- Wait for existing initialization tasks
- Properly handle concurrent initialization attempts

**Problem**: By calling `_genie_init_async_impl()` directly, we bypass the safety checks in `init_async()` that handle concurrent initialization.

### 3. **Initialization Lock Race Condition**

**Location**: `djinn/backend/runtime/initialization.py:249-251`

**Issue**: The check `if _runtime_state.initialized:` happens inside a lock, but we're setting `_runtime_state.initialized = False` OUTSIDE the lock. This creates a window where:
- Thread A sets `initialized = False`
- Thread B checks `initialized` (sees False) and starts initialization
- Thread A calls `_genie_init_async_impl()` which checks `initialized` (might see True if Thread B completed)

### 4. **Event Loop Isolation**

**Location**: `djinn/backend/runtime/initialization.py:145-146`

**Issue**: When `djinn.init()` is called synchronously, it creates a new event loop:
```python
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
```

When `asyncio.run()` is called later, it creates a NEW event loop, potentially causing:
- Coordinator tasks from the old loop to be orphaned
- State inconsistencies between loops
- Coordinator becoming inaccessible

## Solutions

### Solution 1: Use `init_async()` Instead of Direct Call (RECOMMENDED)

**Change**: In `experiment_runner.py`, replace direct calls to `_genie_init_async_impl()` with `init_async()`:

```python
# Instead of:
result = await _genie_init_async_impl(server_address=server_address, ...)

# Use:
from djinn.backend.runtime.initialization import init_async
result = await init_async(server_address=server_address, ...)
```

**Benefits**:
- Proper handling of concurrent initialization
- Waits for in-progress initialization tasks
- Better error handling

### Solution 2: Fix Initialization State Reset Logic

**Change**: Reset `initialized` flag INSIDE the initialization lock:

```python
# In experiment_runner.py
from djinn.backend.runtime.initialization import _runtime_state, init_async

if _runtime_state.initialized and _runtime_state.coordinator is None:
    # Reset inside lock to prevent race conditions
    with _runtime_state.initialization_lock:
        if _runtime_state.initialized and _runtime_state.coordinator is None:
            _runtime_state.initialized = False
            _runtime_state.initialization_task = None  # Clear any stale task
```

### Solution 3: Ensure Coordinator Persistence Across Event Loops

**Change**: The coordinator is stored in `_runtime_state` which is a global singleton, so it SHOULD persist. However, we should verify that:
1. Coordinator doesn't have event-loop-specific state
2. Coordinator tasks are properly cleaned up when event loops change
3. Coordinator can be accessed from any event loop

### Solution 4: Simplify Initialization Check

**Change**: Instead of complex re-initialization logic, just check if coordinator exists and is accessible:

```python
async def _ensure_manager(self):
    if self._manager is not None:
        return self._manager
    
    from djinn.core.enhanced_model_manager import EnhancedModelManager
    from djinn.backend.runtime.initialization import get_coordinator, init_async
    
    # Simple: just ensure initialization happened
    coordinator = get_coordinator()
    if coordinator is None:
        # Initialize if not done
        server_address = self.server_address or "localhost:5556"
        result = await init_async(server_address=server_address, auto_connect=True, profiling=False)
        if result.get("status") != "success":
            raise RuntimeError(f"Initialization failed: {result.get('error')}")
        coordinator = get_coordinator()
    
    if coordinator is None:
        raise RuntimeError("Djinn coordinator unavailable after initialization")
    
    self._manager = EnhancedModelManager(coordinator=coordinator, server_address=self.server_address)
    return self._manager
```

## Recommended Fix

**Priority**: HIGH - This blocks exp5_1 from working

**Action**: Implement Solution 1 + Solution 4 (simplify the logic and use `init_async()`)

**Testing**: After fix, verify:
1. `ensure_initialized()` called synchronously before `asyncio.run()`
2. Coordinator accessible in new event loop
3. Re-initialization works if coordinator is None
4. No race conditions or deadlocks

