# AI Assistant Guide for Genie Development

## How to Use These Documents Effectively

This guide explains how to leverage the Genie documentation for efficient AI-assisted coding.

## Document Structure Overview

```
📁 documents/
├── 📄 genie-context.md              # Start here - overall project understanding
├── 📄 interface-contracts.md        # Component boundaries and data flow
├── 📁 components/
│   ├── 📄 01-lazytensor-component.md     # LazyTensor implementation details
│   ├── 📄 02-semantic-analyzer-component.md  # Pattern recognition
│   └── 📄 03-zerocopy-runtime-component.md   # Network/memory management
└── 📁 implementation-phases/
    └── 📄 phase1-foundation.md      # Week-by-week implementation guide
```

## Recommended Workflow for AI Assistance

### Step 1: Initial Context (Always Include)
**Always provide `genie-context.md` first** to give the AI assistant the big picture:
```
rm P     [paste genie-context.md]"
```

### Step 2: Component-Specific Work
When working on a specific component, provide:
1. The context document (genie-context.md)
2. The relevant component document
3. The interface contracts (for integration points)

**Example for LazyTensor work:**
```
"I need to implement the LazyTensor materialization logic.
Context: [genie-context.md]
Component spec: [01-lazytensor-component.md]
Interfaces: [relevant sections from interface-contracts.md]"
```

### Step 3: Implementation Tasks
For specific implementation tasks, use the phase guides:
```
"I'm on Week 3 of Phase 1, implementing PyTorch device registration.
Context: [genie-context.md]
Phase guide: [relevant section from phase1-foundation.md]"
```

## Document Usage Patterns

### Pattern 1: New Feature Implementation
```
1. genie-context.md (understanding)
2. Relevant component doc (specifications)
3. interface-contracts.md (integration points)
4. Phase guide (implementation steps)
```

### Pattern 2: Debugging/Fixing Issues
```
1. genie-context.md (system understanding)
2. interface-contracts.md (check data flow)
3. Component doc (verify implementation)
```

### Pattern 3: Testing/Validation
```
1. Component doc (testing requirements)
2. interface-contracts.md (contract validation)
3. Phase guide (success metrics)
```

## Key Information by Document

### genie-context.md
- **Purpose**: Overall system understanding
- **Key sections**: Core Innovation, Architecture Layers, Workload Examples
- **Use when**: Starting any new task, onboarding, understanding requirements

### interface-contracts.md
- **Purpose**: Precise component boundaries
- **Key sections**: Data contracts, Error handling, Type definitions
- **Use when**: Implementing interfaces, debugging integration issues

### Component Documents
- **Purpose**: Deep implementation details
- **Key sections**: Core Implementation, Testing Requirements, Integration Points
- **Use when**: Implementing specific components, writing tests

### Phase Guides
- **Purpose**: Step-by-step implementation
- **Key sections**: Tasks, Success Criteria, Common Issues
- **Use when**: Following implementation timeline, checking progress

## Effective Prompting Examples

### Good Prompt Structure
```
"I need to implement [specific feature].

Project context:
[paste genie-context.md - section on architecture]

Current component:
[paste relevant component section]

Interface requirements:
[paste relevant interface contract]

Specific task:
Implement the shape inference logic for LazyTensor that handles
broadcasting rules correctly for element-wise operations.

Please provide:
1. Implementation with proper error handling
2. Unit tests
3. Integration with existing GraphBuilder"
```

### Prompt for Debugging
```
"I'm getting an error in the LazyTensor materialization.

Error: [paste error]

Context: [paste genie-context.md - LazyTensor section]

Current implementation: [paste your code]

Interface contract: [paste MaterializationTracker contract]

Help me identify and fix the issue."
```

### Prompt for Testing
```
"I need comprehensive tests for the PatternRegistry.

Component spec: [paste pattern recognition section]

Interface: [paste WorkloadProfile contract]

Create tests that:
1. Verify pattern matching accuracy
2. Test fallback behavior
3. Validate performance requirements (<100ms)
4. Check confidence thresholds"
```

## Tips for Efficient AI Assistance

### 1. Minimize Context Window Usage
- Only include relevant sections, not entire documents
- Use section headers to guide what to include
- Remove code examples if not needed for the task

### 2. Be Specific About Output Format
- Request specific file names and paths
- Ask for docstrings and type hints
- Specify testing framework (pytest)

### 3. Iterative Development
- Start with core functionality
- Add error handling in second iteration
- Add optimizations in third iteration

### 4. Request Validation
Always ask the AI to:
- Verify interface contracts
- Check performance requirements
- Validate against success criteria

## Document Section Quick Reference

### For Implementation Tasks
- **What to build**: Component document → "Core Implementation"
- **How to integrate**: interface-contracts.md → relevant component section
- **When to build**: Phase guide → timeline and dependencies
- **How to test**: Component document → "Testing Requirements"

### For Architecture Questions
- **System design**: genie-context.md → "Architecture Layers"
- **Data flow**: interface-contracts.md → "Component Interaction Map"
- **Performance targets**: genie-context.md → "Performance Targets"

### For Debugging
- **Error types**: interface-contracts.md → "Error Handling"
- **Validation**: interface-contracts.md → "Contract Validation"
- **Common issues**: Phase guides → "Common Issues and Solutions"

## Example Development Session

### Session 1: Implement Basic LazyTensor
```
Docs needed:
- genie-context.md (overview)
- 01-lazytensor-component.md (full)
- phase1-foundation.md (Week 5-6 section)
```

### Session 2: Add Pattern Recognition
```
Docs needed:
- genie-context.md (workload examples)
- 02-semantic-analyzer-component.md (pattern section)
- interface-contracts.md (PatternMatch contract)
```

### Session 3: Integrate Components
```
Docs needed:
- interface-contracts.md (full)
- All component docs (integration points sections)
```

## Maintenance Notes

- Keep genie-context.md updated with major changes
- Update interface contracts when adding new data flows
- Add new patterns to component docs as discovered
- Update phase guides with actual timelines and issues encountered

## Quick Checklist for AI Sessions

Before starting an AI-assisted coding session:

- [ ] Have genie-context.md ready (always)
- [ ] Identify which component you're working on
- [ ] Find relevant interface contracts
- [ ] Locate phase guide section for timeline
- [ ] Prepare specific task description
- [ ] Know success criteria for the task
- [ ] Have error messages/logs if debugging

This structure ensures efficient use of AI assistance while maintaining full system understanding and correct implementation.
