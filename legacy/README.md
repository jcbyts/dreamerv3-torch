# Legacy Code

This folder contains code that was removed from the main vanilla dreamer implementation but preserved for future use.

## Linear Probing Code

The following files contain linear probing functionality that was part of the hierarchical dreamer implementation:

- `eval_ego_linear_probe.py` - Main linear probing evaluation script
- `linear_probing_analysis.py` - Analysis tools for linear probing results  
- `simple_linear_probing.py` - Simplified linear probing without sklearn dependencies
- `debug_linear_probing.py` - Debug utilities for linear probing
- `test_linear_probe_integration.py` - Integration tests for linear probing

## Usage

To re-enable linear probing in the future:

1. Move the desired files back to the main directory
2. Add the linear probing imports and scheduling back to `dreamer.py`
3. Fix any signature mismatches between the linear probing functions and the calling code
4. Test that the linear probing works with vanilla dreamer models

## Notes

- The linear probing code was originally designed for hierarchical models
- Some function signatures may need adjustment to work with vanilla dreamer
- The code includes both sklearn-based and custom implementations
- VizDoom environment integration is included for ground truth factor extraction
