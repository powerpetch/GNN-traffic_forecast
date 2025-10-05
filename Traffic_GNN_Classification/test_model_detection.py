"""
Quick test to check if model files can be found
"""
import os
from pathlib import Path

print("="*60)
print("Model File Detection Test")
print("="*60)

# Test 1: Current directory
print(f"\n1. Current working directory:")
print(f"   {os.getcwd()}")

# Test 2: Check if outputs folder exists
outputs_relative = Path("outputs")
print(f"\n2. Relative path 'outputs/':")
print(f"   Exists: {outputs_relative.exists()}")
if outputs_relative.exists():
    print(f"   Absolute: {outputs_relative.absolute()}")

# Test 3: Check from script location (like dashboard does)
script_dir = Path(__file__).parent
project_root = script_dir.parent if script_dir.name == "app" else script_dir
outputs_absolute = project_root / "outputs"

print(f"\n3. From script location (dashboard method):")
print(f"   Script dir: {script_dir}")
print(f"   Project root: {project_root}")
print(f"   Outputs path: {outputs_absolute}")
print(f"   Exists: {outputs_absolute.exists()}")

# Test 4: List all .pth files
print(f"\n4. Searching for .pth files...")

model_locations = [
    outputs_absolute / "best_model.pth",
    outputs_absolute / "enhanced_training" / "enhanced_model.pth",
    outputs_absolute / "optimized_training" / "optimized_model.pth",
    outputs_absolute / "quick_training" / "quick_model.pth",
]

found_models = []
for model_path in model_locations:
    if model_path.exists():
        size = model_path.stat().st_size / (1024 * 1024)
        found_models.append((model_path, size))
        print(f"   ✅ Found: {model_path.name} ({size:.1f} MB)")
        print(f"      Path: {model_path}")
    else:
        print(f"   ❌ Missing: {model_path}")

# Test 5: Summary
print(f"\n" + "="*60)
print(f"Summary:")
print(f"  Total models found: {len(found_models)}")
print(f"  Expected path: {outputs_absolute.absolute()}")
print("="*60)

if len(found_models) == 0:
    print("\n⚠️  No models found!")
    print("\nPossible solutions:")
    print("  1. Run training: python train.py")
    print("  2. Check if you're in the correct directory")
    print("  3. Verify outputs/ folder exists")
else:
    print(f"\n✅ Found {len(found_models)} model(s)!")
    print("\nYou can now use the dashboard to view these models.")
