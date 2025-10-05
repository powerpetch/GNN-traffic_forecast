"""
Test congestion distribution at different times
"""
import sys
import os
import numpy as np
from datetime import datetime, timedelta

# Add app to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from data_processing import generate_time_based_predictions

# Mock data
mock_data = {
    'locations': [(13.7563 + i*0.01, 100.5018 + i*0.01) for i in range(169)],
    'location_types': ['Junction'] * 169,
    'location_names': [f'Location {i}' for i in range(169)]
}

print("="*80)
print("Testing Congestion Distribution at Different Times")
print("="*80)

# Test different times
test_times = [
    (3, False, False, True, "Night (03:00)"),
    (8, True, False, False, "Morning Rush (08:00)"),
    (14, False, False, False, "Afternoon (14:00)"),
    (18, True, False, False, "Evening Rush (18:00)"),
    (22, False, False, True, "Late Night (22:00)"),
]

labels = ['Gridlock (0)', 'Congested (1)', 'Moderate (2)', 'Free-flow (3)']

for hour, is_rush, is_weekend, is_night, desc in test_times:
    preds = generate_time_based_predictions(mock_data, hour, is_weekend, is_rush, is_night)
    counts = np.bincount(preds['congestion'], minlength=4)
    percentages = (counts / len(preds['congestion'])) * 100
    
    print(f"\n{desc}")
    print(f"  Time: {hour}:00 | Rush: {is_rush} | Night: {is_night}")
    print(f"  Distribution:")
    for label, count, pct in zip(labels, counts, percentages):
        bar = "█" * int(pct / 2)
        print(f"    {label:20} {count:3} locations ({pct:5.1f}%) {bar}")
    
    # Show sample
    print(f"  Sample (first 10): {preds['congestion'][:10].tolist()}")

print("\n" + "="*80)
print("Expected Behavior:")
print("  Night (03:00, 22:00)    → Mostly Free-flow (3) and Moderate (2)")
print("  Rush Hours (08:00, 18:00) → More Gridlock (0) and Congested (1)")
print("  Normal Hours (14:00)     → Balanced distribution")
print("="*80)
