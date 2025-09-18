#!/usr/bin/env python3

from symbol_operators import SymbolOperator, PatternLearner, AdaptiveSymbolGenerator

# Example scientific instrument log lines
sample_logs = [
    "2024-01-15T10:30:45.123Z INFO [SENSOR-01] Temperature: 23.5°C, Pressure: 1013.25 hPa",
    "2024-01-15T10:30:46.456Z WARN [SENSOR-02] Temperature: 45.2°C, Pressure: 1015.10 hPa",
    "2024-01-15T10:30:47.789Z INFO [SENSOR-01] Temperature: 23.6°C, Pressure: 1013.30 hPa",
    "2024-01-15T10:30:48.012Z ERROR [SENSOR-03] Temperature: -5.1°C, Pressure: 1010.00 hPa",
]

print("=" * 60)
print("PATTERN LEARNING FROM SAMPLES")
print("=" * 60)

# Learn patterns from individual samples
for i, log in enumerate(sample_logs[:2]):
    pattern = PatternLearner.learn_from_sample(log)
    print(f"\nSample {i+1}:\n{log}")
    print(f"Learned pattern:\n{pattern}")

# Generalize patterns
print("\n" + "=" * 60)
print("PATTERN GENERALIZATION")
print("=" * 60)

individual_patterns = [PatternLearner.learn_from_sample(log) for log in sample_logs]
generalized = PatternLearner.generalize_patterns(individual_patterns)
print(f"\nGeneralized pattern from all samples:\n{generalized}")

# Learn structure
structures = PatternLearner.learn_structure_from_samples(sample_logs)
print(f"\nDiscovered structures:")
for name, pattern in structures:
    print(f"  {name}: {pattern[:80]}...")

print("\n" + "=" * 60)
print("SYMBOL OPERATORS - INNER/OUTER PRODUCTS")
print("=" * 60)

# Create operators for different pattern types
timestamp_op = SymbolOperator(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z', 'TIMESTAMP')
sensor_op = SymbolOperator(r'\[SENSOR-\d{2}\]', 'SENSOR')
temp_op = SymbolOperator(r'Temperature: -?\d+\.\d+°C', 'TEMP')
pressure_op = SymbolOperator(r'Pressure: \d+\.\d+ hPa', 'PRESSURE')

test_log = sample_logs[0]
print(f"\nTest log:\n{test_log}")

# Inner products (similarity scores)
print("\nInner products (pattern density):")
print(f"  Timestamp: {timestamp_op.inner_product(test_log):.3f}")
print(f"  Sensor ID: {sensor_op.inner_product(test_log):.3f}")
print(f"  Temperature: {temp_op.inner_product(test_log):.3f}")
print(f"  Pressure: {pressure_op.inner_product(test_log):.3f}")

# Outer products (extraction and transformation)
print("\nOuter products (extraction):")
for op in [timestamp_op, sensor_op, temp_op, pressure_op]:
    extracted, transformed = op.outer_product(test_log)
    print(f"\n  {op.name}:")
    print(f"    Extracted: {extracted}")
    if op == pressure_op:  # Show transformation for last one
        print(f"    Transformed: {transformed}")

# Composite operators
print("\n" + "=" * 60)
print("COMPOSITE OPERATORS")
print("=" * 60)

measurement_op = temp_op.compose(pressure_op)
extracted, transformed = measurement_op.outer_product(test_log)
print(f"\nCombined Temperature + Pressure operator:")
print(f"  Extracted: {extracted}")
print(f"  Transformed: {transformed}")

# Adaptive learning
print("\n" + "=" * 60)
print("ADAPTIVE PATTERN EVOLUTION")
print("=" * 60)

# Start with a simple pattern
adaptive_sensor = AdaptiveSymbolGenerator(r'\[SENSOR-\d{2}\]', 'ADAPTIVE_SENSOR')

# Test with variations
test_cases = [
    "[SENSOR-01]",
    "[SENSOR-99]",
    "[SENSOR-001]",  # Three digits - should be false negative initially
    "[SENS-01]",      # Wrong prefix - should be false positive if matched
]

print("\nTesting adaptive pattern:")
for case in test_cases:
    matches = adaptive_sensor.test_and_adapt(case, expected_matches=[case] if "SENSOR-" in case else [])
    print(f"  '{case}': {'✓ Matched' if matches else '✗ No match'}")

# Pattern differences
print("\n" + "=" * 60)
print("PATTERN DIFFERENCES")
print("=" * 60)

log_level_op = SymbolOperator(r'\b(INFO|WARN|ERROR)\b', 'LOG_LEVEL')
all_caps_op = SymbolOperator(r'\b[A-Z]+\b', 'ALL_CAPS')

diff_matches = all_caps_op.difference(log_level_op, test_log)
print(f"\nAll-caps words that aren't log levels in:\n{test_log}")
print(f"Result: {diff_matches}")

print("\n" + "=" * 60)
print("HIERARCHICAL PATTERN DECOMPOSITION")
print("=" * 60)

# Demonstrate hierarchical extraction
operators = [timestamp_op, log_level_op, sensor_op, temp_op, pressure_op]
remaining_text = test_log

print(f"\nOriginal: {test_log}\n")
for op in operators:
    extracted, remaining_text = op.outer_product(remaining_text)
    if extracted:
        print(f"{op.name:12} -> Extracted: {extracted}")

print(f"\nRemaining text after all extractions: '{remaining_text}'")