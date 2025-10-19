# Neo Diagnostics Examples

This directory contains example constraint diagnostics for testing and demonstration.

## Example Files

### `simple_r1cs_failure.json`

A minimal R1CS constraint failure demonstrating the diagnostic system.

**Constraint**: `z[0] * z[1] = z[2]`  
**Witness**: `z = [2, 3, 7]`  
**Violation**: `2 * 3 ≠ 7` (expected 0, got residual of -1)

**Run the replayer**:
```bash
# Basic replay
cargo run --package neo-diag -- crates/neo-diag/examples/simple_r1cs_failure.json

# Verbose output with gradient details
cargo run --package neo-diag -- crates/neo-diag/examples/simple_r1cs_failure.json -v -g

# Show witness values
cargo run --package neo-diag -- crates/neo-diag/examples/simple_r1cs_failure.json -w

# Generate a regression test
cargo run --package neo-diag -- crates/neo-diag/examples/simple_r1cs_failure.json \
  -t /tmp/test_simple_r1cs.rs
```

## Expected Output

```
📂 Loading diagnostic from: crates/neo-diag/examples/simple_r1cs_failure.json

======================================================================
🔍 CONSTRAINT DIAGNOSTIC
======================================================================

📋 Context:
   Test:        test_diagnostic_example
   Step:        0
   Instruction: Test
   Phase:       Test
   Constraint:  0 (row in CCS)

📐 CCS Structure:
   Hash:       blake3v1:3bf2b3d85...
   Matrices:   3 (t)
   Constraints: 1 (n)
   Variables:   3 (m)

📊 Evaluation:
   Expected:     0000000000000000
   Actual:       00000000ffffffff
   Delta signed: -1

🎯 Gradient Blame Analysis (Top Contributors):
   1. z[2] - gradient: -1, contribution to residual
   2. z[0] - gradient: 3
   3. z[1] - gradient: 2

✅ SUCCESS: Replay matches diagnostic (residual verified)
```

## Generating Your Own Examples

Run any test with diagnostics enabled:

```bash
# Set environment variables
export NEO_DIAGNOSTICS=./diagnostics
export NEO_DIAGNOSTIC_FORMAT=json

# Run tests
cargo test --features prove-diagnostics

# Examples will be in ./diagnostics/
```

## Using in CI

```bash
#!/bin/bash
# Capture diagnostics from failing tests
cargo test --features prove-diagnostics 2>&1 | tee test.log

# Replay all captured diagnostics
for diag in diagnostics/*.json; do
    echo "Replaying: $diag"
    cargo run --package neo-diag -- "$diag"
done
```


