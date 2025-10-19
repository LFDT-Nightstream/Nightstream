# Neo Constraint Diagnostics

Production-grade constraint diagnostic system for debugging CCS/R1CS failures.

## Quick Start

Try the example diagnostic:

```bash
# Run the replayer on the example diagnostic
cargo run --package neo-diag -- crates/neo-diag/examples/simple_r1cs_failure.json

# With verbose output
cargo run --package neo-diag -- crates/neo-diag/examples/simple_r1cs_failure.json -v -g

# Generate a regression test
cargo run --package neo-diag -- crates/neo-diag/examples/simple_r1cs_failure.json \
  -t /tmp/test_example.rs
```

## Features

- **CCS-Native**: Works with arbitrary constraint polynomials
- **Phase-Aware**: Tracks which reduction phase (Π_CCS, Π_RLC, Π_DEC)
- **Gradient-Based Blame**: Mathematically correct sensitivity analysis (∂r/∂z_i)
- **Transcript Capture**: Full reproducibility
- **Leak-Aware Witness Policies**: Minimal disclosure by default
- **Stable Hashing**: Canonical serialization for cross-build stability
- **Replayer Tool**: Verify diagnostics offline and generate regression tests

## Usage

### Enable Diagnostics

```bash
# Enable diagnostic capture
export NEO_DIAGNOSTICS=diagnostics

# Choose format (json | json.gz | cbor)
export NEO_DIAGNOSTIC_FORMAT=json.gz

# Witness policy (row_support | gradient | full)
export NEO_DIAGNOSTIC_WITNESS_POLICY=gradient
export NEO_DIAGNOSTIC_GRADIENT_TOP_K=20

# Safety guard for full witness
export NEO_DIAGNOSTIC_ALLOW_FULL_WITNESS=1
```

### Build with Diagnostics

```bash
# Build with diagnostic support
cargo build --features prove-diagnostics

# Run tests with diagnostics
cargo test --features prove-diagnostics
```

### Replay a Diagnostic

```bash
# Basic replay
cargo run --bin neo-diag diagnostics/test_step_2_constraint_25.json.gz

# Verbose output with gradient details
cargo run --bin neo-diag diagnostics/test_step_2_constraint_25.json.gz --verbose --show-gradient

# Show witness values
cargo run --bin neo-diag diagnostics/test_step_2_constraint_25.json.gz --show-witness

# Generate regression test
cargo run --bin neo-diag diagnostics/test_step_2_constraint_25.json.gz \
  --generate-test tests/regressions/constraint_25.rs
```

## Witness Policies

### RowSupportOnly (Default)
Only captures witness indices that appear in the failing constraint. **Minimal leak**, safest for production.

```bash
export NEO_DIAGNOSTIC_WITNESS_POLICY=row_support
```

### GradientSlice (Recommended for Debugging)
Captures witness indices with large |∂r/∂z_i| (top contributors to residual). **Smart disclosure** based on mathematical sensitivity.

```bash
export NEO_DIAGNOSTIC_WITNESS_POLICY=gradient
export NEO_DIAGNOSTIC_GRADIENT_THRESHOLD=1000
export NEO_DIAGNOSTIC_GRADIENT_TOP_K=20
```

### FullWitness (Use with Caution)
Captures entire witness. **Maximum leak**, requires explicit safety guard.

```bash
export NEO_DIAGNOSTIC_WITNESS_POLICY=full
export NEO_DIAGNOSTIC_ALLOW_FULL_WITNESS=1
```

## Diagnostic Format

### JSON (`.json`)
Human-readable, uncompressed. Good for inspection.

```json
{
  "schema": "neo.constraint.diagnostic@1",
  "context": {
    "test": "test_starstream_tx",
    "step_idx": 2,
    "instruction": "Resume",
    "constraint_idx": 25
  },
  "eval": {
    "expected": "0000000000000000",
    "actual": "ffffffff00000043",
    "delta_signed": "-61"
  },
  "witness": {
    "policy": "gradient_slice",
    "gradient_blame": [
      {
        "i": 11,
        "gradient": "0300000000000000",
        "z_value": "9001000000000000",
        "contribution": "..."
      }
    ]
  }
}
```

### JSON.GZ (`.json.gz`) - Recommended
Compressed JSON, ~10x smaller. Default format.

### CBOR (`.cbor`) - Experimental
Binary format, smallest size. Requires `prove-diagnostics-cbor` feature.

## Integration in Tests

```rust
use neo_fold::diagnostic::{capture_and_export_diagnostic, ContextInfo, FoldingContext};

#[test]
fn test_with_diagnostics() {
    // Enable diagnostics
    std::env::set_var("NEO_DIAGNOSTICS", "diagnostics");
    std::env::set_var("NEO_TEST_NAME", "test_my_constraint");
    
    // Run proof
    let result = prove_ivc_step(...);
    
    // On failure, diagnostic is auto-captured
    match result {
        Err(e) if e.is_constraint_error() => {
            // Diagnostic already saved to diagnostics/
            println!("Constraint failed - diagnostic captured");
        }
        _ => {}
    }
}
```

## CI Integration

```bash
#!/bin/bash
# Auto-generate regression tests from diagnostics

if cargo test 2>&1 | grep "Constraint diagnostic captured"; then
    find diagnostics -name "*.json.gz" | while read diag; do
        test_name=$(basename "$diag" .json.gz)
        cargo run --bin neo-diag "$diag" \
            --generate-test "tests/regressions/${test_name}.rs"
    done
    
    if [ -n "$(ls tests/regressions/*.rs 2>/dev/null)" ]; then
        git add tests/regressions/
        echo "✅ Generated $(ls tests/regressions/*.rs | wc -l) regression tests"
    fi
fi
```

## Architecture

### Gradient-Based Blame

For CCS with f(y_1,...,y_t) and y_j = ⟨M_j[row,:], z⟩:

```
∂r/∂z_i = Σ_j (∂f/∂y_j) * M_j[row, i]
```

For R1CS with r = (Az)(Bz) - Cz:

```
∂r/∂z_i = a_i(Bz) + b_i(Az) - c_i
```

This provides mathematically correct sensitivity, ranking variables by actual contribution to the residual.

### Canonical Hashing

Structure hash computed via:
1. Domain-separated BLAKE3
2. Canonical LE byte encoding
3. Deterministic term ordering

Ensures diagnostics are comparable across builds and platforms.

## Example Output

```
📂 Loading diagnostic from: diagnostics/test_step_2_constraint_25.json.gz

======================================================================
🔍 CONSTRAINT DIAGNOSTIC
======================================================================

📋 Context:
   Test:        test_starstream_tx_should_fail
   Step:        2
   Instruction: Resume
   Phase:       PiCCS
   Constraint:  25 (row in CCS)

📐 CCS Structure:
   Hash:       blake3v1:7a3b2c...
   Matrices:   3 (t)
   Constraints: 55 (n)
   Variables:   48 (m)

📊 Evaluation:
   Expected:     0000000000000000
   Actual:       ffffffff00000043
   Delta:        ffffffff00000043 (canonical)
   Delta signed: -61

🎯 Gradient Blame Analysis (Top Contributors):
   1. z[11] (current_program)
      ∂r/∂z[11] = 0300000000000000 (abs: 3)
      z[11] = 9001000000000000
      contribution = b003000000000000
   
   2. z[18] (expected_program)
      ∂r/∂z[18] = 0200000000000000 (abs: 2)
      z[18] = 2c01000000000000
      contribution = 5802000000000000

🔄 Replaying Constraint...

✅ Replay Results:
   Expected:     0000000000000000
   Computed:     ffffffff00000043
   Delta:        ffffffff00000043
   Delta signed: -61

✅ SUCCESS: Replay matches diagnostic (residual verified)
```

## Performance

- **Capture overhead**: Negligible (only on failure)
- **Storage**: ~10-100KB per diagnostic (compressed)
- **Replay time**: <1ms

## Security Considerations

- Default `RowSupportOnly` policy leaks minimal information
- `GradientSlice` policy leaks ~20 high-sensitivity variables
- `FullWitness` requires explicit opt-in via environment variable
- Diagnostics should not be included in production builds
- Consider `.gitignore`ing the `diagnostics/` directory

## Related Tools

- **neo-diag**: Replay and analyze diagnostics
- **neo-fold**: Core folding protocol with diagnostic hooks
- **neo-tests**: Test infrastructure with diagnostic support

## References

- [Halo2 MockProver](https://zcash.github.io/halo2/user/dev-tools.html)
- [gnark Debug Operations](https://docs.gnark.consensys.net/)
- [Circom Debugging](https://docs.circom.io/getting-started/debugging/)

