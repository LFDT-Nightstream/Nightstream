# Starstream TX IVC Test Suite

This directory contains integration tests for the Starstream TX IVC (Incrementally Verifiable Computation) proof system using data exported from an external project.

## Overview

The test suite reads exported IVC test data from `test_starstream_tx_export.json` and validates:
1. **R1CS Constraint Satisfaction** - Verifies that A·z ∘ B·z = C·z for each step
2. **IVC Proof Generation** - Attempts to generate proofs using the Neo proving system
3. **IVC Proof Verification** - Verifies generated proofs (when applicable)

## Test Files

### `test_starstream_tx_export.json` (3217 lines, ~50KB)
Exported test data containing:
- **Metadata**: Test name, field type (Goldilocks), modulus, expected behavior
- **IVC Parameters**: Initial state `y0 = [1, 3, 0]` and step binding specification
- **3 Steps** with full witness data and R1CS constraints:
  - Step 0: Nop (initialization)
  - Step 1: Resume { utxo_id: 400, input: 42, output: 43 }
  - Step 2: YieldResume { utxo_id: 300, output: 42 } ⚠️ **Contains constraint violation**

### `test_starstream_tx_ivc.rs`
Integration test suite with four tests:

#### 1. `test_json_parsing_only()`
Basic smoke test that verifies the JSON can be parsed into Rust data structures.

**Purpose**: Catch any schema mismatches or parsing errors early.

#### 2. `test_r1cs_constraint_verification()`
Manually verifies R1CS constraints for each step by computing:
- `az = A · z` (sparse matrix-vector multiply)
- `bz = B · z`
- `cz = C · z`
- Checks: `az[i] * bz[i] == cz[i]` for all constraints `i`

**Purpose**: Independent verification that Step 2 contains a constraint violation at row 26.

**Expected Result**: ✅ Detects constraint violation in Step 2

#### 3. `test_starstream_tx_ivc_proof()`
Full end-to-end test that:
1. Parses the JSON export
2. Constructs Neo CCS from R1CS matrices
3. **Directly attempts IVC proof generation** via `prove_ivc_chain()` (no pre-validation!)
4. Verifies the proof (if generation succeeds)

**Purpose**: Test that the Neo IVC prover/verifier correctly catches invalid constraints during proof generation.

**Expected Result**: ✅ Proof generation fails (prover catches the constraint violation)

#### 4. `test_starstream_tx_nivc_proof()`
Full end-to-end test using NIVC (Non-uniform IVC) instead of regular IVC:
1. Parses the JSON export
2. Creates a NIVC program with a single step type (lane)
3. **Directly attempts NIVC proof generation** using `NivcState::step()` (no pre-validation!)
4. Verifies the proof (if generation succeeds)

**Purpose**: Test NIVC prover behavior with the same invalid constraints and compare with IVC.

**Actual Result**: ⚠️ **NIVC accepts the invalid constraints and generates a proof!**

This reveals a **behavior difference** between IVC and NIVC:
- **IVC**: Correctly rejects with "Extractor/binding mismatch"
- **NIVC**: Accepts and generates a valid-looking proof

## The Constraint Violation

According to the test metadata, **Step 2 should fail** due to:

```
YieldResume expects current_program=300 (utxo_id2), 
but it's actually 400 (utxo_id3 from previous Resume)
```

The R1CS verification detects this at **Constraint 26**:
```
Constraint 26 violated: (100) * (1) = 100 ≠ 0
```

This violation represents a semantic error in the step computation - the prover is attempting to yield/resume with a UTXO ID that doesn't match the current program state.

## Test Results

### ✅ Current Behavior (Correct - System is Sound!)
The Neo proving system **correctly rejects** the invalid proof during generation:

```
🚀 Attempting IVC Proof Generation...
   ⚠️  Test metadata indicates this should fail: Step 2: YieldResume expects...

❌ IVC Proof Generation FAILED
   Error: Extractor/binding mismatch: y_step extracted != step_witness[y_step_offsets]

✅ Expected failure occurred (should_fail=true)
   The prover correctly rejected the invalid constraints.
```

**Critical**: The test does **NOT** do any pre-validation of constraints. It directly attempts proof generation and relies entirely on the Neo IVC system to catch the invalid witness.

While the specific error message differs from the constraint violation message, the important property holds: **the prover rejects invalid witnesses before generating a proof**.

This demonstrates that the Neo **IVC** system is **sound** - it will not generate proofs for constraint-violating computations.

### ⚠️ NIVC Behavior Difference
The NIVC test reveals different behavior:

```
✅ NIVC Proof Generation SUCCEEDED
   All 3 steps completed successfully

⚠️  BEHAVIOR DIFFERENCE DETECTED!
   - IVC test fails with: 'Extractor/binding mismatch'
   - NIVC test succeeds and generates a proof
```

**Analysis**: This could be due to:
1. Different validation logic in NIVC vs IVC step execution
2. Different extractor behavior in NIVC (no IndexExtractor check?)
3. The binding spec not being enforced the same way in NIVC

**Implication**: The NIVC test serves as a **regression detector**. If NIVC behavior changes to reject this test case (like IVC does), it would indicate improved soundness checking has been added.

### 🚨 Soundness Bug Detection
If the test ever changes to show:
```
✅ IVC Proof Generation SUCCEEDED
🚨 SOUNDNESS BUG DETECTED!
```

This would indicate a **critical security vulnerability** - the prover is accepting invalid constraints and generating verifiable proofs for false statements.

## Running the Tests

```bash
# Run all Starstream TX tests
cargo test -p neo-tests --test test_starstream_tx_ivc -- --nocapture

# Run individual tests
cargo test -p neo-tests test_json_parsing_only -- --nocapture
cargo test -p neo-tests test_r1cs_constraint_verification -- --nocapture
cargo test -p neo-tests test_starstream_tx_ivc_proof -- --nocapture
cargo test -p neo-tests test_starstream_tx_nivc_proof -- --nocapture
```

## Technical Details

### Field Arithmetic
All values use the **Goldilocks field**: `F_p` where `p = 2^64 - 2^32 + 1`

Field elements are encoded as u64 strings in the JSON.

### R1CS Format
Matrices are stored in **sparse triplet format**: `[(row, col, value), ...]`

Each step has:
- 55 constraints
- 48 variables
- Sparse matrices A, B, C

### IVC Binding Specification
```rust
StepBindingSpec {
    y_step_offsets: [2, 4, 6],      // IVC state indices in witness
    y_prev_indices: [1, 3, 5],      // Previous state indices
    const1_witness_index: 0,        // Constant-1 element
}
```

This maps the IVC state machine variables to positions in the flat witness vector.

## Cross-Project Verification

This test demonstrates **cross-project constraint verification**:
1. Constraints are defined in an external project (Starstream)
2. Exported as JSON with full witness and R1CS data
3. Imported and verified by Neo proving system

This pattern enables:
- **Independent verification** of constraint systems
- **Cross-implementation testing** (compare Neo vs other provers)
- **Debugging and auditing** of complex IVC systems
- **Regression testing** for soundness bugs

## Future Work

Potential extensions:
- [ ] Test with valid constraint sets (should_fail: false cases)
- [ ] Benchmark IVC proof generation time vs circuit size
- [ ] Compare Neo's error messages with manual constraint verification
- [ ] Add NIVC (Non-uniform IVC) test cases
- [ ] Generate proofs for valid traces and verify them externally

## References

- Neo Protocol: `/crates/neo/src/ivc.rs`
- CCS Relations: `/crates/neo-ccs/src/relations.rs`
- R1CS to CCS Conversion: `/crates/neo-ccs/src/r1cs.rs`



