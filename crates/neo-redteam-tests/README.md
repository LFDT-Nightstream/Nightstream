# neo-redteam-tests

Security-focused tests designed to detect vulnerabilities and verify that exploit attempts are properly rejected by the Nightstream protocol.

## Running Tests

**By design, these tests only run when explicitly requested:**

```bash
# Skip all red-team tests (default)  
cargo test --package neo-redteam-tests

# Run red-team tests (with feature flag)
cargo test --package neo-redteam-tests --features redteam
```

This prevents accidental execution during normal development while ensuring security tests are available for dedicated security auditing.

## Red-Team Test Suite

| Test Name | Security Check |
|-----------|----------------|
| `rlc_guard_blocks_unsafe_params` | Verifies RLC soundness guard rejects unsafe parameters |
| `safe_params_are_accepted` | Ensures valid parameters are properly accepted |
| `guard_inequality_boundary_case` | Tests boundary conditions of parameter guard inequalities |
| `forged_commitment_should_fail_without_ajtai_rows` | Detects forged commitment attacks when Ajtai binding is missing |
| `honest_commitment_passes_with_ajtai_rows` | Verifies honest commitments work with proper Ajtai binding |
| `forged_commitment_fails_with_ajtai_rows` | Confirms forged commitments are rejected with Ajtai binding enforced |
| `bridge_rejects_public_io_tamper` | Tests that tampering with public inputs is detected |
| `bridge_rejects_different_ccs` | CCS binding verification - properly rejects cross-CCS proof attacks |
| `bridge_rejects_cross_proof_swap` | Tests prevention of cross-proof swap attacks |
| `bridge_rejects_different_public_inputs` | Validates public input consistency checks |
| `e2e_rejects_tampered_proof` | End-to-end tampered proof detection |
| `rt26_verify_wrong_public_input_must_fail` | **#26** - Verify with wrong `(CCS, public_input)` binding check |
| `rt26_verify_wrong_ccs_must_fail` | **#26** - Verify with wrong CCS structure must fail |
| `rt29_proof_bundle_too_large_rejected_fast` | **#29** - DoS prevention via bundle size guard before decompression |
| `rt7_range_violation_digits_out_of_range_rejected` | **#7** - Range violation detection |
| `rt23_public_io_replay_must_fail` | **#23** - Public input replay under different `(c,X,y,r)` parameters |
| `rt25_header_io_ordering_mutation_must_fail` | **#25** - Hash-MLE IO ordering / header mismatch detection |
| `rt1_or_24_missing_ajtai_rows_rejected_before_snark` | **#1/#24** - Missing Ajtai binding enforcement |
| `rt13_non_invertible_rhos_fail` | **#13** - Non-invertible ρ differences in Π_RLC verification |
| `rt2_digit_commitment_tamper_must_fail_recomposition` | **#2** - Commitment recomposition without per-digit openings |
| `rt11_wrong_r_on_digit_must_fail` | **#11** - Wrong `r` on digits in DEC verification |
| `rt8_base_mismatch_in_verify_must_fail` | **#8** - Base mismatch at verify-time |
| `rt22_digit_bundle_as_parent_must_fail_binding` | **#22** - Parent vs digit binding mismatch |

## Security Status

- **Parameters**: ✅ Robust validation with proper guard checks
- **Ajtai Commitments**: ✅ Fixed via fail-closed bridge guards  
- **CCS Binding**: ✅ Fixed - proofs now properly bound to specific constraint systems
- **Public IO Binding**: ✅ Strong binding prevents input replay attacks
- **Range Constraints**: ✅ Fixed via circuit accumulator anchoring and fail-fast validation
- **DoS Protection**: ✅ Bundle size guards prevent decompression attacks  
- **Header Integrity**: ✅ IO ordering mutations detected and rejected
- **Π_RLC Security**: ✅ Non-invertible ρ differences properly rejected
- **Π_DEC Security**: ✅ Commitment recomposition and consistency checks enforced
- **Bridge Binding**: ✅ Parent vs digit bundle mismatch detection
- **E2E**: ✅ Proof tampering properly detected

## Usage

```bash
# Run all security tests
cargo test --package neo-redteam-tests --features redteam

# Run specific vulnerability detection
cargo test --package neo-redteam-tests --features redteam forged_commitment_should_fail_without_ajtai_rows
cargo test --package neo-redteam-tests --features redteam bridge_rejects_different_ccs

# Run new red team tests for specific attack vectors
cargo test --package neo-redteam-tests --features redteam rt26_verify_wrong_public_input_must_fail
cargo test --package neo-redteam-tests --features redteam rt29_proof_bundle_too_large_rejected_fast
cargo test --package neo-redteam-tests --features redteam rt7_range_violation_digits_out_of_range_rejected
cargo test --package neo-redteam-tests --features redteam rt1_or_24_missing_ajtai_rows_rejected_before_snark

# Run folding-level security tests
cargo test --package neo-redteam-tests --features redteam rt13_non_invertible_rhos_fail
cargo test --package neo-redteam-tests --features redteam rt2_digit_commitment_tamper_must_fail_recomposition  
cargo test --package neo-redteam-tests --features redteam rt11_wrong_r_on_digit_must_fail
cargo test --package neo-redteam-tests --features redteam rt8_base_mismatch_in_verify_must_fail
cargo test --package neo-redteam-tests --features redteam rt22_digit_bundle_as_parent_must_fail_binding
```
