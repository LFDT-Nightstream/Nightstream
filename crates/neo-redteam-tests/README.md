# Neo Red-Team Security Tests

Security-focused tests designed to detect vulnerabilities and verify that exploit attempts are properly rejected by the Neo protocol.

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
| `bridge_rejects_different_ccs` | **🚨 ACTIVE VULNERABILITY** - Currently detects cross-CCS proof acceptance attacks |
| `bridge_rejects_cross_proof_swap` | Tests prevention of cross-proof swap attacks |
| `bridge_rejects_different_public_inputs` | Validates public input consistency checks |
| `e2e_rejects_tampered_proof` | End-to-end tampered proof detection |

## Key Security Findings

✅ **AJTAI BINDING VULNERABILITY FIXED**: Bridge now enforces fail-closed behavior, rejecting proof generation when `MEWitness.ajtai_rows` is missing.

🚨 **CCS BINDING VULNERABILITY DETECTED**: Protocol incorrectly accepts proofs verified against different constraint systems - **needs protocol-level fix**.

## Security Status

- **Parameters**: ✅ Robust validation with proper guard checks
- **Ajtai Commitments**: ✅ Fixed via fail-closed bridge guards
- **CCS Binding**: ❌ **Active vulnerability** - proofs not bound to specific constraint systems  
- **Public IO**: ✅ Tamper detection working
- **E2E**: ✅ Proof tampering properly detected

## Usage

```bash
# Run all security tests
cargo test --package neo-redteam-tests --features redteam

# Run specific vulnerability detection
cargo test --package neo-redteam-tests --features redteam forged_commitment_should_fail_without_ajtai_rows
cargo test --package neo-redteam-tests --features redteam bridge_rejects_different_ccs
```
