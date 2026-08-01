# Gadgets

## Purpose
- **What it is**: CCS circuit gadgets for the embedded verifier: public vector equality (`lhs[i] == rhs[i]`), Ajtai commitment opening (`<L_i, Z> = c_open[i]`), and homomorphic commitment linear combination (`c_next = c_prev + rho * c_step`).
- **Key invariant**: Each gadget produces a `CcsStructure` whose `check_ccs_rowwise_zero` returns `Ok(())` when the witness satisfies the intended constraint, and returns `Err(RowFail)` otherwise.
- **Protocol role**: Used by `neo-fold` to construct the embedded verifier circuit. The public-rho architecture means Fiat-Shamir challenges (rho) are public inputs to these gadgets, not computed in-circuit.

## Target Formulas (Paper -> Rust)

| Paper notation | Paper reference | Rust identifier | Notes |
|---|---|---|---|
| `lhs[i] - rhs[i] = 0` | (embedded verifier) | `public_equality_ccs(len)` | Element-wise equality enforcement |
| `<L_i, Z> = c_open[i]` | (Ajtai opening) | `commitment_opening_from_rows_ccs(rows, msg_len)` | Inner product constraint |
| `c_next = c_prev + rho * c_step` | (folding lincomb) | `commitment_lincomb_ccs(commit_len)` | Homomorphic commitment update |

## Paper Anchors

Source: ./docs/superneo-paper/

- (No direct paper anchor -- gadgets implement the embedded verifier architecture described informally in the protocol.)
- Implicit in the folding pipeline: the verifier circuit must check commitment openings and linear combinations.

## Lean Cross-Reference

| Lean spec | Lean module | Relationship |
|---|---|---|
| (none) | (none) | Gadgets are implementation details of the embedded verifier |

## Contract Surface

| Group | Rust symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Equality | `public_equality_ccs(len)` | fn | Core | CCS enforcing `lhs[k] - rhs[k] = 0` for all k |
| Equality | `multiple_public_equality_constraints(bindings, wit_cols, pub_cols)` | fn | Core | Binding-based equality: `public[i] - witness[j] = 0` |
| Equality | `build_public_vec_eq_witness()` | fn | Helper | Returns `vec![F::ONE]` |
| Opening | `commitment_opening_from_rows_ccs(rows, msg_len)` | fn | Core | CCS for `<L_i, Z_digits> = c_open[i]` |
| Opening | `build_opening_witness(z_digits)` | fn | Helper | Returns `[1, z_digits...]` |
| Lincomb | `commitment_lincomb_ccs(commit_len)` | fn | Core | CCS for `c_next = c_prev + rho * c_step` |
| Lincomb | `build_commitment_lincomb_witness(rho, c_prev, c_step)` | fn | Helper | Returns `(witness, c_next)` |
| Lincomb | `build_commitment_lincomb_public_input(rho, c_prev, c_step, c_next)` | fn | Helper | Full public input vector |

## Invariant Obligations

| Invariant | Verification method | Lean theorem counterpart |
|---|---|---|
| Public equality: valid equality passes `check_ccs_rowwise_zero` | Unit test | (none) |
| Public equality: unequal vectors fail `check_ccs_rowwise_zero` | Unit test | (none) |
| Commitment opening: `<L_i, Z> = c_open[i]` round-trip | Unit test | (none) |
| Commitment lincomb: `c_next = c_prev + rho * c_step` round-trip | Unit test | (none) |
| `build_commitment_lincomb_witness` produces correct `u[i] = rho * c_step[i]` | Unit test | (none) |

## Assumption Ledger

| Assumption | Source | Justification |
|---|---|---|
| `rho` is a public input (not computed in-circuit) | Neo architecture | Public-rho embedded verifier design |
| Gadget CCS structures are composed via `direct_sum` with the main circuit | neo-fold | Standard CCS composition pattern |
| Goldilocks field type is hardcoded (`type F = Goldilocks`) | Design choice | All Neo circuits operate over Goldilocks |

## Dependency and Consumer Map

Upstream dependencies:
- `neo-ccs::relations`: `CcsStructure`, `check_ccs_rowwise_zero`
- `neo-ccs::matrix`: `Mat<F>`
- `neo-ccs::poly`: `SparsePoly<F>`, `Term<F>`
- `p3-goldilocks`: Goldilocks field type

Downstream consumers:
- `neo-fold`: constructs embedded verifier circuit from these gadgets
- `neo-reductions`: folding-layer commitment verification

## Lean Oracle Conformance

| Test file | Oracle family | What it checks |
|---|---|---|
| (none) | (none) | Gadget correctness verified via unit tests against `check_ccs_rowwise_zero` |

## Quality Expectations

- No `unsafe` (enforced crate-wide)
- Gadget dimensions must be consistent (constraints, public inputs, witness columns all match)
- All values in `public_equality_ccs` are public -- no sensitive data hidden in witness
- Commitment lincomb uses exactly 2L constraints (L multiplications + L additions)

## Acceptance Criteria

- `cargo test -p neo-ccs --release` succeeds
- All gadget invariant tests pass
- Gadget CCS structures pass `check_ccs_rowwise_zero` with correct witnesses

## Out of Scope

- Range-check gadgets (belong to Pi_DEC in neo-reductions)
- Poseidon2 in-circuit gadgets (not needed: public-rho architecture)
- Custom gate optimizations
