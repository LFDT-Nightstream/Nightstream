# PiCCS canonical rectangular protocol

## Contract

The protocol follows the local SuperNeo paper, Section 7.3 and Appendix D.4.
It makes one declared change: row and column domains can have different
sizes, so the paper's joint SumCheck is split into one FE SumCheck and one NC
SumCheck.

The implementation is pinned to the local paper snapshot by content:

| Source | SHA-256 |
|---|---|
| `docs/superneo-paper/07-7-neo-s-folding-scheme-for-ccs.md` | `2ed776426ad25c37e9dfe8ee8970dc465605364d9557661ebb9eee6c75de0aed` |
| `docs/superneo-paper/13-d-deferred-theorems-and-proofs.md` | `bb542f19749b44c037af2a430ed72460fc8fbb07c8a748ff8ac454a0f2a3c734` |

The local Markdown has two known formula-display defects. The target is the
absolute joint-Q target, so the carried block starts at `2K+k`. The strict
range polynomial has roots `-(b-1), ..., b-1`. These corrections are explicit
in the Lean `PaperJoint` model and are not rectangular-protocol changes.

This split preserves the paper's source order, signs, target, range
polynomial, coefficient order, and absolute gamma exponents. It does not add a
coefficient/lane SumCheck axis or delay the column opening.

The concrete Goldilocks profile accepts at most 61 fresh sources and at most
`k_rho = 14` running sources. Both prover and verifier enforce these limits
before transcript sampling.

## Polynomials

Let `K` be the number of fresh CCS sources and `k` the number of running CE
sources. Let `I(i,j,l) = i + k*j + k*t*l`, with zero-based coordinates.

The row polynomial contains:

```text
Q_FE(r) =
  eq(r, beta_r) * sum_i gamma^i * CCS_i(r)
  + eq(r, r_old) * sum_(i,j,l) gamma^(2K+k+I(i,j,l)) * Eval_(i,j,l)(r)
```

Its public initial claim is the same absolute carried target from the paper.

The column polynomial contains:

```text
Q_NC(c) =
  eq(c, beta_m) * sum_i gamma^(K+i) * Range_i(c)
```

Its public initial claim is zero. The raw `y_zcol` opening is materialized at
the NC terminal point.

For a square domain and one shared equality point:

```text
Q_joint(x) = Q_FE(x) + Q_NC(x)
```

The Rust `PaperJointSquareOracle` executes this baseline. Lean proves the
pointwise and Boolean-sum identities in
`Nightstream.SuperNeo.Folding.PiCCS.PaperRectangular`.

## Transcript and proof

`PiCcsProofVariant::PaperRectangularV1` has:

- one row equality point and one column equality point;
- one shared gamma challenge;
- exactly `ell_n` FE rounds and `ell_m` NC rounds;
- fixed `d_sc + 1` coefficients in every round;
- no `alpha` or `beta_a` lane challenge;
- one direct row opening and one direct column opening in each output.

The neutral driver in `pi_ccs_rectangular.rs` owns transcript binding,
SumCheck message encoding, proof assembly, and verifier replay. Both engines
call this driver.

## Verification obligations

The verifier recomputes all transcript challenges. It checks:

- proof variant and fixed round widths;
- FE and NC initial claims;
- all SumCheck transitions and terminal claims;
- redundant stored challenges and final values;
- fold digest on the proof and every output;
- source order, commitments, public inputs, row points, and column points;
- zero values in inactive packed public-input columns;
- the canonical FE and NC terminal equations.

## Evidence and scope

The Rust differential tests establish byte equality between the independent
direct engine and the cached optimized engine for the tested square and both
rectangular directions. The Lean theorem is model-level. The generated
Rust-to-Lean artifact establishes exact fixed-shape gamma-layout conformance.
It does not establish full transcript-byte or matrix-evaluator conformance.

The legacy `SplitNcV1` and `BlockLaneNcDelayedV1` code is available only below
`optimized_engine::legacy_split_nc` for the accelerator and recursive-circuit
migration. It is not the canonical protocol. The fixed-profile recursive
circuit does not yet establish R1CS conformance with `PaperRectangularV1`.
