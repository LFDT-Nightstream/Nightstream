# Wide sampler integration checkpoint

Base: `be5b7fcdc`. Worktree branch: `nico/pirlc-wide-sampler-integration`.
Production selection and generated packages remain unchanged at this checkpoint.

The candidate implements the merged whole-vector sampler with the existing
hint vocabulary. Its deterministic schedule enters `[4, i]`, reads one joint
four-field block, and advances Poseidon2 once for each of the 17 scalars.
The map is the specified reduction modulo `5^54`, followed by centered digits.
There is no rejection, retry, or shortfall.

## Implemented and proved

- The exact helper program, including integer bounds before field operations,
  limb regrouping, division by five, and the biased CRT check quotient.
- Soundness and constructive completeness of the executable scalar gadget and
  its 17-scalar lifecycle and the complete candidate PiRLC phase.
  Temporary digit words preserve the existing R1CS ring-recipe shape;
  compact CCS ring rows consume the checked digit bits directly.
- The compact sampler matrix plan: 34 Poseidon permutations and 17 range
  gadgets. Accepted rows imply the exact transcript schedule and scalar map.
- A constructive assignment for that plan, preservation of caller columns,
  exact retained-block encodings, and the unchanged strict norm bound `b=2`.
- The challenge forms used by ring products are the checked digit bits minus
  two. They allocate no separate field witness.
- Temporary helper replacement and reconstruction preserve all checked rows.
  The compact assignment omits those helpers.
- The candidate Stage 1 physical prefix through PiDEC and the running
  transition has exact source endpoints. The PiCCS-to-PiRLC and PiRLC-to-PiDEC
  input interfaces use the actual canonical source columns.
- The sampler and all 969 quotient products compose into a 119,153-row
  candidate plan. Its soundness theorem uses the exact sampled challenges.
  Its completeness theorem takes the caller's output and quotient encodings;
  the full package's retained assignment still needs that connection.

The proof entrypoints are in
`NightstreamFPrime/Layout/PiRlcWideSampler/{Completeness,StateSemantics,Norm,Challenges}.lean`.
The executable hint program is
`NightstreamFPrime/Gadgets/Sampling/WideReduction/Program.lean`.

## Component costs

These are sampler costs, not selected F′ totals.

| Measure | One range gadget | Complete 17-scalar sampler |
|---|---:|---:|
| Logical CCS rows | 681 | 14,501 |
| Retained coordinates | 937 | 135,813 |
| Temporary hint helpers | 1,404 | 23,868 |
| Temporary digit words for R1CS recipes | — | 918 |
| Normalized matrix nonzeros | 8,364 with the Poseidon endpoint input | 1,794,350 |

Rows and retained-coordinate counts are proved in Lean. Matrix nonzeros are
measured by exact field arithmetic from Lean-emitted forms and round constants;
the numeric nonzero total is not a Lean theorem.

Each scalar retains 609 bit values and eight field auxiliaries. Each field
auxiliary uses the existing 41-coordinate encoding. The helpers allocate no
rows or matrix entries. Their execution still does work: 256 input-bit hints,
16 accumulator hints and 592 limb/carry-bit hints, and 540 division hints per
scalar. The unchanged checked gadget also allocates its canonical children
and 353 result bits.

The matrix count uses the previous Poseidon endpoint as input, including its
actual external linear layer. It includes coefficient merging and cancellation.
The sampler total excludes the ring-product matrices. A separate measurement
of their left-operand port gives 229,698,543 baseline entries and 16,904,205
candidate entries across 969 products: 212,794,338 fewer entries. These values
come from normalized Lean-emitted forms at all 108 evaluation points. The
other product ports and the complete selected package are not part of that
measurement. The earlier standalone range count of 7,216 used four separate
field inputs; that is a different input geometry.

## Security scope

Let `p` be Goldilocks, `M=p^4`, `N=5^54`, and `r=M mod N`. The proved per-scalar
statistical distance is `δ=r(N-r)/(N*M) < 2^-132`.

`OracleModel.lean` defines a classical ideal block oracle. Its keys are complete
normalized histories of rate additions, including zero additions for digest
advances. A new key receives one joint uniform four-field block. The cache
returns the same raw block for repeated keys. The adaptive comparison counts
all oracle calls, including adversarial calls and repeats, and bounds the
change in any bounded test by `q*δ`.

The comparison keeps raw blocks consistent with their scalars: it samples a
uniform scalar and a raw preimage conditioned on that scalar. It does not put
an unrelated uniform scalar beside an unchanged raw block.

`ScheduleLaw.lean` proves that the 17 scheduled keys are distinct. Its batch law
requires that those keys are absent from the incoming cache. Under that fresh
batch experiment, it gives the exact law used by V6. The `17*L*δ` bound applies
to `L` independent fresh batches. It is not a bound for an arbitrary adaptive
Fiat–Shamir adversary; that experiment uses its full query budget.

V6 still refers to the existing uniform-challenge interactive extractor. Its
`n/|C|` extraction loss remains separate. These modules do not prove a concrete
Poseidon2 ideal-oracle theorem, a Fiat–Shamir extractor, or a quantum-query
bound. Any application of an established cryptographic reduction must state
its model and applicability conditions. The concrete-to-ideal error remains
an explicit, separate premise in `under_block_oracle_assumption`.

## Validation at this checkpoint

The full production library, all Lean tests, and both focused experiment
executables built successfully (7,769 build jobs). All 507 sampler theorem
and circuit audits passed with only the three allowed axioms. The final
executable cost measurement matches the table above. No Rust code or generated
production artifact has changed.

The source sampler allocates 55,403 private DSL values and has 32,623 DSL
rows, including the 918 temporary digit bindings. Structural lowering gives
58,939 R1CS rows and 26,316 R1CS scratch values. The original sampler uses
1,008,848 R1CS rows. These physical figures describe the reference witness
construction; the compact sampler remains 14,501 CCS rows and 135,813
committed coordinates.

The candidate physical prefix through the running transition has 27,716,409
rows and 27,859,538 source columns. These are source-layout counts, not the
committed width. The application and final public layout are not selected
from this candidate yet.

## Reproduce the focused checks

Run from `formal/nightstream-fprime`, through the required validation wrapper:

```sh
timeout --signal=KILL 1500 scripts/validate.sh build NightstreamFPrime NightstreamFPrimeTests measureWideSamplerCost checkWideSamplerHints
timeout --signal=KILL 1500 scripts/validate.sh lean-executable .lake/build/bin/checkWideSamplerHints
timeout --signal=KILL 1500 scripts/validate.sh lean-executable .lake/build/bin/measureWideSamplerCost > /tmp/nightstream-wide-cost-input.jsonl
```

From the repository root:

```sh
timeout --signal=KILL 300 python3 -B tools/recursive-constraint-minimizer/experiments/wide_sampler_matrix_cost.py /tmp/nightstream-wide-cost-input.jsonl
```

The controls execute the exact hint program on 11 boundary inputs, including
`0`, `N-1`, `N`, `N+1`, `p-1`, `p`, `p+1`, and the upper end of the draw domain.
All 681 rows and all 54 digits are checked for every case.

## Still required for the production switch

- Assemble the new sampler into the complete Stage 1 layout, replace the old
  sampler/selector source ownership, and close whole-package preservation and
  witness construction. Measure the changed ring-product matrices and the full
  F′ counts; do not infer them by subtracting component estimates.
- Apply the security model to the selected production transcript and record the
  applicable Fiat–Shamir assumption and query accounting.
- Update native Rust rho derivation and assignment transport, emit the parity
  fixture, then select and regenerate the package, identities, and fixtures.
- Run the required conformance checks and remove the old sampler dependencies.

The selected baseline remains 3,588,191 rows, 149,293,044 committed coordinates,
and 2,857,409,270 normalized matrix nonzeros. No Rust benchmark has been run.
