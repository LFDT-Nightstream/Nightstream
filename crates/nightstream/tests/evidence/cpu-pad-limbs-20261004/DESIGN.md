# Design: reach the same 38.816910-second CPU target

## Contract and assumptions

Keep the current four-step CPU workload, key, package, proof bytes, all opening
checks, release build, 300-second per-command cap and 16-GiB RSS guard. The last
median is 60.307949 seconds. These are requirements; the current per-block
arithmetic and scheduling are implementation choices. A persistent full key
cache is not necessary and does not fit alongside the current working set.

## Grounded flow

Prover.extend runs PiCCS, PiRLC, PiDEC and fresh witness completion. PiCCS and
PiDEC both calculate Pad and genuine matrix openings. The terminal verifier
recomputes commitments and both opening families independently. Matrix storage
already has a shared owner. The current Pad evaluator visits each degree-54
witness block, calculates its 54 Boolean equality weights, applies the linear
bar transform, and adds the full ring product for every witness. Signed-unit
products use exact integer shift sums. This repeats millions of products even
though equality weights have a tensor factorization.

At the final state, each main witness has about 1.4 million distinct signed
masks among 1.45 million nonzero blocks. Grouping identical blocks therefore
cannot remove most of the work. The public key values are pseudorandom and
have no matching equality-weight structure.

## Serial arena

Rubric: exact semantics; removal of measured work; bounded storage; unchanged
caller interface; simple ownership. This is a serial comparison, not an
independent cross-review, under the repository's no-subagent instruction.

A. Exact convolution kernel: keep per-block execution but replace raw shifted
vector sweeps by Karatsuba or a batched integer convolution. It can help both
commitments and openings. However, the current kernel already uses SIMD
addition, and small signed multipliers make a field-multiplication kernel
potentially slower. It needs a direct comparison, not a complexity claim.

B. Factor Pad equality weights before multiplication: choose the next power
of two above D, L=64. A ring block starts at 54*b, so its offset modulo L has
32 possible phases. Its 54 weights cross at most one L boundary. Therefore
bar(weights_for_block) = h0 * head[phase] + h1 * tail[phase], with 64 small,
point-dependent ring templates in total. Accumulate weighted witness
coefficients by phase first, then multiply each aggregate by its template.
This is exact distributivity and uses no inversion; Boolean challenge values
remain valid. It changes O(blocks*D^2) ring work into O(blocks*D) field work plus
O(32*D^2) final products per witness. It requires only per-task aggregates and
small factored weight tables, not a dense key or assignment cache.

C. Full key caching or pipelining children with fresh commitments: reject for
the existing memory bound or the actual child-to-fresh dependency. Neither
may replace terminal recomputation with a carried digest.

Select B first. It eliminates repeated arithmetic rather than only rescheduling
it. Keep A as the separate commitment-side candidate if the measured result
still needs it. No public caller, transport, or lifecycle ownership changes;
no shallow forwarding layer or caller-managed stage is added.

## Usage and shape

Public calls remain prover.prove, prover.extend and verifier.verify. Internally:

    pad_openings(witnesses: &[SuperneoZBlocks], point: &[K]) -> Vec<[K; D]>

The private Pad implementation owns phase templates, factored high weights,
per-task weighted coefficient sums, and final exact ring products. Matrix
openings retain their existing EqualityWeights. Pad is computed from the
original point so zero factors never require division. The dimensions are
derived from D and the point; existing TASK_BLOCKS supplies the task partition.
Dense, packed and zero witnesses share the same algebra; representation-specific
coefficient reads remain inside SuperneoZBlocks.

For block b, q=floor(54*b/64), o=(54*b mod64):
  head_o[j] = chi_low(o+j) if o+j<64, otherwise 0
  tail_o[j] = chi_low(o+j-64) if o+j>=64, otherwise 0
  result = sum_o bar(head_o)*sum_{b in phase o}(chi_high(q)*Z_b)
                 + bar(tail_o)*sum_{b in phase o}(chi_high(q+1)*Z_b).
The tail term is zero when the block does not cross. Full D-lane weights,
including carrier padding, remain present.

## Validation and risks

Compare all D coefficients against the original dense ring-product formula,
including phase boundaries, partial tasks, mixed storage, and zero/one point
coordinates. Then compare fresh full proof bytes and terminal rejection cases,
and run the unchanged end-to-end benchmark. A component improvement is not
proof of the 38.8-second goal. Potential risk: field additions and aggregate
reduction can outweigh the removed shifts; reject the implementation if the
complete workload does not improve.

## Implementation evidence and revised decision

The factored Pad implementation passes the original dense ring-product oracle
on signed masks, dense field values, virtual zeros, every alignment phase,
partial task boundaries, extension points and Boolean points. Production entry
points retain their shape and real-witness checks. Since a nonzero carrier has
at least 54 coordinates, a valid point has at least six coordinates.

The caller still uses `prover.prove`, `prover.extend` and `verifier.verify`.
The selected internal surfaces are:

```rust,ignore
// neo-reductions: exact complete Pad results from a transcript point.
fn pad_openings(witnesses: &[SuperneoZBlocks], point: &[K]) -> Vec<[K; D]>;
// neo-math: bounded integer representation, with private limb arrays.
impl SplitRing {
    pub fn from_wide256(coefficients: [[u32; 8]; D]) -> Self;
}
// neo-ajtai: unchanged indexed key, two independent elements per call.
fn split_pair(seed: &[u8; 32], row: u32, columns: [u64; 2]) -> [SplitRing; 2];
```

No caller supplies an arithmetic mode, transform plan or cache lifetime.

The commitment kernel also retains a bounded two-limb representation directly
from the same eight SHAKE output words. For x=2^32, first form A+B*x using
x^2=x-1 and x^6=1. Let a=A mod x, B'=B+floor(A/x), b=B' mod x,
c=floor(B'/x). The stored limbs are a-c and b+c. Their represented value
is congruent modulo p because x^2=x-1. Here -2 <= c <= 2, so each limb has
magnitude below x+3. The existing fewer-than-2^25-block accumulator bound
still fits i64; production's 4,708,530-column maximum is smaller. Overlapping
positive/negative mask bits cancel before accumulation. Independent modular
division, independent SHAKE128 and full ring products are the test oracles.

Empty matrix families now return exact zeros after the existing shape checks.
A pure identity family visits only each worker's actual coordinate interval.
All matrix slots and all result coefficients remain present. Existing row,
window, geometric-run, identity, empty-slot and terminal-rejection tests pass.

A separate exact 128-point finite-field transform prototype compared complete
commitments on the first existing 65,536-column task of saved real witnesses.
No floating-point arithmetic was used. The prototype was removed from production
sources after measurement; its source and failed/intermediate trials remain
in the scratch evidence. The same private key API and scalar oracles were used.

| Prototype, milliseconds | Eight witnesses | One fresh witness |
|---|---:|---:|
| Production shift sums, including key generation | 410.338 | 210.364 |
| Transforms, including key generation | 358.819 | 249.487 |
| Cached canonical key, shift sums | 268.686 | 68.428 |
| Cached transformed key | 145.717 | 30.923 |
| Canonical key construction, paid separately | 158.326 | 159.362 |
| Additional transform construction, paid separately | 66.450 | 73.299 |

The ordinary key tile takes 622,854,144 bytes; its transformed form takes
1,476,395,008 bytes. The complete benchmark prefix would take 15,259,774,464
bytes in canonical form or 36,171,317,248 bytes in transformed form. These
are key payloads only. Construction and the existing working data add memory.

This rejects uncached transforms as the sole replacement: they regress the
single-witness path. Cached transforms are a promising performance experiment,
not a demonstrated 38.8-second solution. The tile timings exclude cache
construction and do not model the full working set or memory bandwidth.
A whole-lifecycle result must pay construction inside the existing timed run.

The user was asked whether to retain the 16-GiB guard or permit one explicitly
larger-memory experiment. No larger-memory run has been started. Within the
current guard, keep only the bounded arithmetic and opening changes. There is
not yet measured evidence that these changes reach the requested target.

If a larger-memory experiment is approved, the smallest coherent cache design
is an immutable verifier-derived key owned by the compiled circuit and shared
by its prover and verifier. The commitment module owns expansion, transform
and exact products. It must validate all witness coordinates and recompute
every commitment from the current witness, including during terminal checks.
No cache of prover-supplied commitment claims is permitted. The public lifecycle
call site stays unchanged. A cold full run, exact proof bytes, both opening
rejections and the exact new memory peak decide whether this design is useful.
The experiment would not satisfy the present memory contract by itself.
