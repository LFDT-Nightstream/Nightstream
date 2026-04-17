# RV64IM Main Relation Specification

## Scope

This document specifies the owned RV64IM main relation `R_main^SN` in
`neo-fold-next`.

It fixes the ownership boundary between:

1. the concrete SuperNeo backend contract owned by `riscv-kernel.md`,
2. the owned RV64IM main witness relation that carries the theorem-level
   accumulator meaning for the main lane,
3. and the later recursive or succinct backend that compiles that fixed
   relation.

This document owns:

- the exact theorem that `R_main^SN` must preserve,
- the exact public/private split for that relation,
- the stable authoritative projection exported to later compiler layers,
- and the negative rules forbidding digest-shell replacements.

This document does **not** own:

- the concrete recursive backend choice,
- the concrete outer compression backend,
- the final public Rust API for compressed proofs,
- or the side-opening theorem.

Those remain owned by:

- `riscv-recursive-proof.md`
- `riscv-recursive-instantiation.md`
- `riscv-authoritative-side-proof-bundle.md`

## Normative References

This document is constrained by the following local references:

- `./riscv-kernel.md`
  - the concrete Goldilocks-native SuperNeo backend contract for RV64IM
  - the canonical chunk-local `Π_CCS -> Π_RLC -> Π_DEC` split
- `./riscv-recursive-proof.md`
  - the theorem-facing exported recursive/compressed proof boundary
- `./riscv-recursive-instantiation.md`
  - the concrete backend instantiation, canonical encodings, and API cutover
- `../../docs/superneo-paper/02_2_Technical_overview.md`
  - `Π_SuperNeo := Π_DEC ∘ Π_RLC ∘ Π_CCS`
- `../../docs/superneo-paper/07_7_Neo_s_folding_scheme_for_CCS.md`
  - the `CE(b, L)` relation and the carried `CE(b, L)^k` semantics

## 1. Goal

The RV64IM main lane is the fixed witness-backed main relation to be compiled
by later recursive or succinct proofs.

That relation owns the theorem-level meaning of:

- kernel-export binding,
- chunk-local `Π_CCS -> Π_RLC -> Π_DEC` verification,
- transcript-derived challenge binding,
- accumulator evolution,
- and terminal handoff into the published public statement.

The final external verifier shall not depend on shipped replay witnesses,
shipped native chunk transition packages, or a digest-only shell that can be
mutated and re-digested by an attacker.

The concrete Fiat-Shamir sampler used for `Π_RLC` challenges must also be
owned by the fixed relation. In particular, the backend contract may not rely
on an unbounded rejection loop that has no honest-complete fixed-size circuit
realization. The sampled strong-set element must be derived by a fixed-round
transcript map that is identically enforced in-circuit and natively.

## 2. Theorem Target

`R_main^SN` owns one exact theorem:

> the published RV64IM execution statement is valid if and only if there exists
> private witness material that realizes the same carried accumulator semantics
> as the Fiat-Shamir non-interactive form of
> `Π_SuperNeo := Π_DEC ∘ Π_RLC ∘ Π_CCS` under the concrete backend contract,
> and whose terminal carried object has theorem-level meaning equal to a valid
> carried `CE(b, L)^k` bundle.

Equivalently, `R_main^SN(stmt_pub, U_N^pub; w) = 1` iff there exist canonical
carried accumulator states `A_0..A_N` and per-chunk fresh witness material
such that:

1. `A_0` is the canonical initial carried bundle derived from the fixed RV64IM
   program binding and public initial machine state under `Init_SN`;
2. each chunk transition is exactly the Fiat-Shamir non-interactive form of
   `Π_DEC ∘ Π_RLC ∘ Π_CCS` over the canonical fresh RV64IM chunk claim and the
   previous carried bundle;
3. `A_N` has theorem-level meaning equal to a valid carried `CE(b, L)^k`
   bundle under the concrete backend contract fixed by `riscv-kernel.md`;
4. `stmt_pub` binds the canonical RV64IM public theorem statement and any
   minimal public accumulator handle `U_N^pub` required by the recursive or
   compression backend;
5. every helper digest or summary used by an implementation is recomputed from
   authoritative public/witness data inside the owned relation and is never
   treated as a second authority.

This bridge theorem is mandatory.

A conforming implementation may encode `R_main^SN` as:

- a direct circuit for Definition 13 style `CE(b, L)` membership plus the
  carried fold semantics, or
- a recursive/verifier relation whose correctness theorem explicitly proves the
  equivalence above.

A conforming implementation may also decompose the owned relation internally
into:

1. a verifier-style replay subrelation that proves the accepted
   `Π_CCS -> Π_RLC -> Π_DEC` transition under the canonical transcript-derived
   challenges; and
2. a carrier-opening subrelation that proves any explicit packed-witness
   representation used by the compiler encoding is consistent with the carried
   claims.

When such a decomposition is used, only the combined relation is theorem-facing.
The carrier-opening subrelation is an internal proof device only. It may not be
exported as a second authority, and it may not be used to weaken the replay
theorem into digest consistency or prover-chosen helper data.

In particular, the carrier-opening subrelation may project backend-specific
claim objects down to the paper `CE(b, L)` surface before opening explicit
packed witnesses. For `Π_DEC` children, replay-only backend channels such as
`s_col`, `y_zcol`, `ct`, or other convenience openings may remain owned by the
replay theorem, while the carrier-opening layer proves only the paper child
semantics `(c_i, x_i, r, {y_{i,j}}; z_i)` plus the norm/alphabet condition.
Likewise, if backend claim objects pad `y_ring` or similar rows beyond the true
ring degree `D`, that padded tail may remain replay-owned convenience structure
rather than carrier-opened theorem surface.
Such projected fields remain non-authoritative unless replay re-derives them or
the next carried transition reopens them from authoritative witnesses.

It may not replace the bridge theorem with “digest consistency”.

Direct proof of the exported `final_main_claims` alone is not sufficient unless
the implementation first proves those carried claims are themselves direct
`CE(b, L)` instances with witnesses satisfying Definition 13.

In particular, an implementation may not assume that a legacy exported
carried `CE` claims can be proved by pairing each claim with the corresponding
carried `Z` matrix and invoking a direct CE-membership circuit. If the backend
reconciles public channels such as `X` during `Π_DEC` without applying the same
correction to the carried witness matrix, then those exported claims are not
direct `CE(b, L)` witnesses and must instead be justified by the full bridge
theorem relation.

## 3. Public Instance

The public instance of `R_main^SN` shall contain only the minimal authoritative
bindings required by the theorem and by later compiler composition.

At minimum, the public instance shall bind:

- the canonical RV64IM public proof statement or its canonical digest,
- the canonical fold schedule / exact step-count binding if that metadata is
  required by the fixed relation,
- the canonical terminal carried-handle binding if required by the chosen
  backend,
- and the canonical statement-to-accumulator linkage required by the bridge
  theorem.

No digest is theorem-facing by default merely because a direct verifier
verifier path exports it.

## 4. Private Witness

The private witness shall contain the concrete theorem-bearing objects needed to
realize the main theorem above.

This includes objects such as:

- the kernel-export witness material,
- the chunk-local fresh proof/witness material needed to justify each carried
  transition,
- the carried accumulator witness material,
- and any lower-level row/family/opening objects needed to prove the owned
  theorem from the public instance.

If a value is purely derived from other witness or public data, it shall be
recomputed inside the relation rather than carried as an independent
authority.

## 5. Stable Projection Requirement

`R_main^SN` shall expose one stable authoritative projection `phi_main` for use
by later recursive or succinct compiler layers.

`phi_main` shall project only the authoritative public bindings that identify
the carried main relation instance. It shall not project convenience digests
that can be recomputed from those authoritative values.

Any later compiler layer that composes above the main relation must bind the
same `phi_main`.

## 6. Fixed-Relation Requirement

`R_main^SN` shall be specified first as a fixed public/private theorem.

Only after that may a backend compile it.

Therefore:

- the main relation shall not be defined as “verification of an arbitrary proof
  blob”;
- a backend may prove the verifier of `R_main^SN`, but it may not replace the
  owned relation with a variable-length shell surface;
- and any per-shape or padded-family backend choice must preserve the same
  public/private theorem.

## 7. Implementation Inventory Boundary

This theorem spec does not freeze any particular legacy Rust struct, proof
container, or export shell.

Any repo-local inventory mapping from legacy exported fields into:

- theorem-facing public,
- recursive-internal public,
- private witness,
- or audit-only

is owned by `riscv-recursive-instantiation.md`.

That implementation-level classification may specialize this theorem boundary,
but it may not weaken the bridge theorem in Section 2 or promote a legacy
digest shell to theorem-facing authority.

## 8. Negative Rules

The following are forbidden at this theorem boundary:

- treating digest consistency as evidence of carried `CE(b, L)^k` semantics,
- treating self-consistent digest chains as authority across trust boundaries,
- exporting replay witnesses as part of the final theorem-facing verifier API,
- carrying duplicate authorities for the same fact across the public boundary,
- and leaving the bridge theorem implicit.

## 9. Compiler Requirement

Any later recursive wrapper or succinct backend over `R_main^SN` must satisfy
all of the following:

- it compiles a fixed relation with a stable public/private interface,
- it preserves the exact bridge theorem in Section 2,
- it does not expose the private witness as theorem-facing output,
- it recomputes or canonically binds any helper digest it uses,
- and it does not weaken the carried semantics below valid carried
  `CE(b, L)^k` meaning under the concrete backend contract.
