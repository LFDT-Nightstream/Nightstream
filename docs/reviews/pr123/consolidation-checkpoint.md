# PR123 consolidation checkpoint

September 26, 2026. Starting source: `e1a7c96a617237005859fd6a01ae3c21ba0727da`.
Branch: `nico/pirlc-wide-sampler-integration`.

**Unfinished checkpoint. Do not merge or treat the existing fixtures and green
CI at `e1a7c96a6` as validation of this source.** The owner requested this commit
to preserve the work before a clean replacement from
`nico/f-prime-constraints-cuda-formal`.

## Changes preserved

- `ProductionKey.key` directly uses the total wide sampler. The separate Wide
  key override and duplicate lifecycle relation were removed.
- Shared PiCCS boundary facts, physical prefix bounds, and PiDEC verifier facts
  were separated from obsolete bounded-sampler completeness consumers.
- PiDEC proof inputs load directly at the selected offsets. The old translated
  loader was removed. Combination bounds moved to their mathematical owner.
- `HyperNovaCompleteness.recursive_nifs` constructs an accepted fold and exact
  child openings from accepted terminal openings, without a sampler-success
  premise. `HyperNovaStepData` now uses the canonical key and caller's relation.
- `HyperNovaAcceptedNext.lean` contains proposed recursive and base successor
  proofs for the selected package. **These proofs have not passed compilation.**
- The separately supplied `PR123_CURRENT_REVIEW.md` is preserved unchanged. It
  reviews `e1a7c96a6`, not this checkpoint, and is not a signed acceptance.

No Rust source, selected artifact, verifier pin, protocol parameter, protected
owner file, or independent acceptance record changed in this checkpoint.

## Validation and known breakage

The following focused builds passed with pinned Lean `v4.32.2` through
`scripts/validate.sh`:

- Canonical production key and its PiRLC circuit agreement.
- Shared accumulator, PiCCS boundary, and PiDEC verifier facts.
- Selected emitter after the initial key consolidation (15 seconds).
- Direct PiDEC loader (3 seconds) and input bounds (36 seconds).
- Honest NIFS completeness (5 seconds) and semantic step data (2 seconds).

These results cover the named intermediate source states, not the complete
checkpoint. The final accepted-successor build failed in
`Layout/Stage1/Wide/PhysicalPrefixCompleteness.lean` because
`PilotNifsCompleteness.pilot_prefix` was no longer imported. Its needed pilot
constructor must be separated from the obsolete full C/R/D constructor. The
build was then stopped at the owner's strategy review; it exited 143 after
243 seconds. The accepted-successor module was not checked successfully.

The complete library, evidence roots, and axiom targets were not rebuilt.
Several old roots and audits still refer to removed declarations. The old
sampler/layout dependencies and stale evidence targets remain. No complete
conformance or artifact-promotion claim is made.

Temporary logs are in `/tmp/nightstream-pr123-consolidation/` on the original
computer. They are not required to understand this checkpoint.

## Replacement direction

Use the actual base branch, not `main` or the detached cleanup history. Its
observed head is `7f51e1010ce382d15206d4d1fabcb27d88754cfe`; fetch and verify it.
Keep `formal/nightstream-fprime` authoritative for this integration.

Reuse the independently checkable sampler specification, gadget proofs, Rust
decoder, and relevant regression checks. Replace the original key and layout
in place, preserve soundness and accepted-successor completeness, and derive
one canonical emitted package. Do not carry over the Wide layout overlay or
optional constraint reductions as an integration prerequisite.

Commit `6623c49e8` contains a Phi81 low-norm invertibility proof in the same
formal project. Port and audit the proof against the chosen base; do not merge
its entire review branch. Keep the approved Fiat-Shamir and fixed-seed MSIS
boundaries explicit. A standalone sampler bias result is not a numerical
security bound for the complete concrete verifier.

Retire obsolete consumers and fix current evidence targets. Preserve complete
matrix, assignment, nonzero-fold, proof-byte, and mutation checks. Regenerate
artifacts and pins only after the applicable exact checks. Keep Lean local as
the owner requested and bind validation records to the final source. Refresh
independent review requests without creating the author's own acceptance.
