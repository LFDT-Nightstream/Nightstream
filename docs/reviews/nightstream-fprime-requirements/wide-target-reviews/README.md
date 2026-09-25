# Wide-key target reviews

Date: 2026-09-25. Reviewed commit: `9e9cd23d1`. Its Lean sources are those of
`bab9b858`, which moved the security chain to the wide key.

Four independent review agents (Claude, fresh context, read-only) reviewed the
four changed evidence targets. Each file here holds one proposal and one
response. The local tool imports are diagnostic; accepted closure needs the
controlled review process.

| Obligation | Target meaning | Decomposition (five checks) | Tool record |
|---|---|---|---|
| `stage1-terminal-assignment` | pass | pass | request `8477d1ba…`, record `e81af56d…` |
| `stage1-terminal-parent` | pass | pass | request `6257629a…`, record `59ef1ee0…` |
| `hypernova-linear-security` | pass | pass | none (capture defect below) |
| `hypernova-terminal-false-acceptance` | pass | pass | none (capture defect below) |

## Findings and dispositions

- **Minor, all four reviews:** the headers of `Wide/Relation.lean`,
  `PiRLC/Wide/Key.lean`, `Wide/FixedPoint.lean` and `Wide/SetupBinding.lean`
  called the wide relation and key "candidates" that production had not
  selected. Fixed (comments only). The captured source changed, so the two
  tool records are stale for the new cut; no statement or proof changed.
- **Minor:** `scripts/lean_graph/HYPERNOVA_LINEAR_SECURITY.md` named a
  nonexistent `history_probability_bound` and square-root theorem. Removed.
  Its FS premise wording now names the target's relation and wide key.
- **Tool defect:** `evidence.py review-request` cannot capture the
  `security-validation` gate. Its `rust` source group contains the symlink
  `crates/nightstream/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json`,
  whose target is excluded from the `lean` group. The symlink dates from
  2026-09-19. The fix is a change to `scripts/lean_graph/obligations.json`,
  which the owner controls.
- **Notes, no change:**
  - `Wide.selected` has no `AuthorityStream.prepare compiled = .ok parts`
    premise, and no theorem uses it. Each target covers every wide target;
    the prepared-parts link belongs to `P.binding.context`.
  - No Lean theorem proves that `RangePlan.compile?` succeeds or that a
    recursive `target.Holds` can hold. Emitter and test runs show both.
  - `Terminal.HoldsFor` builds the narrow `Lifecycle.setup`. The terminal
    transition reads only its verifier keys, so this has no effect.
  - The interactive challenge instances name `ProductionKey.key`. The wide
    key shares its challenge set; only `piRlcResponse` differs.
  - "No valid history" counts advice of any message length, as before.
  - The `argument` text of `hypernova-linear-security` in `obligations.json`
    omits `WideFiatShamir` from the chain.
  - `STAGE1_BASELINE.md` lines 500-501, `STAGE1_BASELINE_STATUS.md` and
    `FALSE_ACCEPTANCE_REVIEW.json` still cite `8084c256` and the narrow event.
