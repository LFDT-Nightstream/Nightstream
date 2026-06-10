# Formal (Lean) Subprojects

Five standalone Lean 4 projects live under `formal/`, deliberately outside the Rust
workspace. Lean-specific build/maintenance instructions live in each subproject's
`AGENTS.md` / `README.md`.

| Project | Role | Status |
|---|---|---|
| `superneo-lean/` | The theorem-facing model of core SuperNeo math: ring/field/norm primitives, Definition 7/8 embeddings, Theorem 4/5 evaluation homomorphism, strong-sampling sets, Π_CCS/Π_RLC/Π_DEC protocol relations. Module structure mirrors paper §4–§7 via barrel files (Primitives, EmbeddingTheory, SecurityModel, FoldingProtocol). | **Authoritative** — Lean is the mathematical source of truth for these surfaces. Maintenance boundary is Lean-only; Rust-generated conformance vectors are currently out of the maintained build path. |
| `direct-ccs-fprime-lean/` | Direct-CCS F′ protocol-boundary checks; first module proves the `DecAuthorization` wiring theorem. | Active, narrow scope |
| `opening-convergence-lean/` | Soundness of the opening-convergence pipeline: reducing ~600 family-level evaluation claims to 6 final Ajtai PCS openings. | Standalone |
| `twist-shout-lean/` | Paper-faithful formalization of Setty–Thaler Twist/Shout (memory arguments), as a standalone mathematical artifact — not a SuperNeo specialization. | Standalone; relevant to the Nebula memory-checking roadmap |
| `nightstream-lean/` | Composition layer above superneo-lean and twist-shout-lean for the published-proof boundary. | Prototype-era, parked |

## How the formal work relates to the Rust code

- The Rust crates' `specs/*.spec.md` files state MUST/SHOULD contracts; the Lean
  projects prove the mathematical content behind the load-bearing ones (bar-transform
  identities, split_b round-trips, evaluation homomorphism, sampling-set bounds).
- When Rust behavior and a Lean theorem disagree, the Lean statement wins — fix the
  code or the spec, not the theorem.
- For the SuperNeo project specifically, read `formal/superneo-lean/AGENTS.md` before
  touching anything; it defines the spec/interface/implementation layout and the
  closure standard.
