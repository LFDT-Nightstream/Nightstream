# Direct CCS F' Lean

This Lean project contains the direct CCS `F'` protocol-boundary checks. It is
separate from `formal/superneo-lean`, but imports SuperNeo theorem surfaces where
the direct path reuses the paper-level protocol facts.

## Layout

The tree follows the ArkLib reading style:

```text
DirectCcsFPrime/
  Core/                         construction-2 and folded F' authority basics
  Commitment/Parent/Spec/       parent-source, encoding, and digest statement shapes
  Commitment/Parent/Impl/       implementation-facing parent hash and Ajtai adapters
  Commitment/Parent/Security/   parent binding and opening consequences
  ProofSystem/PrivatePiDec/     private DEC specs, concrete verifier models, soundness
  ProofSystem/Stage/            direct stage semantics and SuperNeo stage reuse
  ProofSystem/Terminal/         terminal accumulator and no-swap theorem surfaces
  ProofSystem/Production/       production verifier, public-IO, replay, endpoint layers
  Bridge/                       adapters into `formal/superneo-lean`
  Audit/                        counterexamples, necessity lemmas, red-team surfaces
```

Each protocol subtree is organized by responsibility:

```text
Spec/      protocol objects, relations, verifier predicates
Impl/      concrete verifier/runtime models and implementation-shaped adapters
Security/  soundness, binding, uniqueness, replay, endpoint consequences
```

`DirectCcsFPrime.lean` is intentionally only a conceptual barrel:

```lean
import DirectCcsFPrime.Core
import DirectCcsFPrime.ProofSystem
import DirectCcsFPrime.Commitment
import DirectCcsFPrime.Bridge
import DirectCcsFPrime.Audit
```

## Reader Path

For the theorem story, read in this order:

1. `Core/`: construction-2 public images, folded F' authority, and compressed
   prior-authority boundaries.
2. `ProofSystem/PrivatePiDec/`: pointwise private DEC authorization and why
   aggregate summaries are insufficient.
3. `Commitment/Parent/`: parent source encoding, digest binding, Poseidon2 parent
   hash boundary, and Ajtai-backed residue opening.
4. `ProofSystem/Stage/`: direct `Pi_CCS -> Pi_RLC` stage computation and SuperNeo
   theorem reuse.
5. `ProofSystem/Terminal/`: terminal accumulator soundness and hidden-child
   no-swap consequences.
6. `ProofSystem/Production/`: runtime prior-verifier, raw/exact public IO,
   replay-stable verifier certification, and final endpoint packages.

## Specs

Markdown specs live under `specs/` with the same responsibility layout as the
Lean tree. They are supporting design notes; the Lean modules are the
authoritative proof surface.

## Build

```bash
cd formal/direct-ccs-fprime-lean
lake build
```

The Lake library glob is recursive, so nested modules are built through the same
`DirectCcsFPrime` library target.
