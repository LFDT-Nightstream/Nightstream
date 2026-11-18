# Claims Layer

**Status:** Accepted

## Problem

The Lean package has about 7,500 theorems. A reader could not see which of
them state what Nightstream proves. The main results had hypotheses in
`variable` blocks, and the list of main results was a hand-written list of
names. A summary in a document can differ from the Lean.

## SuperNeo

SuperNeo v1.2 fixes the protocol and its theorems. It does not fix how an
implementation states its machine-checked results.

## Decision

The owner approved this on 2026-11-15.

1. **One namespace.** Nightstream claims exactly the theorems of
   `NightstreamFPrime.Claims`. Each other theorem is a lemma.
2. **A claim is a restated theorem.** It has every hypothesis in its
   signature, and its proof is one application of the theorem that proves it.
   The Lean kernel checks the claim against that theorem.
3. **Inclusion.** A theorem is a claim when its hypotheses are about an
   emitted package, a verifier entry point, or a protocol relation of the
   paper. Soundness and completeness of one protocol part are in one file.
4. **Checks.** `tests/PremiseCensus.lean` reads the namespace. Each
   hypothesis of each claim has one class in the premise table of
   [SECURITY_MODEL.md](../formal/nightstream-fprime/SECURITY_MODEL.md).
   [SPEC.md](../SPEC.md) section 9 gives a claim for each verifier entry point.
