# Duplex-Sponge Fiat–Shamir Assumption

**Status:** Accepted

## Problem

The knowledge results of the two folds are theorems of a random-oracle model.
The verifier reads its challenges from the Poseidon2 sponge, so a premise must
move the failure event to the ideal-permutation model. The event includes the
retries of the extractor. Sponge indifferentiability (`[SPONGE]`) moves one
run only, so the premise was an open row of the premise table. That row
asserted the error `N²/q⁴` and let the model adversary have a query bound that
the premise did not fix. No published proof gives that error or that bound.

## SuperNeo

SuperNeo v1.2 proves its reductions as interactive protocols. It does not fix
the Fiat–Shamir transformation or its model.

## Decision

The owner approved the assumption `DSFS` on 2026-11-18.

1. **One named assumption.** The premise of `Claims.Fold.light_knowledge` and
   `Claims.Fold.refresh_knowledge` is `Spec.Sponge.DuplexTransfer`. It has the
   shape of the lemma of Chiesa and Orrù, ePrint 2025/536, Lemma 5.1: a prover
   with `N` permutation calls has a model prover with at most `N` queries, and
   the event's probability is at most `Spec.Sponge.duplexError` larger. The
   error `25·N²/q⁴` is the error of ePrint 2025/536, Theorem 6.2, at a codec
   bias of 0.
2. **The parts without a proof are in the assumption.** The paper proves the
   overwrite mode, and its rewinding definitions are for deterministic
   provers. Nightstream's sponge adds each block, and the model prover of the
   lemma samples values. Its start from the zero state, which absorbs the
   fold's domain and statement first, is a case that ePrint 2025/536,
   Section 2.2, says the analysis covers. The entry `DSFS` of
   [SECURITY_MODEL.md](../formal/nightstream-fprime/SECURITY_MODEL.md) states
   both parts. A proof of a part removes it from the entry.
3. **The conditions stay Lean theorems.** The conditions of the premise are
   `Export.BoundedDepth.FoldSponge.Conditions`, which
   `Export.BoundedDepth.FoldSponge.conditions` proves for each fold schedule.
4. **`SPONGE` keeps the single-run uses.** The collision finders and the
   other uses of `Spec.Sponge.Transfer` are one run with one output, so they
   stay under `[SPONGE]`.
