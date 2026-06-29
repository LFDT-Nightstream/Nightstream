# Direct Parent-Only Production Private DEC No-Swap Audit

This component exposes the certificate-level private DEC anti-substitution
facts from the production parent-only terminal package.

## Inputs

- Section 7.1-backed production context.
- Final non-aggregate private DEC/stage facts, or the final end-to-end package.
- An alternate private child table that satisfies the full pointwise private
  DEC requirements for the same parent source.

## Guarantees

The projection returns a `PrivateDecNoSwapAudit` containing the equalities that
justify the parent-only `CE(B)` handle:

```text
opened parent residues agree
private DEC digit tables agree pointwise
CE witness-derived digit tables agree pointwise
child-bundle next Pi_CCS wires agree
requested next Pi_CCS input tables agree
```

The alternate child table must satisfy the same full pointwise private DEC
requirements as the audited table. Aggregate child summaries are not accepted
as a substitute for those requirements.

## Boundary Assumptions

The projection uses the production context's Poseidon2 parent-hash binding
object and MSIS-to-Ajtai binding assumptions. It does not implement Poseidon2
or introduce a new hash family.
