# DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening

`DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening` specifies the
opening-level authority boundary for the raw public-IO prior `F'` verifier.

The verifier-visible raw public-IO checks bind:

```text
compact image replay
Construction-2 boundary replay
transcript replay
canonical proof statement
raw public vector = terminal F' public values ++ Construction-2 boundary values
```

The opening-level authority certificate separates the compressed verifier
assumption into two facts:

```text
accepted bound raw public IO opens through the fixed authority opener
the opened authority carries the same steps and image as the bound statement
```

Since the opened object is a proof-carrying folded `F'` authority, Lean derives
`FoldedFPrimeAuthority.Accepts` from those public field equalities. The module
therefore packages an opening-level raw public-IO verifier into the same
certified prior verifier consumed by the parent-only terminal theorem.

The module also exposes the direct consequences needed by callers: accepted
verification opens authority for the same `(steps, image)` pair, any concrete
authority returned by the fixed opener accepts that pair, accepted verification
reaches the claimed prior image, accepted verification exposes the prior
public-image invariants, same proof acceptance is functional, and unreachable
prior images cannot be accepted.
