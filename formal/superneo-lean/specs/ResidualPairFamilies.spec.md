# Residual pair families

## Purpose

`ResidualPairFamilies` specializes the protocol-neutral adjacent pair and
odd-tail schedule to two constraint families while preserving the arbitrary
Goldilocks residual theorem underneath.

The module is mathematical accounting only. It neither describes a generated
matrix nor authorizes replacement of production rows.

## Algebraic contract

For arbitrary `r1 r2 : F`, define:

```text
pair(r1, r2) = r1^2 - 7*r2^2.
```

Seven is a Goldilocks quadratic nonresidue, so:

```text
pair(r1, r2) = 0  iff  r1 = 0 and r2 = 0.
```

The two concrete residual families are:

| Family | Residual | Exact leaf semantics | Pair degree | Selector-gated pair degree |
|---|---|---|---:|---:|
| one-product R1CS | `A*B-C` | `A*B=C` | 4 | 5 |
| centered unit | `d^3-d` | `d` is exactly `-1`, `0`, or `1` | 6 | 7 |

An odd tail retains the ordinary residual. Its selector-gated degrees are
three and four respectively. Selector equivalence applies only when the
verifier fixes the selector to one.

## Structural schedule

The component reuses `BooleanPairRows.schedule`; it does not define another
row ordering. For an ordered family-local input list of length `n`:

- adjacent inputs become `n / 2` pair rows;
- one final input becomes `n % 2` odd-tail rows;
- the total is `(n + 1) / 2` rows;
- flattening row inputs recovers the original list in exact order.

Separate explicit witnesses for each concrete family show that deleting its
pair row or odd-tail row admits a nonzero residual.

## Non-goals and authority boundary

The theorems do not prove:

- which production columns belong to either family;
- that pairing may cross a protocol stage or constraint-family boundary;
- that a generated row has the stated polynomial;
- that a selector is verifier-fixed to one;
- global gate minimality or an information-theoretic gate lower bound;
- that any Rust row can be removed.

A production integration must replay the generated row/matrix expressions,
column order, stage/family reset points, selector binding, and odd-tail census
into these theorems.

## Required theorem surface

| Theorem | Guarantee |
|---|---|
| `residualPairHolds_iff` | arbitrary nonresidue pair exactness |
| `oneProductPairHolds_iff` | exact two-equation R1CS specialization |
| `selectorGatedOneProductPairHolds_iff` | selector-one R1CS equivalence |
| `oneProduct_selectorGatedDegree_le_five` | all pair/tail rows stay within degree five |
| `centeredUnitResidual_eq_zero_iff` | cubic residual has exactly the centered roots |
| `centeredUnitPairHolds_iff` | exact two-digit centered specialization |
| `selectorGatedCenteredUnitPairHolds_iff` | selector-one centered equivalence |
| `centeredUnit_selectorGatedDegree_le_seven` | all pair/tail rows stay within degree seven |
| `familySchedule_order_exact` | exact deterministic input order |
| `familySchedule_shape_counts` | exact pair/tail/total census |
| `oneProduct_pairRow_is_necessary` / `oneProduct_oddTailRow_is_necessary` | R1CS deletion witnesses |
| `centeredUnit_pairRow_is_necessary` / `centeredUnit_oddTailRow_is_necessary` | centered-unit deletion witnesses |
