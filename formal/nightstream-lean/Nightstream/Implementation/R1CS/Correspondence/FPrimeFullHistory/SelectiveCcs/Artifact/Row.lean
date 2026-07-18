import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Row.Boolean
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Row.SelectorComposition
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Row.Gating

/-!
Materialized-row bridge for the final selective CCS structure.

| Child | Owns | Emits constraints? |
|---|---|---|
| `Row.Decoder` | canonical untrusted-wire decoding | no |
| `Row.Boolean` | sparse action and Boolean-row classification | no |
| `Row.SelectorComposition` | selector-total and generic arm-gate classification | no |
| `Row.Gating` | coefficient-only selector class and exact decoded-row-to-relation action boundary | no |

This parent intentionally exports no production artifact value and grants no
row-removal authority.
-/
