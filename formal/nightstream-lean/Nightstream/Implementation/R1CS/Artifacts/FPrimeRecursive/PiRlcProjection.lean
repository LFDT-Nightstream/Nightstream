import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.BetaLadder
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.RhoEvaluations
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.YZcolIdentities

/-!
Stable artifact surface for emitted three-matrix diagnostic PiRLC projection ownership.

| Child | Physical obligation | Assurance |
|---|---|---|
| `BetaLadder` | one 55-power, 272-row shared ladder | artifact-checked |
| `RhoEvaluations` | 15 ordered 54-coefficient evaluations, 1,620 source rows | artifact-checked |
| `YZcolIdentities` | both complete degree-106 identities, reusing the shared ladder, rho evaluations, and output evaluations | artifact-checked; large closed-list facts use `native_decide` and have focused trust-surface guards |
-/
