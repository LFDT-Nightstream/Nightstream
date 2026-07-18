import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcProjection.BetaLadder
import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcProjection.RhoEvaluations
import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcProjection.YZcolIdentities
import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcProjection.YZcolNormalForm

/-!
Stable correspondence surface for emitted three-matrix diagnostic PiRLC projection rows.

| Child | Semantic result | Remaining boundary |
|---|---|---|
| `BetaLadder` | exact rows force the 55 advertised powers and the 54-power `y_zcol` prefix | beta transcript derivation and whole-row embedding |
| `RhoEvaluations` | exact rows force all 15 physical outputs to evaluate their exact coefficient columns at the ladder's beta wire | rho transcript/semantic authority and whole-row embedding |
| `YZcolIdentities` | four exact source-row premises imply both complete identities are batch-accepted, hence exact or at a named bad root | transcript/column authority, whole-row embedding, and bad-root probability; native-compiled artifact facts have focused trust-surface guards |
| `YZcolNormalForm` | exact limb identities decode to one typed `RingK` source aggregate; bound columns give parent equality or a named bad root | transcript/opening derivation of the explicit column-binding premise and bad-root probability |
-/
