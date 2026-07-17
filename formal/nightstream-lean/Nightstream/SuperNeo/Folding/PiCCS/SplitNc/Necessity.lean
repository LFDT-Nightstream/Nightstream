import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity.CarrierCoverage
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity.YZcolTerminal

/-!
Necessity results for the independent Phi81 SplitNc semantic model.

Owns: the protocol-to-phase index for counterexamples and impossibility
results showing why candidate recursive-verifier obligations cannot be
removed or weakened.

Does not own: semantic truth definitions, executable refinement, production
bug claims, R1CS rows, row removal, or constraint counts.

Emits constraints: no.

| Child | Mathematical obligation | Emits constraints? | Lean owner |
|---|---|---|---|
| `CarrierCoverage` | a logical-width cube omits completed-carrier coordinates | no | `Necessity.CarrierCoverage` |
| `YZcolTerminal` | scalar terminal equality is necessary but cannot replace source binding | no | `Necessity.YZcolTerminal` |
-/
