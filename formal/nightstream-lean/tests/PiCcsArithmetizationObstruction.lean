import Nightstream.SuperNeo.Folding.PiCCS.ArithmetizationObstruction

/-!
Focused regression for semantic-ghost opacity in a fixed generic
`PiCCS.Attempt`.
-/

namespace NightstreamTests.PiCcsArithmetizationObstruction

open Nightstream.SuperNeo.Folding.PiCCS.ArithmetizationObstruction

example :
    ∃ (candidate :
        Nightstream.SuperNeo.Folding.PiCCS.Attempt
          Unit Unit Unit Unit Unit Nat Nat params arity)
      (openings : Fin arity.total -> Unit),
      Nightstream.SuperNeo.Folding.PiCCS.Accepted ops candidate ∧
      Nightstream.SuperNeo.Folding.PiCCS.PayloadsHold
        semantics candidate openings ∧
      Nightstream.SuperNeo.Folding.PiCCS.NormsHold
        semantics params openings ∧
      Nightstream.SuperNeo.Folding.PiCCS.AmbientOutputsHold
        semantics params candidate openings ∧
      ¬ Nightstream.SuperNeo.Folding.PiCCS.Arithmetization
        semantics params ops candidate openings :=
  accepted_payloads_norms_ambient_without_arithmetization

end NightstreamTests.PiCcsArithmetizationObstruction
