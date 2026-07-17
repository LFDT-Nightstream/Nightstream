import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.RunningAuthority

/-!
Compile-time surface regression for the model-level fixed-active checked
running-authority baseline.

| Stage path | Property under test |
|---|---|
| `fprime.active.honest_baseline.running.pi_dec` | a valid combined opening and exact canonical children imply checked running authority |
-/

open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline

#check RunningAuthority.accepted_of_combinedOpening
