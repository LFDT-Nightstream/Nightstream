import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.SideAnchor

/-!
Compile-time surface regression for honest side-anchor construction and
actual-type outer countermodel realization.

| Stage path | Property under test |
|---|---|
| `fprime.active.necessity.anchor.outcome` | honest NIFS completeness constructs a side anchor or exact sampler shortfall without any target outer equation |
| `fprime.active.necessity.realize` | an actual bad outer view lifts through the independent active relation |
| `fprime.active.necessity.realize.*` | each of the three outer families has a typed conditional constructor |
-/

open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity

#check HonestNifs.SemanticPremises
#check HonestNifs.Premises
#check exists_sideAnchor_or_samplerShortfall_of_semanticPremises
#check exists_sideAnchor_of_honestNifs
#check StableSideMutation
#check StableSideMutation.transport
#check ConcreteRealization
#check ConcreteRealization.lift
#check ConcreteRealization.activeIteration
#check ConcreteRealization.priorPublicLink
#check ConcreteRealization.dispatch
