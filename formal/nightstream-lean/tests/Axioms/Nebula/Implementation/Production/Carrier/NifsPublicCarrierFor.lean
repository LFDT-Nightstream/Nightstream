import Nightstream.Implementation.Nebula.Production.Carrier.NifsPublicCarrierFor
import tests.Axioms.Support

/-! Dependency audit for the exponent-indexed carrier-to-PiCCS bridge. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionFullClaimNifsPublicCarrierFor.fields_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionFullClaimNifsPublicCarrierFor.fields_length

/-- info: 'Nightstream.Implementation.Nebula.ProductionFullClaimNifsPublicCarrierFor.fieldValues_eq_publicNifsFields' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionFullClaimNifsPublicCarrierFor.fieldValues_eq_publicNifsFields

/-- info: 'Nightstream.Implementation.Nebula.ProductionFullClaimNifsPublicCarrierFor.piCcsPlacement' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionFullClaimNifsPublicCarrierFor.piCcsPlacement
