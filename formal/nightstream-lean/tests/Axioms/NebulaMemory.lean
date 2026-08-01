import Nightstream.Protocol.Nebula
import tests.Axioms.Support

/-! Fail-closed dependency guards for the Lean-owned Nebula memory model. -/

/-- info: 'Nightstream.Protocol.Nebula.Fingerprint.packed_injective' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.Fingerprint.packed_injective

/-- info: 'Nightstream.Protocol.Nebula.Fingerprint.product_perm' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.Fingerprint.product_perm

/-- info: 'Nightstream.Protocol.Nebula.Fingerprint.exact_or_collision_of_equal_product' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.Fingerprint.exact_or_collision_of_equal_product

/-- info: 'Nightstream.Protocol.Nebula.Memory.applies_product' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.Memory.applies_product

/-- info: 'Nightstream.Protocol.Nebula.Memory.executes_product' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.Memory.executes_product

/-- info: 'Nightstream.Protocol.Nebula.Memory.executes_balanced' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.Memory.executes_balanced
