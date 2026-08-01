import Nightstream.Implementation.Lowering.Nebula.BitSemantics
import tests.Axioms.Support

/-! Fail-closed dependency guards for canonical Nebula bit decoding. -/

/-- info: 'Nightstream.Implementation.Lowering.Nebula.BitSemantics.isBit_iff_zero_or_one' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.BitSemantics.isBit_iff_zero_or_one

/-- info: 'Nightstream.Implementation.Lowering.Nebula.BitSemantics.eval_word_val_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.BitSemantics.eval_word_val_exact
