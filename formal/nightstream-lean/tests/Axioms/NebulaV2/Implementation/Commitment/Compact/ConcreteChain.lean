import Nightstream.Implementation.NebulaV2.Commitment.Compact.ConcreteChain
import tests.Axioms.Support

/-! Fail-closed dependency audit for the concrete compact-chain hash. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ConcreteCompactChain.toFrame_injective' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ConcreteCompactChain.toFrame_injective

/-- info: 'Nightstream.Implementation.NebulaV2.ConcreteCompactChain.encodedFrame_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ConcreteCompactChain.encodedFrame_injective

/-- info: 'Nightstream.Implementation.NebulaV2.ConcreteCompactChain.injective_or_named_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ConcreteCompactChain.injective_or_named_collision
