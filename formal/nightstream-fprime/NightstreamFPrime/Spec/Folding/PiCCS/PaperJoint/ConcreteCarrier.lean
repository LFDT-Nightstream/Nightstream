import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.NoZeroDivisors
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolDataRefinement

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/PaperJoint/ConcreteCarrier.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Concrete carrier closure for canonical SuperNeo v1.1 `Pi_CCS`.

Protocol: SuperNeo `Pi_CCS` (Section 7.3 / Appendix D.4).
Phase: concrete embedding semantics.
Constraint family: protocol-level semantic refinement only; this file emits
no rows.

Owns: protocol-level zero reflection, strict-norm compatibility of `K.embed`,
and the concrete `ProtocolDataRefinement.ProtocolLift`.

Does not own: base/extension operations or algebra laws, sparse-polynomial
evaluation laws, proofs of the modulus-level Euclid and seven-nonresidue
properties, proof that Rust field arithmetic refines these definitions,
coefficient-expanded matrix derivation, transcript hashing, SumCheck degree
bounds, output CE projection, Pi_RLC handoff, R1CS, row removal, or counts.

Emits constraints: no.

Authority boundary: arithmetic and the zero/one/add/mul embedding laws come
from `ConcreteCarrier.Algebra`. This file adds only the norm-aware protocol
obligations and verifier composition. The still-unproved Goldilocks modulus
Euclid property remains an explicit premise rather than being hidden inside
the carrier instance.

| Protocol | Phase | Family | Concrete owner / result |
|---|---|---|---|
| imported algebra | carrier arithmetic | base/extension operations and laws | `ConcreteCarrier.Algebra` |
| imported cancellation | extension irreducibility | conditional no-zero-divisor derivation | `ConcreteCarrier.NoZeroDivisors` |
| `Pi_CCS` | carrier placement | zero reflection | `zeroReflectingLift` |
| `Pi_CCS` | norm placement | strict cubic commutes with `K.embed` | `embed_strictNorm` |
| `Pi_CCS` | carrier placement | full norm-aware `F -> K` contract | `protocolLift` |
| open arithmetic | number-theoretic instantiation | modulus Euclid and seven nonresidue | explicit theorem premises |
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

private theorem k_embed_strictNorm (value : F) :
    K.mul (K.mul (K.add (K.embed value) K.one) (K.embed value))
        (K.sub (K.embed value) K.one) =
      K.embed (NormRange.cubicResidual value) := by
  rw [show K.one = K.embed 1 from rfl]
  exact NormRange.embed_cubicResidual value

/-- The concrete embedding preserves and reflects the semantic zero. -/
def zeroReflectingLift :
    ConcreteJointData.ZeroReflectingLift baseOps extensionOps K.embed where
  zero_iff := by
    intro value
    constructor
    · intro equal
      have component := congrArg K.c0 equal
      simpa only [K.embed, K.zero, baseOps, extensionOps] using component
    · intro equal
      subst equal
      rfl

/-- Applying the extension-carrier norm polynomial after embedding is exactly
embedding the independent concrete base cubic. -/
theorem embed_strictNorm (value : F) :
    ProtocolPolynomial.strictNormResidual extensionOps (K.embed value) =
      K.embed (NormRange.cubicResidual value) := by
  unfold ProtocolPolynomial.strictNormResidual
  rw [derived_sub_eq_concrete_sub]
  exact k_embed_strictNorm value

/-- Concrete `F -> K` placement assembled solely from the named leaf
theorems above. -/
def protocolLift :
    ProtocolDataRefinement.ProtocolLift baseOps extensionOps K.embed where
  toZeroReflectingLift := zeroReflectingLift
  map_one := embed_one
  map_add := embed_add
  map_mul := embed_mul
  map_strictNorm := embed_strictNorm

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
