import CompressionLean.Spec.Program
import CompressionLean.Spec.Messages
import CompressionLean.Spec.Openings
import CompressionLean.Spec.Replay
import CompressionLean.Layer1.Soundness
import CompressionLean.Shrink.Ccs
import CompressionLean.Ports.Shrink.Game
import CompressionLean.Game.Path
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFPolynomial
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.NoZeroDivisors
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetConvention
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.BaseLinear

namespace CompressionLean.Spec

open CompressionLean.Circuit CompressionLean.Terminal.Package
open CompressionLean.Layer1
open NightstreamFPrime.Spec (goldilocksModulus)
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFPolynomial (modulus)

noncomputable abbrev game (key : Layer1Key) (S : Setup Gl) (L0 L1 : ℕ) :=
  layer1Game (shapeOf key.relation) modulus barOf S L0 L1 (2 ^ (key.relation.setupVars - 2)) goldilocksModulus
    (queryAt (K := Cubic) key.relation)

end CompressionLean.Spec
