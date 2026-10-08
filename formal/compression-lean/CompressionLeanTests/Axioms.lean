import tests.AxiomAudit
import CompressionLean

/-! Axiom gate for every exported terminal theorem. -/

#audit_axioms CompressionLean.Terminal.circuitPackage_implies_terminal
#audit_axioms CompressionLean.Terminal.priorSpec
#audit_axioms CompressionLean.Terminal.piCcsPhaseHolds
#audit_axioms CompressionLean.Terminal.piCcsSpecHolds
#audit_axioms CompressionLean.Terminal.selectedVerifierContext
#audit_axioms CompressionLean.Terminal.piRlcPhaseHolds
#audit_axioms CompressionLean.Terminal.piRlcSpecHolds
#audit_axioms CompressionLean.Terminal.piRlcCombinationRows
#audit_axioms CompressionLean.Terminal.piRlcSamplerChain
#audit_axioms CompressionLean.Terminal.priorExtraRows_hold
#audit_axioms CompressionLean.Terminal.arithmeticRows_hold
#audit_axioms CompressionLean.Terminal.samplerInitialState_eq
#audit_axioms CompressionLean.Terminal.commitment_eval
#audit_axioms CompressionLean.Terminal.publicInput_eval
#audit_axioms CompressionLean.Terminal.point_eval
#audit_axioms CompressionLean.Terminal.eval_K_eval
#audit_axioms CompressionLean.Terminal.eval_A_eval
#audit_axioms CompressionLean.Terminal.state_eval
