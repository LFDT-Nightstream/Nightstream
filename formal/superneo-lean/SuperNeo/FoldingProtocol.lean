import SuperNeo.ProofSystem.ConstraintSystem
import SuperNeo.ProofSystem.SumCheck
import SuperNeo.ProofSystem.Folding
import SuperNeo.FoldingProtocol.ProtocolRelations
import SuperNeo.FoldingProtocol.ProtocolSection71Data
import SuperNeo.FoldingProtocol.ProtocolSection71Context
import SuperNeo.FoldingProtocol.PiCCS
import SuperNeo.FoldingProtocol.PiRLC
import SuperNeo.FoldingProtocol.PiDEC
import SuperNeo.FoldingProtocol.ArithmeticBundle
import SuperNeo.FoldingProtocol.ArithmeticObligations
import SuperNeo.FoldingProtocol.ProtocolTarget
import SuperNeo.FoldingProtocol.ProtocolTargetData
import SuperNeo.FoldingProtocol.ProtocolMathTarget
import SuperNeo.FoldingProtocol.ProtocolTheorem
import SuperNeo.ProofSystem.Protocol
import SuperNeo.FoldingProtocol.FiatShamirReroute

/-! Section 7 (Neo's folding scheme for CCS) barrel:
    CCS relations, Π_CCS, Π_RLC, Π_DEC, arithmetic obligations, protocol theorem. -/
