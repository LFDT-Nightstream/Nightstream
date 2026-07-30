import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsAllocationCoverage
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.ConcreteNifsAllocationCoverage

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFixedPhaseEndpointCoverage.chain' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KFixedPhaseEndpointCoverage.chain

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcClaimedEndpointCoverage.rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcClaimedEndpointCoverage.rows

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerAllocation.allocation_mem_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalSamplerAllocation.allocation_mem_iff

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionAudit.dense_column_mem' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPiRlcActionAudit.dense_column_mem

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionAudit.columns_written' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPiRlcActionAudit.columns_written

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsAllocationCoverage.endpointColumns_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsAllocationCoverage.endpointColumns_eq

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsAllocationCoverage.operationalEndpoints' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsAllocationCoverage.operationalEndpoints

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsAllocationCoverage.operationalSampler' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsAllocationCoverage.operationalSampler

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsAllocationCoverage.operationalSamplerAllocation_mem_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsAllocationCoverage.operationalSamplerAllocation_mem_iff

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsAllocationCoverage.allocation_used' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsAllocationCoverage.allocation_used

end NightstreamTests.Axioms.ConcreteNifsAllocationCoverage
