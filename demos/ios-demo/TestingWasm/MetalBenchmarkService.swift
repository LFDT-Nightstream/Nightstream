import Foundation
import Combine

#if canImport(NeoMetalBench)
import NeoMetalBench
#endif

struct MetalBenchmarkReport: Decodable {
    struct Device: Decodable {
        let os: String
        let arch: String
        let gpuName: String
        let unifiedMemory: Bool
        let recommendedWorkingSetBytes: UInt64

        enum CodingKeys: String, CodingKey {
            case os, arch
            case gpuName = "gpu_name"
            case unifiedMemory = "unified_memory"
            case recommendedWorkingSetBytes = "recommended_working_set_bytes"
        }
    }

    struct Timing: Decodable {
        let samples: Int
        let medianMs: Double
        let minMs: Double
        let maxMs: Double
        let p95Ms: Double
        let coefficientOfVariation: Double

        enum CodingKeys: String, CodingKey {
            case samples
            case medianMs = "median_ms"
            case minMs = "min_ms"
            case maxMs = "max_ms"
            case p95Ms = "p95_ms"
            case coefficientOfVariation = "coefficient_of_variation"
        }
    }

    struct Primitive: Decodable, Identifiable {
        let name: String
        let workItems: Int
        let parityOk: Bool
        let crossoverRequired: Bool
        let cpu: Timing
        let candidates: [Candidate]
        let selectedCandidate: String
        let selectedSpeedupOverCpu: Double
        let crossoverGatePassed: Bool

        var id: String { name }

        enum CodingKeys: String, CodingKey {
            case name, cpu, candidates
            case workItems = "work_items"
            case parityOk = "parity_ok"
            case crossoverRequired = "crossover_required"
            case selectedCandidate = "selected_candidate"
            case selectedSpeedupOverCpu = "selected_speedup_over_cpu"
            case crossoverGatePassed = "crossover_gate_passed"
        }
    }

    struct Candidate: Decodable, Identifiable {
        let name: String
        let setupMs: Double
        let timing: Timing
        let speedupOverCpu: Double

        var id: String { name }

        enum CodingKeys: String, CodingKey {
            case name, timing
            case setupMs = "setup_ms"
            case speedupOverCpu = "speedup_over_cpu"
        }
    }

    struct Lifecycle: Decodable, Identifiable {
        let name: String
        let backend: String
        let verificationMode: String
        let synthesisMs: Double
        let preprocessingMs: Double
        let online: Timing
        let pipeline: Pipeline?
        let verifyMs: Timing
        let nifsProfile: NifsProfile?
        let semanticResultOk: Bool
        let proofParityOk: Bool

        var id: String { "\(name):\(backend)" }

        enum CodingKeys: String, CodingKey {
            case name, backend, online, pipeline
            case verificationMode = "verification_mode"
            case synthesisMs = "synthesis_ms"
            case preprocessingMs = "preprocessing_ms"
            case verifyMs = "verify_ms"
            case nifsProfile = "nifs_profile"
            case semanticResultOk = "semantic_result_ok"
            case proofParityOk = "proof_parity_ok"
        }
    }

    struct Pipeline: Decodable {
        let synthesisWork: Timing
        let foldWork: Timing
        let finalMaterialization: Timing
        let overlapSaved: Timing

        enum CodingKeys: String, CodingKey {
            case synthesisWork = "synthesis_work"
            case foldWork = "fold_work"
            case finalMaterialization = "final_materialization"
            case overlapSaved = "overlap_saved"
        }
    }

    struct LifecycleCrossover: Decodable, Identifiable {
        let name: String
        let crossoverRequired: Bool
        let medianSpeedupOverCpu: Double
        let p95SpeedupOverCpu: Double
        let proofParityOk: Bool
        let passed: Bool

        var id: String { name }

        enum CodingKeys: String, CodingKey {
            case name, passed
            case crossoverRequired = "crossover_required"
            case medianSpeedupOverCpu = "median_speedup_over_cpu"
            case p95SpeedupOverCpu = "p95_speedup_over_cpu"
            case proofParityOk = "proof_parity_ok"
        }
    }

    struct Sustained: Decodable {
        let secondsPerBackend: Int
        let cpuProofsPerSecond: Double
        let metalProofsPerSecond: Double
        let speedupOverCpu: Double
        let proofParityOk: Bool
        let passed: Bool

        enum CodingKeys: String, CodingKey {
            case passed
            case secondsPerBackend = "seconds_per_backend"
            case cpuProofsPerSecond = "cpu_proofs_per_second"
            case metalProofsPerSecond = "metal_proofs_per_second"
            case speedupOverCpu = "speedup_over_cpu"
            case proofParityOk = "proof_parity_ok"
        }
    }

    struct NifsProfile: Decodable {
        let foldsPerSample: Int
        let total: Timing
        let piCcs: Timing
        let ajtaiYEval: Timing
        let piRlc: Timing
        let piDec: Timing
        let decFormBuild: Timing
        let decProjection: Timing
        let decHostMaterialization: Timing
        let feOnMetal: Bool
        let ajtaiYEvalOnMetal: Bool
        let ncOnMetal: Bool
        let ncMaskNativeOnMetal: Bool
        let rlcWitnessOnMetal: Bool
        let rlcWitnessResidentOnly: Bool
        let rlcRhoSmallCoefficients: Bool
        let decSplitOnMetal: Bool
        let decRecompositionOnMetal: Bool
        let decFormsOnMetal: Bool
        let decYOnMetal: Bool
        let decCommitOnMetal: Bool
        let residentInputFolds: Int
        let residentOutputFolds: Int
        let deferredProofFolds: Int
        let deferredRunningFolds: Int
        let recursiveCompileReverifyRequired: Bool
        let activityPerSample: Activity

        enum CodingKeys: String, CodingKey {
            case total
            case foldsPerSample = "folds_per_sample"
            case piCcs = "pi_ccs"
            case ajtaiYEval = "ajtai_y_eval"
            case piRlc = "pi_rlc"
            case piDec = "pi_dec"
            case decFormBuild = "dec_form_build"
            case decProjection = "dec_projection"
            case decHostMaterialization = "dec_host_materialization"
            case feOnMetal = "fe_on_metal"
            case ajtaiYEvalOnMetal = "ajtai_y_eval_on_metal"
            case ncOnMetal = "nc_on_metal"
            case ncMaskNativeOnMetal = "nc_mask_native_on_metal"
            case rlcWitnessOnMetal = "rlc_witness_on_metal"
            case rlcWitnessResidentOnly = "rlc_witness_resident_only"
            case rlcRhoSmallCoefficients = "rlc_rho_small_coefficients"
            case decSplitOnMetal = "dec_split_on_metal"
            case decRecompositionOnMetal = "dec_recomposition_on_metal"
            case decFormsOnMetal = "dec_forms_on_metal"
            case decYOnMetal = "dec_y_on_metal"
            case decCommitOnMetal = "dec_commit_on_metal"
            case residentInputFolds = "resident_input_folds"
            case residentOutputFolds = "resident_output_folds"
            case deferredProofFolds = "deferred_proof_folds"
            case deferredRunningFolds = "deferred_running_folds"
            case recursiveCompileReverifyRequired = "recursive_compile_reverify_required"
            case activityPerSample = "activity_per_sample"
        }
    }

    struct Activity: Decodable {
        let commandBuffers: UInt64
        let dispatches: UInt64
        let hostWaits: UInt64
        let uploadedBytes: UInt64
        let downloadedBytes: UInt64

        enum CodingKeys: String, CodingKey {
            case dispatches
            case commandBuffers = "command_buffers"
            case hostWaits = "host_waits"
            case uploadedBytes = "uploaded_bytes"
            case downloadedBytes = "downloaded_bytes"
        }
    }

    let schemaVersion: Int
    let device: Device
    let primitives: [Primitive]
    let lifecycle: [Lifecycle]
    let m1ParityPassed: Bool
    let m1CrossoverPassed: Bool
    let m2LifecyclePassed: Bool
    let m2CrossoverPassed: Bool
    let m3ResidencyPassed: Bool
    let m3CrossoverPassed: Bool
    let m4ProjectionPassed: Bool
    let m4CrossoverPassed: Bool
    let m5AdapterPassed: Bool
    let lifecycleCrossover: [LifecycleCrossover]
    let sustained: Sustained?
    let m6PipelinePassed: Bool
    let m6CrossoverPassed: Bool
    let m6SustainedPassed: Bool
    let m6Passed: Bool
    let notes: [String]

    enum CodingKeys: String, CodingKey {
        case device, primitives, lifecycle, notes, sustained
        case lifecycleCrossover = "lifecycle_crossover"
        case schemaVersion = "schema_version"
        case m1ParityPassed = "m1_parity_passed"
        case m1CrossoverPassed = "m1_crossover_passed"
        case m2LifecyclePassed = "m2_lifecycle_passed"
        case m2CrossoverPassed = "m2_crossover_passed"
        case m3ResidencyPassed = "m3_residency_passed"
        case m3CrossoverPassed = "m3_crossover_passed"
        case m4ProjectionPassed = "m4_projection_passed"
        case m4CrossoverPassed = "m4_crossover_passed"
        case m5AdapterPassed = "m5_adapter_passed"
        case m6PipelinePassed = "m6_pipeline_passed"
        case m6CrossoverPassed = "m6_crossover_passed"
        case m6SustainedPassed = "m6_sustained_passed"
        case m6Passed = "m6_passed"
    }
}

@MainActor
final class MetalBenchmarkService: ObservableObject {
    enum Profile: Sendable {
        case quick
        case full
        case m6

        fileprivate var configuration: Data? {
            switch self {
            case .quick:
                return try? JSONSerialization.data(withJSONObject: [
                    "samples": 2,
                    "field_elements": 1 << 10,
                    "poseidon_hashes": 1 << 7,
                    "poseidon_fields_per_hash": 8,
                    "kx_elements": 1 << 9,
                    "kx_rounds": 4,
                    "ajtai_rows": 2,
                    "ajtai_cols": 3,
                    "fe_table_elements": 1 << 9,
                    "lifecycle_repetitions": 1,
                    "run_sha256_lifecycle": false,
                    "run_nebula_lifecycle": false,
                ])
            case .full:
                return nil
            case .m6:
                return try? JSONSerialization.data(withJSONObject: [
                    "samples": 5,
                    "field_elements": 1 << 18,
                    "poseidon_hashes": 1 << 15,
                    "poseidon_fields_per_hash": 8,
                    "kx_elements": 1 << 18,
                    "kx_rounds": 64,
                    "ajtai_rows": 18,
                    "ajtai_cols": 8_377,
                    "fe_table_elements": 1 << 18,
                    "lifecycle_repetitions": 5,
                    "lifecycle_soak_seconds": 60,
                    "run_sha256_lifecycle": true,
                    "run_nebula_lifecycle": true,
                ])
            }
        }
    }

    static var isAvailable: Bool {
        #if canImport(NeoMetalBench)
        true
        #else
        false
        #endif
    }

    @Published private(set) var isRunning = false
    @Published private(set) var report: MetalBenchmarkReport?
    @Published private(set) var rawJSON = ""
    @Published private(set) var errorMessage: String?

    func run(_ profile: Profile) {
        guard Self.isAvailable, !isRunning else { return }
        isRunning = true
        report = nil
        rawJSON = ""
        errorMessage = nil
        let configuration = profile.configuration
        Task {
            do {
                let json = try await Task.detached(priority: .userInitiated) {
                    try Self.invoke(configuration: configuration)
                }.value
                rawJSON = json
                report = try JSONDecoder().decode(MetalBenchmarkReport.self, from: Data(json.utf8))
            } catch {
                errorMessage = error.localizedDescription
            }
            isRunning = false
        }
    }

    private nonisolated static func invoke(configuration: Data?) throws -> String {
        #if canImport(NeoMetalBench)
        var outputPointer: UnsafeMutablePointer<UInt8>?
        var outputLength = 0
        var errorPointer: UnsafeMutablePointer<UInt8>?
        var errorLength = 0
        let status = configuration?.withUnsafeBytes { bytes in
            neo_metal_benchmark_run_json(
                bytes.bindMemory(to: UInt8.self).baseAddress,
                bytes.count,
                &outputPointer,
                &outputLength,
                &errorPointer,
                &errorLength
            )
        } ?? neo_metal_benchmark_run_json(
            nil,
            0,
            &outputPointer,
            &outputLength,
            &errorPointer,
            &errorLength
        )

        defer {
            if let outputPointer {
                neo_metal_benchmark_free_bytes(outputPointer, outputLength)
            }
            if let errorPointer {
                neo_metal_benchmark_free_bytes(errorPointer, errorLength)
            }
        }
        if status == 0, let outputPointer {
            return String(decoding: UnsafeBufferPointer(start: outputPointer, count: outputLength), as: UTF8.self)
        }
        if let errorPointer {
            let message = String(decoding: UnsafeBufferPointer(start: errorPointer, count: errorLength), as: UTF8.self)
            throw MetalBenchmarkFailure(message: message)
        }
        throw MetalBenchmarkFailure(message: "Metal benchmark failed with status \(status)")
        #else
        throw MetalBenchmarkFailure(message: "NeoMetalBench.xcframework is unavailable")
        #endif
    }
}

private struct MetalBenchmarkFailure: LocalizedError, Sendable {
    let message: String
    var errorDescription: String? { message }
}
