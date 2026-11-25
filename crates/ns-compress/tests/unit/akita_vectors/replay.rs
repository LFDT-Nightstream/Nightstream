//! The replay of one Akita opening from public functions. It follows `akita-verifier` at the
//! pinned revision: `fold/verify.rs`, `fold/root.rs`, `fold/suffix.rs`, `fold/mod.rs`,
//! `fold/terminal.rs`, `stages/` and `terminal/direct.rs`.

use akita_algebra::{CyclotomicRing, EqPolynomial};
use akita_challenges::{Challenges, FoldDraw, SparseChallenge, VerifierFoldDraw};
use akita_config::proof_optimized::fp64;
use akita_config::{transcript_instance_descriptor, RecursiveCommitmentConfig};
use akita_error::AkitaError;
use akita_params::sis::checked_centered_l2_sq;
use akita_params::{
    dispatch_for_field, gadget_row_scalars, nonterminal_level_layout, verifier_setup_matrix_capacity_for_schedule,
    BasisMode, CommittedGroupParams, DigitRangePlan, FoldSchedule, GrindingSite, InnerCommitSecurityRoute,
    OpeningClaimsLayout, OpeningMethod, PhysicalL2NormProofShape, PolynomialGroupLayout, RelationWitnessGeometry,
    SetupMode, SubringCoefficientPackingGeometry, SumcheckProtocol, TailSegmentGroupLayout, TerminalFoldParams,
    SETUP_SUMCHECK_DEGREE,
};
use akita_sumcheck::{verify_eq_factored_sumcheck_rounds, verify_sumcheck_rounds, SumcheckShape};
use akita_transcript::{
    bind_instance, extension_slots, field_digest, new_verifier_channel, public_bytes, public_extensions,
    receive_bounded_bytes, receive_field_digest, receive_field_group,
};
use akita_types::proof::relation::relation_row_weight;
use akita_types::{
    assemble_compressed_relation_rhs, assemble_relation_rhs, batch_l2_virtual_evaluations,
    canonical_extension_opening_reduction_shape, decode_terminal_z_golomb_payload,
    derive_tensor_extension_opening_claim_from_partials, draw_group_fold_challenges, eor_final_claims, eor_prefix,
    l2_prefix, l2_virtual_evaluations, prepare_opening_point, recover_ring_subfield_inner_product,
    relation_claim_from_compressed_rhs_extension, ring_subfield_packed_extension_opening_point, row_coefficients,
    sample_akita_setup_seed, stage1_child_claims, stage1_range_image, stage2_w_eval, stage3_claim, stage3_prefix_eval,
    stage3_public_slot, tensor_equality_factor_eval_at_point, tensor_opening_split, tensor_reduction_claim_from_rows,
    tensor_row_partials_from_columns, AkitaSetupDescriptor, CommittedGroup, FoldSite, GrindingReplay,
    GrindingSumcheckVerifier, GroupFoldChallenges, OpeningClaims, OpeningFamily, PhysicalResponsePlan,
    PolynomialGroupClaims, PreparedOpeningPoint, PreparedSubringCoefficientPackingPoint, RelationRangeImagePlan,
    RingRelationGroupOpening, RingRelationInstance, RingVec, SetupPrefixVerifierRegistry, VerifierGrinding,
    EOR_SUMCHECK_INVOCATION,
};
use jolt_field::{One, Ring, Zero};
use serde_json::{json, Value};

use super::trace::{
    base_words, bytes_of, extension_words, failure, hex, parse, sparse, work, Step, Trace, OBSERVED, RECONSTRUCTED,
    STATEMENT,
};
use crate::akita::verifier::schedule::{catalog, row};
use crate::akita::{Base, Extension, Schedules, Shape};

type Config = RecursiveCommitmentConfig<fp64::Dense>;
type Grinding<'a, 'proof> = VerifierGrinding<'proof, 'a>;
type Point = OpeningFamily<PreparedOpeningPoint<Base, Extension>, PreparedSubringCoefficientPackingPoint<Extension>>;

struct Sumcheck {
    output: Extension,
    point: Vec<Extension>,
}

/// `akita-sumcheck/src/proof_stream.rs`: the claim is absorbed, then each round has its message
/// and its challenge. A round message has `degree` elements: the round polynomial without its
/// linear coefficient, or, with the equality factor, without its constant term. The check
/// `<name>.claim_chain` computes the end claim from the round messages that the trace reads
/// (`jolt-poly/src/compressed_univariate.rs`, `akita-sumcheck/src/single.rs`).
fn sumcheck(
    trace: &mut Trace<'_>,
    grinding: &mut Grinding<'_, '_>,
    protocol: SumcheckProtocol,
    level: u32,
    stage: u32,
    name: &str,
    equality: Option<&[Extension]>,
    claim: Extension,
    shape: SumcheckShape,
) -> Step<Sumcheck> {
    trace.absorb(
        level,
        &format!("{name}.claim"),
        RECONSTRUCTED,
        extension_words(&[claim]),
    );
    let site = |round| GrindingSite::SumcheckRound {
        protocol,
        level,
        stage,
        round,
    };
    let mut channel = GrindingSumcheckVerifier::<Base, Extension>::new(grinding, protocol, level, stage);
    let invocation = if protocol == SumcheckProtocol::ExtensionOpeningReduction {
        EOR_SUMCHECK_INVOCATION
    } else {
        0
    };
    let replay = match equality {
        Some(point) => {
            verify_eq_factored_sumcheck_rounds::<Base, Extension, _>(point, claim, shape, &mut channel, invocation)
        }
        None => verify_sumcheck_rounds::<Base, Extension, _>(&mut channel, invocation, claim, shape),
    }
    .map_err(|_| trace.rounds_failure(level, name, shape, site))?;
    trace.events.push(json!({
        "op": "rounds", "level": level, "site": name, "source": RECONSTRUCTED, "rounds": shape.num_rounds(),
        "degree": shape.degree_bound(), "equality_factor": equality.is_some(),
    }));
    let mut chain = claim;
    for round in 0..shape.num_rounds() {
        let message = trace.round(level, &format!("{name}.round"), round, shape.degree_bound());
        trace.work(level, site(round as u32));
        let challenge = replay.challenges[round];
        let horner = |coefficients: &[Extension]| {
            coefficients
                .iter()
                .rev()
                .fold(Extension::zero(), |sum, &coefficient| sum * challenge + coefficient)
        };
        let sum = message
            .iter()
            .fold(Extension::zero(), |sum, &coefficient| sum + coefficient);
        chain = match equality {
            Some(point) => chain - point[round] * sum + horner(&message) * challenge,
            None => {
                let linear = chain - message[0] - sum;
                message[0] + challenge * (linear + challenge * horner(&message[1..]))
            }
        };
    }
    trace.challenge(level, &format!("{name}.point"), extension_words(&replay.challenges));
    trace.value(
        level,
        &format!("{name}.output"),
        OBSERVED,
        extension_words(&[replay.output_claim]),
    );
    if let Some(point) = equality {
        let factor = EqPolynomial::mle(point, &replay.challenges).expect("equality");
        trace.value(
            level,
            &format!("{name}.equality_factor"),
            RECONSTRUCTED,
            extension_words(&[factor]),
        );
    }
    assert_eq!(
        chain, replay.output_claim,
        "the round messages of {name} at level {level}"
    );
    trace.check(
        level,
        &format!("{name}.claim_chain"),
        true,
        &[format!("{name}.round").as_str()],
    )?;
    Ok(Sumcheck {
        output: replay.output_claim,
        point: replay.challenges,
    })
}

fn one(
    grinding: &mut Grinding<'_, '_>,
    trace: &mut Trace<'_>,
    level: u32,
    name: &str,
    site: GrindingSite,
) -> Step<Extension> {
    let challenge = grinding
        .grinded_ext_challenge::<Extension>(site)
        .map_err(work(level, name))?;
    trace.work(level, site);
    trace.challenge(level, name, extension_words(&[challenge]));
    Ok(challenge)
}

struct Prefix {
    points: Vec<Point>,
    openings: Vec<Extension>,
    reduction: Option<(Vec<Extension>, Vec<Extension>)>,
}

struct Carried {
    point: Vec<Extension>,
    opening: Extension,
    witness: Vec<Base>,
    setup: Option<(Vec<Extension>, Extension)>,
}

fn packing_points(
    points: &[&[Extension]],
    batch: &OpeningClaimsLayout,
    basis: BasisMode,
    lp: &CommittedGroupParams,
) -> Vec<Point> {
    (0..batch.num_groups())
        .map(|group| {
            let params = lp.group_params_geometry(batch, group).expect("group");
            let OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension,
            } = params.opening_method()
            else {
                panic!("a group with coefficient packing");
            };
            let geometry = SubringCoefficientPackingGeometry::try_new(
                2,
                params.inner_commit_matrix_params().ring_dimension(),
                challenge_subring_dimension,
            )
            .expect("geometry");
            OpeningFamily::SubringCoefficientPacking(
                PreparedSubringCoefficientPackingPoint::new(
                    geometry,
                    basis,
                    params.num_live_ring_elements_per_claim(),
                    params.num_positions_per_block(),
                    batch.group_layout(group).expect("layout").num_vars(),
                    points[group],
                )
                .expect("point"),
            )
        })
        .collect()
}

struct Reduction {
    rho: Vec<Extension>,
    finals: Vec<Extension>,
    factors: Vec<Extension>,
}

/// `stages/opening_claims/extension_claim.rs`, `verify_eor_sumcheck_with_stream`.
fn tensor_reduction(
    trace: &mut Trace<'_>,
    grinding: &mut Grinding<'_, '_>,
    level: u32,
    groups: &[(Vec<Extension>, Extension)],
    batch: &OpeningClaimsLayout,
) -> Step<Reduction> {
    let openings: Vec<Extension> = groups.iter().map(|group| group.1).collect();
    let (split, width) = tensor_opening_split::<Base, Extension>().expect("split");
    let rounds = canonical_extension_opening_reduction_shape(batch, width)
        .expect("shape")
        .sumcheck
        .len();
    trace.absorb(level, "tensor.openings", OBSERVED, extension_words(&openings));
    let count = width * openings.len();
    let partials =
        extension_slots::<Extension, _>(grinding.state_mut(), count).map_err(parse(level, "tensor.partials"))?;
    let prefix = trace.read_then_worked(
        eor_prefix::<Base, Extension, _>(grinding, batch, &openings, partials, level),
        level,
        "tensor.partials",
        2 * count,
    )?;
    trace.extension_message(level, "tensor.partials", &prefix.partials);
    trace.work(level, GrindingSite::ExtensionOpeningPoint { level });
    trace.challenge(level, "tensor.eta", extension_words(&prefix.eta));
    trace.work(level, GrindingSite::ExtensionOpeningClaimBatch { level });
    if openings.len() > 1 {
        trace.challenge(
            level,
            "tensor.claim_coefficients",
            extension_words(&prefix.claim_coefficients),
        );
    } else {
        trace.value(
            level,
            "tensor.claim_coefficients",
            OBSERVED,
            extension_words(&prefix.claim_coefficients),
        );
    }
    let mut claim = Extension::zero();
    for (index, ((point, opening), partials)) in groups
        .iter()
        .zip(prefix.partials.chunks_exact(width))
        .enumerate()
    {
        let derived = derive_tensor_extension_opening_claim_from_partials::<Base, Extension>(point, partials)
            .expect("opening of the partials");
        trace.check(
            level,
            "tensor.opening",
            derived == *opening,
            &["tensor.partials", "tensor.openings"],
        )?;
        let rows = tensor_row_partials_from_columns::<Base, Extension>(partials).expect("rows");
        claim += prefix.claim_coefficients[index]
            * tensor_reduction_claim_from_rows::<Base, Extension>(&rows, &prefix.eta).expect("claim");
    }
    let replay = sumcheck(
        trace,
        grinding,
        SumcheckProtocol::ExtensionOpeningReduction,
        level,
        0,
        "tensor.sumcheck",
        None,
        claim,
        SumcheckShape::new(rounds, akita_params::EXTENSION_OPENING_REDUCTION_DEGREE).expect("shape"),
    )?;
    let mut finals = extension_slots::<Extension, _>(grinding.state_mut(), openings.len())
        .map_err(parse(level, "tensor.final_claims"))?;
    eor_final_claims::<Base, Extension, _>(grinding, batch, &mut finals, level)
        .map_err(parse(level, "tensor.final_claims"))?;
    trace.extension_message(level, "tensor.final_claims", &finals);
    let end = finals
        .iter()
        .zip(&prefix.claim_coefficients)
        .fold(Extension::zero(), |sum, (&claim, &coefficient)| {
            sum + coefficient * claim
        });
    trace.check(
        level,
        "tensor.end",
        replay.output == end,
        &["tensor.sumcheck.output", "tensor.final_claims"],
    )?;
    let factors: Vec<Extension> = groups
        .iter()
        .map(|(point, _)| {
            let tail = &point[split..];
            let mut factor =
                tensor_equality_factor_eval_at_point::<Base, Extension>(tail, &prefix.eta, &replay.point[..tail.len()])
                    .expect("factor");
            for &extra in &replay.point[tail.len()..] {
                factor *= Extension::one() - extra;
            }
            factor
        })
        .collect();
    trace.value(level, "tensor.factor", RECONSTRUCTED, extension_words(&factors));
    Ok(Reduction {
        rho: replay.point,
        finals,
        factors,
    })
}

fn protocol_point(ring: usize, rho: &[Extension]) -> Result<Vec<Extension>, AkitaError> {
    dispatch_for_field!(ProtocolDispatchSlot::Role(RingRole::Inner), Base, ring, |D| {
        ring_subfield_packed_extension_opening_point::<Base, Extension, D>(rho.len(), rho)
    })
}

struct Fold<'a> {
    lp: &'a CommittedGroupParams,
    level: u32,
    batch: OpeningClaimsLayout,
    commitments: Vec<Vec<Base>>,
    prefix: Prefix,
    witness_len: usize,
    next: Option<&'a CommittedGroupParams>,
    terminal_ring: usize,
    terminal_t: usize,
    registry: &'a SetupPrefixVerifierRegistry<Base>,
}

/// `fold/mod.rs`, `verify_fold`, from the opening payload to the witness evaluation and Stage 3.
/// The end of the fused sum-check and the end of Stage 3 need `RelationMatrixEvaluator`, whose
/// constructor is private. The replay has no check of them.
fn fold(trace: &mut Trace<'_>, grinding: &mut Grinding<'_, '_>, input: Fold<'_>) -> Step<Carried> {
    let Fold {
        lp,
        level,
        batch,
        commitments,
        prefix,
        witness_len,
        next,
        terminal_ring,
        terminal_t,
        registry,
    } = input;
    let geometry = RelationWitnessGeometry::for_level(lp, &batch, 2).expect("geometry");
    let payload_geometry = geometry
        .rhs_layout()
        .opening_payload_geometry()
        .expect("payload geometry");
    let payload = receive_field_group::<Base>(
        grinding.state_mut(),
        FoldSite::OpeningPayload {
            level,
            ring_dimension: payload_geometry.transcript_ring_dimension(),
        }
        .id()
        .expect("site"),
        payload_geometry.transmitted_coefficients(),
    )
    .map_err(parse(level, "opening_payload"))?;
    trace.base_message(level, "opening_payload", &payload);

    let rows = row_coefficients::<Base, Extension, _>(&batch, GrindingSite::EvaluationBatch { level }, grinding)
        .map_err(work(level, "row_coefficients"))?;
    if batch.requires_row_batch_challenge() {
        trace.work(level, GrindingSite::EvaluationBatch { level });
        trace.challenge(level, "row_coefficients", extension_words(&rows));
    }
    let (claim_coefficients, target) = match &prefix.reduction {
        Some((finals, factors)) => (
            batch
                .scale_row_coefficients_by_group(&rows, factors)
                .expect("scaled coefficients"),
            finals
                .iter()
                .zip(&rows)
                .fold(Extension::zero(), |sum, (&claim, &row)| sum + row * claim),
        ),
        None => (
            rows.clone(),
            batch
                .batched_eval_target(&rows, &prefix.openings)
                .expect("target"),
        ),
    };
    trace.value(level, "opening_target", RECONSTRUCTED, extension_words(&[target]));

    let nonce = grinding
        .read_fold_response(GrindingSite::FoldResponse { level })
        .map_err(|_| failure("fold_response", level, "fold_nonce"))?;
    trace.fold_nonce(level, nonce);
    let mut drawn: Vec<GroupFoldChallenges> = Vec::new();
    for group in 0..batch.num_groups() {
        let params = lp.group_params_geometry(&batch, group).expect("group");
        let claims = batch.group_layout(group).expect("layout").num_polynomials();
        let challenges = {
            let mut live = VerifierFoldDraw::new(grinding.state_mut(), level, group as u32);
            draw_group_fold_challenges::<Base, Extension, _>(&mut live, &params, claims)
                .ok()
                .flatten()
                .ok_or_else(|| failure("fold_challenges", level, "fold_challenges"))?
        };
        grinding
            .record_fold_challenges(level, group as u32, params.num_live_blocks() * claims)
            .expect("plan");
        trace.events.push(json!({
            "op": "fold_challenges", "level": level, "group": group, "source": OBSERVED,
            "ring_dimension": params.inner_commit_matrix_params().ring_dimension(),
            "packing": matches!(challenges, OpeningFamily::SubringCoefficientPacking(_)),
            "sparse": sparse(challenges.ambient_a()),
        }));
        drawn.push(challenges);
    }

    let dims = lp.role_dims();
    let (gamma, row_rings) = dispatch_for_field!(ProtocolDispatchSlot::Role(RingRole::Inner), Base, dims.d_a(), |D| {
        RingRelationInstance::<Base>::gamma_and_row_rings_from_coefficients::<D, Extension>(&rows)
    })
    .expect("row rings");
    let rhs = if lp.payload_mode.is_compressed() {
        let groups: Vec<&[Base]> = commitments.iter().map(Vec::as_slice).collect();
        assemble_compressed_relation_rhs::<Base>(geometry.rhs_layout(), &groups, &payload)
    } else {
        assemble_relation_rhs::<Base>(
            geometry.rhs_layout(),
            &RingVec::from_coeffs(payload.clone()),
            &RingVec::from_coeffs(commitments.concat()),
        )
    }
    .expect("relation right-hand side");
    let openings = drawn
        .into_iter()
        .zip(&prefix.points)
        .map(|(challenges, point)| match (challenges, point) {
            (OpeningFamily::EvaluationTrace(challenges), OpeningFamily::EvaluationTrace(point)) => {
                RingRelationGroupOpening::evaluation_trace(challenges, point.ring_multiplier_point.clone())
            }
            (OpeningFamily::SubringCoefficientPacking(challenges), OpeningFamily::SubringCoefficientPacking(_)) => {
                RingRelationGroupOpening::coefficient_packing(challenges)
            }
            _ => panic!("the challenges and the point of a group have two opening methods"),
        })
        .collect();
    let relation =
        RingRelationInstance::new(openings, 2, batch.clone(), gamma, row_rings, rhs, dims).expect("relation");

    let (next_ring, next_count, last) = match next {
        Some(next) => (
            next.d_a(),
            next.outer_payload_geometry()
                .expect("outer payload")
                .transmitted_coefficients(),
            false,
        ),
        None => (terminal_ring, terminal_t, true),
    };
    let site = if last {
        FoldSite::NextWitnessInnerState { level }
    } else {
        FoldSite::NextWitnessPayload { level }
    };
    let name = if last { "terminal_t" } else { "next_witness" };
    let witness = receive_field_group::<Base>(grinding.state_mut(), site.id().expect("site"), next_count)
        .map_err(parse(level, name))?;
    trace.base_message(level, name, &witness);
    if last {
        let digest = receive_field_digest(
            grinding.state_mut(),
            FoldSite::NextWitnessInnerStateDigest { level }
                .id()
                .expect("site"),
        )
        .map_err(parse(level, "terminal_t_last_block_digest"))?;
        trace.message(level, "terminal_t_last_block_digest", &digest, json!(hex(&digest)));
    }

    let alpha = one(grinding, trace, level, "alpha", GrindingSite::RingSwitchAlpha { level })?;
    let layout = relation.segment_layout(lp, None).expect("witness layout");
    let address = lp
        .relation_address_geometry(&batch, 2, next_ring, layout.live_coeff_len())
        .expect("address geometry");
    assert_eq!(
        witness_len,
        address.digit_witness_domain().live_len(),
        "witness of level {level}"
    );
    let tau0 = grinding
        .grinded_ext_challenges::<Extension>(
            GrindingSite::Tau0Point { level },
            address.relation_point_variable_count(),
        )
        .map_err(work(level, "tau0"))?;
    trace.work(level, GrindingSite::Tau0Point { level });
    trace.challenge(level, "tau0", extension_words(&tau0));
    let tau1 = grinding
        .grinded_ext_challenges::<Extension>(
            GrindingSite::Tau1Point { level },
            lp.relation_row_index_num_vars(&batch).expect("rows"),
        )
        .map_err(work(level, "tau1"))?;
    trace.work(level, GrindingSite::Tau1Point { level });
    trace.challenge(level, "tau1", extension_words(&tau1));
    let relation_claim = relation_claim_from_compressed_rhs_extension::<Base, Extension>(
        geometry.rhs_layout(),
        &tau1,
        alpha,
        relation.rhs(),
    )
    .expect("relation claim");
    trace.value(
        level,
        "relation_claim",
        RECONSTRUCTED,
        extension_words(&[relation_claim]),
    );

    let basis = 1usize << lp.open().digits.log_basis;
    let range = DigitRangePlan::new(basis).expect("range plan");
    let image = RelationRangeImagePlan::new(geometry.clone(), address, range, layout, &batch).expect("image plan");
    let low = address.relation_coefficient_variable_count();
    let equality = akita_types::DigitRangeEqualityPoint::from_column_then_ring_challenges(&tau0, tau0.len() - low, low)
        .expect("equality point");
    let rounds = tau0.len();
    let leaves = range.leaf_coeffs::<Extension>();
    let mut point = equality.coordinates().to_vec();
    let mut claim = Extension::zero();
    let mut weights = vec![Extension::one()];
    for (stage, &arity) in range.product_stage_arities().iter().enumerate() {
        let name = format!("range.{stage}");
        let replay = sumcheck(
            trace,
            grinding,
            SumcheckProtocol::Stage1,
            level,
            stage as u32,
            &name,
            Some(&point),
            claim,
            SumcheckShape::new(rounds, arity).expect("shape"),
        )?;
        let count = range
            .stage_shape(rounds, stage)
            .expect("stage")
            .child_claims;
        let children_name = format!("{name}.children");
        let mut children =
            extension_slots::<Extension, _>(grinding.state_mut(), count).map_err(parse(level, &children_name))?;
        stage1_child_claims::<Base, Extension, _>(grinding, level, stage as u32, &mut children)
            .map_err(parse(level, &children_name))?;
        trace.extension_message(level, &children_name, &children);
        let expected =
            weights
                .iter()
                .zip(children.chunks_exact(arity))
                .fold(Extension::zero(), |sum, (&weight, claims)| {
                    sum + weight
                        * claims
                            .iter()
                            .copied()
                            .fold(Extension::one(), |product, claim| product * claim)
                });
        trace.check(
            level,
            &format!("{name}.end"),
            replay.output == expected,
            &[format!("{name}.output").as_str(), children_name.as_str()],
        )?;
        let gamma = one(
            grinding,
            trace,
            level,
            &format!("{name}.batch"),
            GrindingSite::Stage1InterstageBatch {
                level,
                stage: stage as u32,
            },
        )?;
        weights = range.interstage_batch_weights(gamma, children.len());
        claim = range.batch_claims(&weights, &children).expect("batch");
        point = replay.point;
    }
    let leaf = range
        .batch_leaf_polynomials(&weights, &leaves)
        .expect("leaf polynomial");
    let stage = range.product_stage_arities().len() as u32;
    let physical = PhysicalResponsePlan::new(lp, &image).expect("response plan");
    let (stage1_point, range_image, norm_claim) = match &physical {
        None => {
            let replay = sumcheck(
                trace,
                grinding,
                SumcheckProtocol::Stage1,
                level,
                stage,
                "range.leaf",
                Some(&point),
                claim,
                SumcheckShape::new(point.len(), leaf.len() - 1).expect("shape"),
            )?;
            let image = stage1_range_image::<Base, Extension, _>(grinding, level, stage, Extension::zero())
                .map_err(parse(level, "range.image"))?;
            trace.extension_message(level, "range.image", &[image]);
            trace.check(
                level,
                "range.leaf.end",
                replay.output == range.evaluate_leaf_polynomial(&leaf, image),
                &["range.leaf.output", "range.image"],
            )?;
            (replay.point, image, Extension::zero())
        }
        Some(plan) => {
            assert!(
                matches!(plan.shape(), PhysicalL2NormProofShape::Direct { .. }),
                "the direct norm shape"
            );
            let InnerCommitSecurityRoute::L2 {
                response_l2_sq_cap: cap,
                ..
            } = lp.inner().matrix.security_route()
            else {
                panic!("a response plan without the Euclidean route");
            };
            let profile = lp.inner().matrix.sis_modulus_profile();
            let mut none = extension_slots::<Extension, _>(grinding.state_mut(), 0).expect("slots");
            let energy = l2_prefix::<Base, Extension, _>(grinding, level, 0, &mut none)
                .map_err(parse(level, "norm.response_l2_sq"))?;
            trace.message(
                level,
                "norm.response_l2_sq",
                &energy.to_le_bytes(),
                json!(energy.to_string()),
            );
            let merge = one(
                grinding,
                trace,
                level,
                "norm.merge",
                GrindingSite::L2NormMerge { level },
            )?;
            let replay = sumcheck(
                trace,
                grinding,
                SumcheckProtocol::PhysicalL2,
                level,
                0,
                "norm.sumcheck",
                None,
                claim + merge * Extension::from_u128(energy),
                SumcheckShape::new(plan.domain().num_vars(), leaf.len()).expect("shape"),
            )?;
            let mut virtuals = extension_slots::<Extension, _>(grinding.state_mut(), 1)
                .map_err(parse(level, "norm.virtual_evaluations"))?;
            l2_virtual_evaluations::<Base, Extension, _>(grinding, level, &mut virtuals)
                .map_err(parse(level, "norm.virtual_evaluations"))?;
            trace.extension_message(level, "norm.virtual_evaluations", &virtuals);
            let image = stage1_range_image::<Base, Extension, _>(grinding, level, stage, Extension::zero())
                .map_err(parse(level, "range.image"))?;
            trace.extension_message(level, "range.image", &[image]);

            // `stages/physical_l2_norm.rs`, `validate_integer_claim` and `expected_output_claim`.
            trace.events.push(json!({
                "op": "value", "level": level, "site": "norm.bounds", "source": RECONSTRUCTED,
                "response_l2_sq": energy.to_string(), "cap": cap.to_string(), "modulus": profile.modulus().to_string(),
            }));
            let sound = plan
                .shape()
                .validate_integer_soundness(profile, plan.fold_basis(), plan.fold_digit_count())
                .is_ok();
            trace.check(
                level,
                "norm.integer",
                sound && energy <= cap && energy < profile.modulus(),
                &["norm.response_l2_sq", "norm.bounds"],
            )?;
            let range_equality = EqPolynomial::mle(&point, &replay.point).expect("equality");
            let range_leaf = leaf
                .iter()
                .rev()
                .fold(Extension::zero(), |sum, &coefficient| sum * image + coefficient);
            let square = virtuals[0] * virtuals[0];
            trace.value(
                level,
                "norm.range_equality",
                RECONSTRUCTED,
                extension_words(&[range_equality]),
            );
            trace.value(level, "norm.range_leaf", RECONSTRUCTED, extension_words(&[range_leaf]));
            trace.value(level, "norm.square", RECONSTRUCTED, extension_words(&[square]));
            trace.check(
                level,
                "norm.end",
                replay.output == range_equality * range_leaf + merge * square,
                &[
                    "norm.sumcheck.output",
                    "norm.virtual_evaluations",
                    "range.image",
                    "norm.range_equality",
                    "norm.range_leaf",
                    "norm.square",
                ],
            )?;

            let eta = one(
                grinding,
                trace,
                level,
                "norm.virtual_batch",
                GrindingSite::L2VirtualBatch { level },
            )?;
            (replay.point, image, batch_l2_virtual_evaluations(eta, &virtuals).0)
        }
    };
    if lp.payload_mode.is_compressed() {
        one(
            grinding,
            trace,
            level,
            "binary_batch",
            GrindingSite::CompressionBinary { level },
        )?;
    }
    let batching = one(
        grinding,
        trace,
        level,
        "stage2_batch",
        GrindingSite::Stage2Batch { level },
    )?;
    trace.value(level, "stage1_point", OBSERVED, extension_words(&stage1_point));

    let opening_claim = match &prefix.points[0] {
        OpeningFamily::SubringCoefficientPacking(_) => {
            let weight = relation_row_weight(image.scalar_opening_row_index().expect("row"), &tau1).expect("weight");
            let authenticated = prefix
                .openings
                .iter()
                .zip(&claim_coefficients)
                .fold(Extension::zero(), |sum, (&opening, &coefficient)| {
                    sum + opening * coefficient
                });
            assert_eq!(
                authenticated, target,
                "`stages/opening_semantics.rs`, the packing target"
            );
            weight * authenticated
        }
        OpeningFamily::EvaluationTrace(_) => {
            relation_row_weight(lp.evaluation_trace_row_index(&batch).expect("row"), &tau1).expect("weight") * target
        }
    };
    trace.value(level, "opening_claim", RECONSTRUCTED, extension_words(&[opening_claim]));
    let shape = nonterminal_level_layout(64, 128, lp, address, next)
        .expect("level layout")
        .stage2_sumcheck();
    let replay = sumcheck(
        trace,
        grinding,
        SumcheckProtocol::Stage2,
        level,
        0,
        "fused",
        None,
        batching * range_image + relation_claim + opening_claim + norm_claim,
        shape,
    )?;
    let evaluation = stage2_w_eval::<Base, Extension, _>(grinding, level, Extension::zero())
        .map_err(parse(level, "witness_evaluation"))?;
    trace.extension_message(level, "witness_evaluation", &[evaluation]);

    // `stages/stage3.rs`, `verify_stage3`, with `FoldSetup::Ordinary`.
    let setup = match next.and_then(|next| next.setup_prefix()) {
        None => None,
        Some(prefix) => {
            let id = prefix.slot_id().expect("slot");
            let slot = registry.get(&id).expect("the slot of the setup prefix");
            let encoded = bytes_of(&slot.id);
            stage3_public_slot(grinding, level, &encoded).expect("slot");
            trace.events.push(json!({
                "op": "absorb", "level": level, "site": "stage3.slot", "source": STATEMENT, "bytes": hex(&encoded),
            }));
            let claim = stage3_claim::<Base, Extension, _>(grinding, level, Extension::zero())
                .map_err(parse(level, "stage3.setup_claim"))?;
            trace.extension_message(level, "stage3.setup_claim", &[claim]);
            let rounds = id.n_prefix().expect("prefix length").trailing_zeros() as usize;
            let replay = sumcheck(
                trace,
                grinding,
                SumcheckProtocol::Stage3,
                level,
                0,
                "stage3",
                None,
                claim,
                SumcheckShape::new(rounds, SETUP_SUMCHECK_DEGREE).expect("shape"),
            )?;
            let value = stage3_prefix_eval::<Base, Extension, _>(grinding, level, Extension::zero())
                .map_err(parse(level, "stage3.prefix_evaluation"))?;
            trace.extension_message(level, "stage3.prefix_evaluation", &[value]);
            Some((replay.point, value))
        }
    };
    Ok(Carried {
        point: replay.point,
        opening: evaluation,
        witness,
        setup,
    })
}

fn times<const D: usize>(challenge: &SparseChallenge, value: &CyclotomicRing<Base, D>) -> CyclotomicRing<Base, D> {
    let mut product = [Base::zero(); D];
    for (&position, &coefficient) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let scale = Base::from_i64(i64::from(coefficient));
        for (index, &cell) in value.coefficients().iter().enumerate() {
            let at = index + position as usize;
            if at < D {
                product[at] += scale * cell;
            } else {
                product[at - D] -= scale * cell;
            }
        }
    }
    CyclotomicRing::from_coefficients(product)
}

struct Terminal<'a> {
    level: u32,
    params: &'a TerminalFoldParams,
    group: &'a TailSegmentGroupLayout,
    point: &'a [Extension],
    sent: &'a [Base],
    digest: [u8; 32],
    challenges: &'a Challenges,
    response: &'a [u8],
    target: Extension,
    scale: Extension,
}

/// `terminal/direct.rs`: the decoding of `z`, its norm, the consistency relation with the
/// recovery of the last block of `e`, and `verify_terminal_trace`. The rows of `A`
/// (`a_row_sides`, `terminal/ntt.rs`) and the digest of the last block of `t` are private; the
/// replay has no check of them.
fn terminal_checks<const D: usize>(trace: &mut Trace<'_>, input: &Terminal<'_>) -> Step<()> {
    let &Terminal {
        level,
        params,
        group,
        point,
        sent,
        digest,
        challenges,
        response,
        target,
        scale,
    } = input;
    let z = decode_terminal_z_golomb_payload(response, group).map_err(|_| failure("decode", level, "terminal.z"))?;
    trace.events.push(json!({
        "op": "value", "level": level, "site": "terminal.z.coefficients", "source": RECONSTRUCTED, "words": z,
    }));
    let norm = checked_centered_l2_sq(&z);
    let cap = params.response_l2_sq_cap();
    trace.events.push(json!({
        "op": "value", "level": level, "site": "terminal.z.bounds", "source": RECONSTRUCTED,
        "l2_sq": norm.map(|norm| norm.to_string()), "cap": cap.map(|cap| cap.to_string()),
    }));
    trace.check(
        level,
        "terminal.z.norm",
        cap.is_none_or(|cap| norm.is_some_and(|norm| norm <= cap)),
        &["terminal.z.coefficients", "terminal.z.bounds"],
    )?;

    let (rings, rest) = z.as_chunks::<D>();
    assert!(rest.is_empty(), "the response has whole ring elements");
    let positions = params.blocks.positions_per_block;
    let digits = params.inner.digits.num_digits;
    assert_eq!(rings.len(), positions * digits);
    let gadget = gadget_row_scalars::<Base>(digits, params.inner.digits.log_basis);
    let prepared = prepare_opening_point::<Base, Extension, D>(
        point,
        BasisMode::Lagrange,
        positions,
        params.blocks.live_blocks,
        D.trailing_zeros() as usize,
    )
    .expect("point");
    let multiplier = &prepared.ring_multiplier_point;
    assert!(multiplier.as_base().is_none(), "a point in the extension field");
    let mut reduced = CyclotomicRing::<Base, D>::zero();
    for position in 0..positions {
        let mut value = CyclotomicRing::<Base, D>::zero();
        for digit in 0..digits {
            let ring = &rings[position * digits + digit];
            value +=
                CyclotomicRing::from_coefficients(std::array::from_fn(|index| Base::from_i64(i64::from(ring[index]))))
                    .scale(&gadget[digit]);
        }
        multiplier
            .accumulate_position_product(position, &value, &mut reduced)
            .expect("product");
    }

    let sent = RingVec::from_coeffs(sent.to_vec());
    let sent = sent.as_ring_slice::<D>().expect("blocks of e");
    let (last, others) = challenges.as_slice().split_last().expect("challenges");
    assert_eq!(sent.len(), others.len());
    let folded = others
        .iter()
        .zip(sent)
        .fold(CyclotomicRing::<Base, D>::zero(), |sum, (challenge, block)| {
            sum + times(challenge, block)
        });
    let inverse = times(last, &CyclotomicRing::one())
        .inverse()
        .ok_or_else(|| failure("equation", level, "terminal.e_last_block"))?;
    let recovered = inverse * (reduced - folded);
    assert!(folded + times(last, &recovered) == reduced, "the consistency relation");
    trace.value(
        level,
        "terminal.e_last_block",
        RECONSTRUCTED,
        base_words(recovered.coefficients()),
    );
    trace.check(
        level,
        "terminal.e_last_block.digest",
        field_digest(recovered.coefficients()) == digest,
        &[
            "terminal.e",
            "terminal.e_last_block_digest",
            "terminal.e_last_block",
            "terminal.z",
            "terminal.z.coefficients",
            "fold_challenges",
        ],
    )?;

    let packed = prepared.packed_inner_trusted::<D>().expect("point");
    let mut opening = Extension::zero();
    for (block, value) in sent.iter().chain([&recovered]).enumerate() {
        let weight = multiplier
            .fold_subfield_value::<Extension>(block)
            .expect("weight")
            .expect("a point in the extension field");
        opening +=
            weight * recover_ring_subfield_inner_product::<Base, Extension, D>(value, packed).expect("inner product");
    }
    trace.value(level, "terminal.opening", RECONSTRUCTED, extension_words(&[opening]));
    trace.check(
        level,
        "terminal.trace",
        scale * opening == target,
        &["terminal.opening", "tensor.final_claims", "tensor.factor"],
    )
}

pub(super) struct Statement<'a> {
    pub(super) schedules: &'static Schedules,
    pub(super) shape: Shape,
    pub(super) commitments: &'a [CommittedGroup<Base>],
    pub(super) points: &'a [Vec<Extension>],
    pub(super) values: &'a [Extension],
    pub(super) registry: &'a SetupPrefixVerifierRegistry<Base>,
    pub(super) session: &'a [u8],
}

fn layout_of(groups: &[(Vec<Extension>, Extension)]) -> OpeningClaimsLayout {
    OpeningClaimsLayout::from_groups(
        groups
            .iter()
            .map(|(point, _)| PolynomialGroupLayout::new(point.len(), 1))
            .collect(),
    )
    .expect("layout")
}

pub(super) fn replay(statement: &Statement<'_>, proof: &[u8]) -> Step<Value> {
    let &Statement {
        schedules,
        shape,
        commitments,
        points,
        values,
        registry,
        session,
    } = statement;
    let catalog = catalog(schedules).expect("catalog");
    let resolved = row(&catalog, shape).expect("row");
    let schedule: &FoldSchedule = resolved.schedule();
    let groups: Vec<_> = points
        .iter()
        .zip(values)
        .zip(commitments)
        .map(|((point, &value), commitment)| {
            PolynomialGroupClaims::new(point.clone(), vec![value], commitment).expect("group")
        })
        .collect();
    let claims = OpeningClaims::from_groups(groups).expect("claims");
    let batch = claims.committed_layout().expect("layout");
    let layout = resolved.profiles().opening_layout().expect("layout");
    let descriptor = AkitaSetupDescriptor {
        max_num_vars: shape.key.max(shape.witness).max(shape.inverse),
        max_num_batched_polys: 3,
        num_field_elements: verifier_setup_matrix_capacity_for_schedule(schedule, &layout, SetupMode::Check)
            .expect("capacity")
            .num_field_elements,
        setup_seed: sample_akita_setup_seed(),
    };
    let (plan, descriptor_bytes) = transcript_instance_descriptor::<Base, Config>(
        &descriptor,
        &batch,
        resolved.selection(),
        schedule,
        BasisMode::Lagrange,
        None,
    )
    .expect("descriptor");
    let runs: Value = plan
        .runs()
        .iter()
        .map(|run| {
            json!({
                "site": format!("{:?}", run.site()), "loss": run.loss_factor(), "bits": run.grind_bits(),
                "nonce_bits": run.nonce_bits(), "count": run.multiplicity(), "set_bits": run.challenge_set_bits(),
            })
        })
        .collect();

    let mut state = new_verifier_channel(session, proof);
    bind_instance(&mut state, &descriptor_bytes).expect("instance");
    let mut grinding = VerifierGrinding::new(&mut state, &plan);
    let mut trace = Trace {
        proof,
        cursor: 0,
        work: plan
            .runs()
            .iter()
            .map(|run| (run.site(), run.grind_bits()))
            .collect(),
        events: Vec::new(),
    };

    let root = &schedule.root_fold().params;
    let geometry = RelationWitnessGeometry::for_level(root, &batch, 2).expect("geometry");
    for (group, commitment) in commitments.iter().enumerate() {
        let ring_dimension = geometry
            .rhs_layout()
            .compression_plan_for_group(group)
            .expect("plan")
            .maps()
            .last()
            .expect("map")
            .ring_dimension();
        public_bytes(
            grinding.state_mut(),
            FoldSite::RootCommitment { group, ring_dimension }
                .id()
                .expect("site"),
            commitment.digest().as_bytes(),
        )
        .expect("digest");
        trace.events.push(json!({
            "op": "absorb", "level": 0, "site": "root.commitment_digest", "source": STATEMENT, "group": group,
            "bytes": hex(commitment.digest().as_bytes()),
        }));
    }
    for (group, claims) in claims.groups().iter().enumerate() {
        public_extensions::<Base, Extension, _>(
            grinding.state_mut(),
            FoldSite::RootPoint { group }.id().expect("site"),
            claims.point(),
        )
        .expect("point");
        trace.absorb(
            0,
            &format!("root.point.{group}"),
            STATEMENT,
            extension_words(claims.point()),
        );
    }
    let openings = claims.flat_evaluations();
    public_extensions::<Base, Extension, _>(
        grinding.state_mut(),
        FoldSite::Openings { level: 0 }.id().expect("site"),
        &openings,
    )
    .expect("openings");
    trace.absorb(0, "openings", STATEMENT, extension_words(&openings));

    let folds = &schedule.recursive_folds;
    let terminal = &schedule.terminal;
    let terminal_group = &terminal.response_shape.layout.groups[0];
    let root_points: Vec<&[Extension]> = claims.groups().iter().map(|group| group.point()).collect();
    let order = batch.root_group_order().expect("order");
    let mut carried = fold(
        &mut trace,
        &mut grinding,
        Fold {
            lp: root,
            level: 0,
            commitments: order
                .into_iter()
                .map(|group| commitments[group].rows().coeffs().to_vec())
                .collect(),
            prefix: Prefix {
                points: packing_points(&root_points, &batch, BasisMode::Lagrange, root),
                openings: openings.clone(),
                reduction: None,
            },
            witness_len: root
                .output_witness_len::<Base>(&batch, 2)
                .expect("witness length"),
            batch,
            next: folds.first().map(|next| &next.params),
            terminal_ring: terminal.d_a(),
            terminal_t: terminal_group.t_field_elems,
            registry,
        },
    )?;

    for (offset, step) in folds.iter().enumerate() {
        let level = offset as u32 + 1;
        let lp = &step.params;
        let mut groups = vec![(carried.point.clone(), carried.opening)];
        let mut payloads = vec![carried.witness.clone()];
        match (&carried.setup, lp.setup_prefix()) {
            (Some((point, value)), Some(prefix)) => {
                groups.insert(0, (point.clone(), *value));
                let slot = registry
                    .get(&prefix.slot_id().expect("slot"))
                    .expect("the slot of the setup prefix");
                let rows: Vec<Base> = slot
                    .commitment
                    .rows
                    .iter()
                    .flat_map(|row| row.coeffs().to_vec())
                    .collect();
                trace.value(level, "setup_prefix.commitment_rows", STATEMENT, base_words(&rows));
                payloads.insert(0, rows);
            }
            (None, None) => {}
            _ => panic!("the setup prefix of level {level}"),
        }
        let batch = layout_of(&groups);
        let group_points: Vec<&[Extension]> = groups.iter().map(|group| &group.0[..]).collect();
        let openings: Vec<Extension> = groups.iter().map(|group| group.1).collect();
        let prefix = if matches!(lp.opening_method(), OpeningMethod::SubringCoefficientPacking { .. }) {
            for (group, point) in group_points.iter().enumerate() {
                public_extensions::<Base, Extension, _>(
                    grinding.state_mut(),
                    FoldSite::GroupPoint { level, group }.id().expect("site"),
                    point,
                )
                .expect("point");
                trace.absorb(level, &format!("point.{group}"), OBSERVED, extension_words(point));
            }
            public_extensions::<Base, Extension, _>(
                grinding.state_mut(),
                FoldSite::Openings { level }.id().expect("site"),
                &openings,
            )
            .expect("opening");
            trace.absorb(level, "openings", OBSERVED, extension_words(&openings));
            Prefix {
                points: packing_points(&group_points, &batch, BasisMode::Lagrange, lp),
                openings,
                reduction: None,
            }
        } else {
            let reduction = tensor_reduction(&mut trace, &mut grinding, level, &groups, &batch)?;
            let (split, _) = tensor_opening_split::<Base, Extension>().expect("split");
            let mut prepared_points = Vec::new();
            let mut protocol_points = Vec::new();
            for (group, point) in group_points.iter().enumerate() {
                let params = lp.group_params(&batch, group).expect("group");
                let ring = lp.group_role_dims(&batch, group).expect("dims").d_a();
                let protocol = protocol_point(ring, &reduction.rho[..point.len() - split]).expect("point");
                let prepared = dispatch_for_field!(ProtocolDispatchSlot::Role(RingRole::Inner), Base, ring, |D| {
                    prepare_opening_point::<Base, Extension, D>(
                        &protocol,
                        BasisMode::Lagrange,
                        params.num_positions_per_block(),
                        params.num_live_blocks(),
                        ring.trailing_zeros() as usize,
                    )
                })
                .expect("point");
                prepared_points.push(OpeningFamily::EvaluationTrace(prepared));
                protocol_points.push(protocol);
            }
            for (group, protocol) in protocol_points.iter().enumerate() {
                public_extensions::<Base, Extension, _>(
                    grinding.state_mut(),
                    FoldSite::GroupPoint { level, group }.id().expect("site"),
                    protocol,
                )
                .expect("point");
                trace.absorb(
                    level,
                    &format!("point.{group}"),
                    RECONSTRUCTED,
                    extension_words(protocol),
                );
            }
            Prefix {
                points: prepared_points,
                openings,
                reduction: Some((reduction.finals, reduction.factors)),
            }
        };
        let order = batch.root_group_order().expect("order");
        carried = fold(
            &mut trace,
            &mut grinding,
            Fold {
                lp,
                level,
                commitments: order
                    .into_iter()
                    .map(|group| payloads[group].clone())
                    .collect(),
                batch,
                prefix,
                witness_len: step.output_witness_len,
                next: folds.get(offset + 1).map(|next| &next.params),
                terminal_ring: terminal.d_a(),
                terminal_t: terminal_group.t_field_elems,
                registry,
            },
        )?;
    }

    // `fold/terminal.rs`, `verify_terminal_suffix`.
    let level = folds.len() as u32 + 1;
    assert!(carried.setup.is_none(), "the terminal has no group of the setup prefix");
    let groups = [(carried.point.clone(), carried.opening)];
    let batch = layout_of(&groups);
    let reduction = tensor_reduction(&mut trace, &mut grinding, level, &groups, &batch)?;
    let protocol = protocol_point(terminal.d_a(), &reduction.rho).expect("point");
    public_extensions::<Base, Extension, _>(
        grinding.state_mut(),
        FoldSite::TerminalPoint { level }.id().expect("site"),
        &protocol,
    )
    .expect("point");
    trace.absorb(level, "terminal.point", RECONSTRUCTED, extension_words(&protocol));
    row_coefficients::<Base, Extension, _>(&batch, GrindingSite::EvaluationBatch { level }, &mut grinding)
        .map_err(work(level, "row_coefficients"))?;
    let sent = receive_field_group::<Base>(
        grinding.state_mut(),
        FoldSite::TerminalEFields { level }.id().expect("site"),
        terminal_group.e_field_elems,
    )
    .map_err(parse(level, "terminal.e"))?;
    trace.base_message(level, "terminal.e", &sent);
    let digest = receive_field_digest(
        grinding.state_mut(),
        FoldSite::TerminalEDigest { level }.id().expect("site"),
    )
    .map_err(parse(level, "terminal.e_last_block_digest"))?;
    trace.message(level, "terminal.e_last_block_digest", &digest, json!(hex(&digest)));
    let nonce = grinding
        .read_fold_response(GrindingSite::FoldResponse { level })
        .map_err(|_| failure("fold_response", level, "fold_nonce"))?;
    trace.fold_nonce(level, nonce);
    let family = terminal.fold_challenge_family().expect("family");
    let challenges = VerifierFoldDraw::new(grinding.state_mut(), level, 0)
        .draw_fold_challenges(&family, terminal.blocks.live_blocks, 1)
        .ok()
        .flatten()
        .ok_or_else(|| failure("fold_challenges", level, "fold_challenges"))?;
    grinding
        .record_fold_challenges(level, 0, terminal.blocks.live_blocks)
        .expect("plan");
    trace.events.push(json!({
        "op": "fold_challenges", "level": level, "group": 0, "source": OBSERVED, "ring_dimension": terminal.d_a(),
        "packing": false, "sparse": sparse(&challenges),
    }));
    let response = receive_bounded_bytes(
        grinding.state_mut(),
        FoldSite::TerminalZPayload { level }.id().expect("site"),
        terminal_group.z_payload_bytes,
    )
    .map_err(parse(level, "terminal.z"))?;
    trace.message(
        level,
        "terminal.z.length",
        &(response.len() as u32).to_le_bytes(),
        json!(response.len()),
    );
    trace.message(level, "terminal.z", &response, json!(hex(&response)));
    let input = Terminal {
        level,
        params: terminal,
        group: terminal_group,
        point: &protocol,
        sent: &sent,
        digest,
        challenges: &challenges,
        response: &response,
        target: reduction.finals[0],
        scale: reduction.factors[0],
    };
    dispatch_for_field!(ProtocolDispatchSlot::Role(RingRole::Inner), Base, terminal.d_a(), |D| {
        Ok::<_, AkitaError>(terminal_checks::<D>(&mut trace, &input))
    })
    .expect("ring dimension")?;

    // `fold/verify.rs`, `batched_verify`.
    let pending = grinding
        .finish()
        .map_err(|_| failure("plan", level, "grinding_plan"))?;
    pending
        .check_eof(state)
        .map_err(|_| failure("trailing_bytes", level, "end"))?;
    assert_eq!(
        trace.cursor,
        proof.len(),
        "the cursor of the trace is at the end of the proof"
    );
    let events = trace.events_with_their_checks();
    Ok(json!({
        "descriptor": hex(&descriptor_bytes),
        "setup_field_elements": descriptor.num_field_elements,
        "grinding_plan": runs,
        "events": events,
    }))
}
