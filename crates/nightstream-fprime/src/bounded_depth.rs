//! The bounded-depth step packages: the light and the refresh step circuits that
//! Lean writes (`Export/BoundedDepth/StepPackage`).
//!
//! Owns decoding a package, the one identity of both packages
//! (`StepPackage.identity`) and the expansion of a package into the plan rows
//! of its circuit's relation (`CircuitRelation.plan`, in `expand`). Does not
//! own the relation's meaning, the witness program or the fold.

mod application;
mod expand;
pub mod layout;
mod witness;
pub mod words;

pub use application::{NativeBlock, NativeShape};
pub use expand::PlanRows;

use std::sync::Arc;

use neo_ccs::crypto::poseidon2_goldilocks as poseidon2;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use serde_json::Value;

use crate::application_records::ApplicationRecords;
use crate::identity::append_value_preimage;
use crate::package::json::{array, exact_array, usize_atom};
use crate::package::{IdentityKind, PackageError};
use crate::witness::{validate_witness_batch, validate_witness_batch_order, RawWitnessBatch, WitnessBatch};

const IDENTITY_DOMAIN: &[u8] = b"Nightstream/OptionB/packages/v1";
/// The identity of the packages in `formal/nightstream-fprime/artifacts`.
pub const BOUNDED_DEPTH_PACKAGES_IDENTITY: [u64; 4] = [
    9_707_591_829_227_951_822,
    2_741_336_479_425_895_323,
    6_230_668_085_004_830_174,
    14_367_988_604_119_679_107,
];

/// A lane or a row side as `Pieces.linear?` reads it: a constant and terms.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Linear {
    pub constant: Goldilocks,
    pub terms: Vec<(usize, Goldilocks)>,
}

impl Linear {
    fn scale(mut self, factor: Goldilocks) -> Self {
        self.constant *= factor;
        for term in &mut self.terms {
            term.1 *= factor;
        }
        self
    }

    fn plus(mut self, other: Self) -> Self {
        self.constant += other.constant;
        self.terms.extend(other.terms);
        self
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Row {
    pub(crate) a: Linear,
    pub(crate) b: Linear,
    pub(crate) c: Linear,
}

/// A package piece (`StepPackage.PackagePiece`).
#[derive(Clone, Debug)]
pub(crate) enum Piece {
    Rows(Vec<Row>),
    Ring {
        cells: usize,
        start: usize,
        quotient: usize,
        lanes: Vec<Linear>,
    },
    Block {
        start: usize,
        lanes: Vec<Linear>,
    },
}

/// The application's region (`StepPackage.Package.application`): its witness
/// words, its first local and local count, its pieces from the first piece and
/// its fresh columns from `first_fresh`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Region {
    witness_words: usize,
    local_start: usize,
    locals: usize,
    pieces: usize,
    fresh: usize,
}

/// One step circuit's package (`StepPackage.Package`). A native application
/// in place of Lean's fills its locals from its records.
#[derive(Clone, Debug)]
pub struct StepPackage {
    circuit: usize,
    public_columns: usize,
    width: usize,
    first_fresh: usize,
    sbox_start: usize,
    pieces: Vec<Piece>,
    witness: Vec<WitnessBatch>,
    slots: Vec<(usize, usize, u8)>,
    application: Region,
    native: Option<Arc<ApplicationRecords>>,
}

impl StepPackage {
    pub fn circuit(&self) -> usize {
        self.circuit
    }

    pub fn public_columns(&self) -> usize {
        self.public_columns
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn first_fresh(&self) -> usize {
        self.first_fresh
    }

    pub fn sbox_start(&self) -> usize {
        self.sbox_start
    }

    /// The application's witness word count, which places the shared words
    /// after it.
    pub fn witness_words(&self) -> usize {
        self.application.witness_words
    }

    pub fn witness_batch_count(&self) -> usize {
        self.witness.len()
    }

    pub fn block_count(&self) -> usize {
        self.blocks().count()
    }

    /// The packed Poseidon2 blocks in package order.
    pub fn blocks(&self) -> impl Iterator<Item = Block<'_>> {
        self.pieces.iter().filter_map(|piece| match piece {
            Piece::Block { start, lanes } => Some(Block { start: *start, lanes }),
            _ => None,
        })
    }

    fn decode(value: &Value) -> Result<Self, PackageError> {
        let fields = exact_array(value, 9, "bounded-depth package")?;
        let pieces = array(&fields[5], "bounded-depth pieces")?
            .iter()
            .map(decode_piece)
            .collect::<Result<_, _>>()?;
        let mut region = [0; 5];
        let mut rest = &fields[8];
        for (index, word) in region.iter_mut().enumerate() {
            if index == 4 {
                *word = usize_atom(rest, "bounded-depth application")?;
            } else {
                let pair = exact_array(rest, 2, "bounded-depth application")?;
                *word = usize_atom(&pair[0], "bounded-depth application")?;
                rest = &pair[1];
            }
        }
        let [witness_words, local_start, locals, pieces_count, fresh] = region;
        let application = Region {
            witness_words,
            local_start,
            locals,
            pieces: pieces_count,
            fresh,
        };
        let first_fresh = usize_atom(&fields[3], "bounded-depth first fresh column")?;
        let witness = array(&fields[6], "bounded-depth witness")?
            .iter()
            .map(|raw| {
                let raw: RawWitnessBatch = serde_json::from_value(raw.clone())?;
                validate_witness_batch(raw, local_start, first_fresh, 0, first_fresh)
            })
            .collect::<Result<Vec<_>, _>>()?;
        validate_witness_batch_order(&witness)?;
        Ok(Self {
            circuit: usize_atom(&fields[0], "bounded-depth circuit")?,
            public_columns: usize_atom(&fields[1], "bounded-depth public columns")?,
            width: usize_atom(&fields[2], "bounded-depth width")?,
            first_fresh,
            sbox_start: usize_atom(&fields[4], "bounded-depth S-box start")?,
            pieces,
            witness,
            slots: array(&fields[7], "bounded-depth slots")?
                .iter()
                .map(|run| {
                    let pair = exact_array(run, 2, "bounded-depth slot run")?;
                    let tail = exact_array(&pair[1], 2, "bounded-depth slot run")?;
                    let kind = usize_atom(&tail[1], "bounded-depth slot kind")?;
                    Ok((
                        usize_atom(&pair[0], "bounded-depth slot start")?,
                        usize_atom(&tail[0], "bounded-depth slot count")?,
                        u8::try_from(kind)
                            .ok()
                            .filter(|kind| *kind <= 2)
                            .ok_or(PackageError::Malformed("bounded-depth slot kind"))?,
                    ))
                })
                .collect::<Result<_, PackageError>>()?,
            application,
            native: None,
        })
    }
}

/// A packed Poseidon2 block: its 1,096 locals start at `start`, and its
/// output locals are the last 16.
#[derive(Clone, Copy, Debug)]
pub struct Block<'a> {
    start: usize,
    lanes: &'a [Linear],
}

impl Block<'_> {
    pub fn start(&self) -> usize {
        self.start
    }

    /// The first output local.
    pub fn outputs(&self) -> usize {
        self.start + 1080
    }

    /// The input lanes evaluated on `assignment`.
    pub fn lane_values(&self, assignment: &[Goldilocks]) -> Vec<Goldilocks> {
        self.lanes
            .iter()
            .map(|lane| {
                lane.terms
                    .iter()
                    .fold(lane.constant, |sum, &(column, coefficient)| {
                        sum + coefficient * assignment[column]
                    })
            })
            .collect()
    }
}

/// The light and the refresh packages and their one identity.
#[derive(Clone, Debug)]
pub struct BoundedDepthPackages {
    light: StepPackage,
    refresh: StepPackage,
    identity: [u64; 4],
}

impl BoundedDepthPackages {
    pub fn light(&self) -> &StepPackage {
        &self.light
    }

    pub fn refresh(&self) -> &StepPackage {
        &self.refresh
    }

    pub fn identity(&self) -> [u64; 4] {
        self.identity
    }
}

/// Decodes both packages and checks their identity against `expected`.
pub fn load_bounded_depth_packages(
    light: &[u8],
    refresh: &[u8],
    expected: [u64; 4],
) -> Result<BoundedDepthPackages, PackageError> {
    let (pair, light, refresh) = decode_pair(light, refresh)?;
    let identity = bounded_depth_identity(&pair)?;
    if identity != expected {
        return Err(PackageError::IdentityMismatch {
            kind: IdentityKind::Package,
            expected,
            computed: identity,
        });
    }
    Ok(BoundedDepthPackages {
        identity,
        light,
        refresh,
    })
}

/// Decodes both packages without hashing them: their identity is `pinned`.
/// Only for bytes that the caller's binary embeds, whose identity a test
/// checks with `load_bounded_depth_packages`.
pub fn load_pinned_bounded_depth_packages(
    light: &[u8],
    refresh: &[u8],
    pinned: [u64; 4],
) -> Result<BoundedDepthPackages, PackageError> {
    let (_, light, refresh) = decode_pair(light, refresh)?;
    Ok(BoundedDepthPackages {
        identity: pinned,
        light,
        refresh,
    })
}

fn decode_pair(light: &[u8], refresh: &[u8]) -> Result<(Value, StepPackage, StepPackage), PackageError> {
    let pair = Value::Array(vec![parse_atoms(light)?, parse_atoms(refresh)?]);
    let [light_value, refresh_value] = exact_array(&pair, 2, "bounded-depth package pair")? else {
        return Err(PackageError::Malformed("bounded-depth package pair"));
    };
    let light = StepPackage::decode(light_value)?;
    let refresh = StepPackage::decode(refresh_value)?;
    if light.circuit != 0 || refresh.circuit != 1 {
        return Err(PackageError::Malformed("bounded-depth circuit indices"));
    }
    Ok((pair, light, refresh))
}

/// `StepPackage.identity`: Poseidon2 over the domain words and the canonical
/// preimage of the package pair.
pub fn bounded_depth_identity(pair: &Value) -> Result<[u64; 4], PackageError> {
    let mut input = poseidon2::Poseidon2Hasher::default();
    for &byte in IDENTITY_DOMAIN {
        input.update(&[Goldilocks::from_u64(u64::from(byte))]);
    }
    append_value_preimage(pair, &mut input)?;
    Ok(input.finalize().map(|word| word.as_canonical_u64()))
}

/// Reads Lean's rendering of a `Codec.Value`: arrays and natural numbers. It
/// keeps an explicit stack, because expressions nest deeper than serde_json's
/// default recursion limit.
pub fn parse_atoms(bytes: &[u8]) -> Result<Value, PackageError> {
    let mut stack: Vec<Vec<Value>> = Vec::new();
    let mut done = None;
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        match byte {
            b'[' => stack.push(Vec::new()),
            b']' => {
                let finished = Value::Array(
                    stack
                        .pop()
                        .ok_or(PackageError::Malformed("unbalanced array"))?,
                );
                match stack.last_mut() {
                    Some(parent) => parent.push(finished),
                    None if done.is_none() => done = Some(finished),
                    None => return Err(PackageError::Malformed("trailing value")),
                }
            }
            b',' | b' ' | b'\n' | b'\r' | b'\t' => {}
            b'0'..=b'9' => {
                let end = bytes[index..]
                    .iter()
                    .position(|byte| !byte.is_ascii_digit())
                    .map_or(bytes.len(), |offset| index + offset);
                let number = std::str::from_utf8(&bytes[index..end])
                    .ok()
                    .and_then(|digits| digits.parse::<u64>().ok())
                    .ok_or(PackageError::Malformed("package atom"))?;
                stack
                    .last_mut()
                    .ok_or(PackageError::Malformed("atom outside an array"))?
                    .push(Value::from(number));
                index = end;
                continue;
            }
            _ => return Err(PackageError::Malformed("package byte")),
        }
        index += 1;
    }
    if !stack.is_empty() {
        return Err(PackageError::Malformed("unbalanced array"));
    }
    done.ok_or(PackageError::Malformed("empty package"))
}

fn field(value: &Value, location: &'static str) -> Result<Goldilocks, PackageError> {
    let word = value.as_u64().ok_or(PackageError::Malformed(location))?;
    if word >= Goldilocks::ORDER_U64 {
        return Err(PackageError::NonCanonicalField { location, value: word });
    }
    Ok(Goldilocks::from_u64(word))
}

/// `(column, coefficient)` pairs.
fn decode_terms(value: &Value) -> Result<Vec<(usize, Goldilocks)>, PackageError> {
    array(value, "bounded-depth terms")?
        .iter()
        .map(|term| {
            let pair = exact_array(term, 2, "bounded-depth term")?;
            Ok((
                usize_atom(&pair[0], "bounded-depth term column")?,
                field(&pair[1], "bounded-depth term coefficient")?,
            ))
        })
        .collect()
}

/// `StepPackage.combinationFormat`: the constant and the terms.
fn decode_combination(value: &Value) -> Result<Linear, PackageError> {
    let fields = exact_array(value, 2, "bounded-depth combination")?;
    Ok(Linear {
        constant: field(&fields[0], "bounded-depth combination constant")?,
        terms: decode_terms(&fields[1])?,
    })
}

fn decode_row(value: &Value) -> Result<Row, PackageError> {
    let fields = exact_array(value, 3, "bounded-depth row")?;
    Ok(Row {
        a: decode_combination(&fields[0])?,
        b: decode_combination(&fields[1])?,
        c: decode_combination(&fields[2])?,
    })
}

/// `Pieces.linear?` on an encoded expression. A product is linear only when
/// one factor is a constant node; a lane that is not linear is rejected,
/// because a package's blocks and ring pieces pack only with linear lanes.
fn decode_linear(value: &Value) -> Result<Linear, PackageError> {
    let fields = array(value, "bounded-depth expression")?;
    let tag = fields
        .first()
        .and_then(Value::as_u64)
        .ok_or(PackageError::Malformed("bounded-depth expression tag"))?;
    match (tag, fields) {
        (0, [_, column]) => Ok(Linear {
            constant: Goldilocks::ZERO,
            terms: vec![(usize_atom(column, "bounded-depth expression column")?, Goldilocks::ONE)],
        }),
        (1, [_, constant]) => Ok(Linear {
            constant: field(constant, "bounded-depth expression constant")?,
            terms: Vec::new(),
        }),
        (2, [_, left, right]) => Ok(decode_linear(left)?.plus(decode_linear(right)?)),
        (3, [_, left, right]) => match (constant_node(left)?, constant_node(right)?) {
            (Some(factor), _) => Ok(decode_linear(right)?.scale(factor)),
            (None, Some(factor)) => Ok(decode_linear(left)?.scale(factor)),
            (None, None) => Err(PackageError::Malformed("bounded-depth lane is not linear")),
        },
        _ => Err(PackageError::Malformed("bounded-depth expression")),
    }
}

fn constant_node(value: &Value) -> Result<Option<Goldilocks>, PackageError> {
    match array(value, "bounded-depth expression")? {
        [tag, constant] if tag.as_u64() == Some(1) => Ok(Some(field(constant, "bounded-depth expression constant")?)),
        _ => Ok(None),
    }
}

fn decode_lanes(value: &Value) -> Result<Vec<Linear>, PackageError> {
    array(value, "bounded-depth lanes")?
        .iter()
        .map(decode_linear)
        .collect()
}

fn decode_piece(value: &Value) -> Result<Piece, PackageError> {
    let fields = array(value, "bounded-depth piece")?;
    match (fields.first().and_then(Value::as_u64), fields) {
        (Some(0), [_, rows]) => Ok(Piece::Rows(
            array(rows, "bounded-depth rows")?
                .iter()
                .map(decode_row)
                .collect::<Result<_, _>>()?,
        )),
        (Some(1), [_, cells, start, quotient, lanes]) => Ok(Piece::Ring {
            cells: usize_atom(cells, "bounded-depth ring cells")?,
            start: usize_atom(start, "bounded-depth ring start")?,
            quotient: usize_atom(quotient, "bounded-depth ring quotient")?,
            lanes: decode_lanes(lanes)?,
        }),
        (Some(2), [_, start, lanes]) => Ok(Piece::Block {
            start: usize_atom(start, "bounded-depth block start")?,
            lanes: decode_lanes(lanes)?,
        }),
        _ => Err(PackageError::Malformed("bounded-depth piece")),
    }
}
