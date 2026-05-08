use serde::{Deserialize, Serialize};

use super::super::ivc::{DirectCcsFPrimeSnarkError, DirectCcsIvcState};
use super::super::public_image::DirectCcsIvcPublicImage;
use super::image::DirectCcsCompactFPrimeImage;
use super::low_norm::DirectCcsFPrimeLowNormSourceImage;
use crate::construction2::{Construction2EncodedPublicInput, Construction2PublicBoundary};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectCcsNativeFPrimeAdvice {
    compact_image: DirectCcsCompactFPrimeImage,
    construction2_u_in: Construction2PublicBoundary,
    construction2_u_out: Construction2PublicBoundary,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectCcsNativeFPrimeStepImage {
    compact_image: DirectCcsCompactFPrimeImage,
    construction2_u_out: Construction2PublicBoundary,
    terminal_public_image: DirectCcsIvcPublicImage,
}

impl DirectCcsNativeFPrimeAdvice {
    pub fn from_latest_state(state: &DirectCcsIvcState) -> Result<Self, DirectCcsFPrimeSnarkError> {
        let compact_image = DirectCcsCompactFPrimeImage::from_latest_state(state)?;
        let last = state.last_step.as_ref().ok_or_else(|| {
            DirectCcsFPrimeSnarkError::Input("direct native F' advice requires an appended step".into())
        })?;
        let advice = Self {
            compact_image,
            construction2_u_in: Construction2PublicBoundary::from_fresh_instance(&last.construction2_u_i),
            construction2_u_out: Construction2PublicBoundary::from_fresh_instance(&state.construction2_u_i),
        };
        advice.validate()?;
        Ok(advice)
    }

    pub fn compact_image(&self) -> &DirectCcsCompactFPrimeImage {
        &self.compact_image
    }

    pub fn construction2_u_in(&self) -> &Construction2PublicBoundary {
        &self.construction2_u_in
    }

    pub fn construction2_u_out(&self) -> &Construction2PublicBoundary {
        &self.construction2_u_out
    }

    pub fn low_norm_source_image(&self) -> Result<DirectCcsFPrimeLowNormSourceImage, DirectCcsFPrimeSnarkError> {
        DirectCcsFPrimeLowNormSourceImage::from_native_advice(self)
    }

    pub fn evaluate(&self) -> Result<DirectCcsNativeFPrimeStepImage, DirectCcsFPrimeSnarkError> {
        self.validate()?;
        Ok(DirectCcsNativeFPrimeStepImage {
            compact_image: self.compact_image.clone(),
            terminal_public_image: self
                .compact_image
                .terminal_public_image(self.construction2_u_out.clone()),
            construction2_u_out: self.construction2_u_out.clone(),
        })
    }

    pub(super) fn validate(&self) -> Result<(), DirectCcsFPrimeSnarkError> {
        self.compact_image.validate()?;
        validate_construction2_boundary_digest(
            &self.construction2_u_in,
            &self.compact_image.x_in,
            self.compact_image.construction2_u_in_digest,
            "input",
        )?;
        validate_construction2_boundary_digest(
            &self.construction2_u_out,
            &self.compact_image.x_out,
            self.compact_image.construction2_u_out_digest,
            "output",
        )
    }
}

impl DirectCcsNativeFPrimeStepImage {
    pub fn compact_image(&self) -> &DirectCcsCompactFPrimeImage {
        &self.compact_image
    }

    pub fn construction2_u_out(&self) -> &Construction2PublicBoundary {
        &self.construction2_u_out
    }

    pub fn terminal_public_image(&self) -> &DirectCcsIvcPublicImage {
        &self.terminal_public_image
    }
}

fn validate_construction2_boundary_digest(
    boundary: &Construction2PublicBoundary,
    expected_x_i: &Construction2EncodedPublicInput,
    expected_instance_digest: [u8; 32],
    role: &str,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    if !boundary.has_canonical_commitment_shape()
        || boundary.commitment_digest != boundary.expected_commitment_digest()
        || boundary.fresh_instance_digest != boundary.expected_fresh_instance_digest()
        || &boundary.x_i != expected_x_i
    {
        return Err(DirectCcsFPrimeSnarkError::Input(format!(
            "direct native F' advice {role} Construction-2 boundary is inconsistent"
        )));
    }
    if boundary.fresh_instance_digest != expected_instance_digest {
        return Err(DirectCcsFPrimeSnarkError::Input(format!(
            "direct native F' advice {role} Construction-2 instance digest is inconsistent"
        )));
    }
    Ok(())
}
