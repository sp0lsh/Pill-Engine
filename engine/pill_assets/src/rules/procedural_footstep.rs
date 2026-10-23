use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::util::write_atomic;
use crate::Rule;

const SAMPLE_RATE: u32 = 22_050;

/// Footstep sound synthesized in code from a `.procedural_footstep` spec → `{stem}.cooked.wav`
/// (16-bit mono PCM): a dull heel thump (low-passed noise and a low knock), then a softer,
/// brighter toe scuff. The spec is a JSON object of [`FootstepParams`] fields; missing ones keep
/// their defaults, unknown ones are an error.
pub struct ProceduralFootstep;

impl Rule for ProceduralFootstep {
    fn name(&self) -> &'static str {
        "procedural_footstep"
    }

    fn input_glob(&self) -> &'static str {
        "**/*.procedural_footstep"
    }

    fn output_for(&self, input: &Path) -> PathBuf {
        let stem = input.file_stem().unwrap().to_str().unwrap();
        input.with_file_name(format!("{stem}.cooked.wav"))
    }

    fn build(&self, input: &Path, output: &Path) -> Result<()> {
        let text = std::fs::read_to_string(input).with_context(|| format!("read {input:?}"))?;
        let params = FootstepParams::parse(&text).with_context(|| format!("{input:?}"))?;
        write_atomic(output, &wav_bytes(&footstep(&params)))
    }
}

/// Synthesis parameters of one footstep (times in seconds, frequencies in Hz).
#[derive(Clone, Debug, PartialEq)]
pub struct FootstepParams {
    /// Noise seed: same seed and values, same sound.
    pub seed: u64,
    pub length: f32,
    /// Low-pass cutoff of the heel's noise (lower = duller).
    pub heel_cutoff: f32,
    /// Heel decay time constant.
    pub heel_decay: f32,
    /// Pitch and level of the low knock under the heel.
    pub knock_frequency: f32,
    pub knock_level: f32,
    /// When the toe scuff starts after the heel, its decay and level relative to the heel.
    pub toe_delay: f32,
    pub toe_decay: f32,
    pub toe_level: f32,
    /// High cutoff of the toe's scuff (higher = brighter, grittier).
    pub toe_cutoff: f32,
    /// Peak sample after normalizing (0..1).
    pub peak: f32,
}

impl Default for FootstepParams {
    fn default() -> Self {
        Self {
            seed: 1,
            length: 0.16,
            heel_cutoff: 460.0,
            heel_decay: 0.028,
            knock_frequency: 80.0,
            knock_level: 0.5,
            toe_delay: 0.045,
            toe_decay: 0.035,
            toe_level: 0.55,
            toe_cutoff: 3200.0,
            peak: 0.6,
        }
    }
}

impl FootstepParams {
    /// Parses a JSON object of parameters (an empty file: all defaults).
    pub fn parse(text: &str) -> Result<Self> {
        let mut params = Self::default();
        if text.trim().is_empty() {
            return Ok(params);
        }
        let spec: serde_json::Value = serde_json::from_str(text).context("invalid JSON")?;
        let Some(fields) = spec.as_object() else {
            bail!("expected a JSON object of parameters");
        };
        for (key, value) in fields {
            let number = || {
                value
                    .as_f64()
                    .map(|number| number as f32)
                    .with_context(|| format!("`{key}`: expected a number, got {value}"))
            };
            match key.as_str() {
                "seed" => {
                    params.seed = value
                        .as_u64()
                        .with_context(|| format!("`seed`: expected an integer, got {value}"))?
                }
                "length" => params.length = number()?,
                "heel_cutoff" => params.heel_cutoff = number()?,
                "heel_decay" => params.heel_decay = number()?,
                "knock_frequency" => params.knock_frequency = number()?,
                "knock_level" => params.knock_level = number()?,
                "toe_delay" => params.toe_delay = number()?,
                "toe_decay" => params.toe_decay = number()?,
                "toe_level" => params.toe_level = number()?,
                "toe_cutoff" => params.toe_cutoff = number()?,
                "peak" => params.peak = number()?,
                _ => bail!("unknown parameter `{key}`"),
            }
        }
        if !(0.02..=2.0).contains(&params.length) {
            bail!("length {} s out of 0.02..2", params.length);
        }
        Ok(params)
    }
}

/// Deterministic uniform random numbers (xorshift64*).
struct Random(u64);

impl Random {
    fn signed(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let value = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d);
        (value >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

fn one_pole_lowpass(samples: &[f32], cutoff: f32) -> Vec<f32> {
    let alpha = 1.0 - (-2.0 * std::f32::consts::PI * cutoff / SAMPLE_RATE as f32).exp();
    let mut state = 0.0;
    samples
        .iter()
        .map(|&sample| {
            state += alpha * (sample - state);
            state
        })
        .collect()
}

/// One footstep, samples in −peak..peak.
fn footstep(params: &FootstepParams) -> Vec<f32> {
    let mut random = Random(params.seed | 1);
    let count = (SAMPLE_RATE as f32 * params.length) as usize;
    let noise: Vec<f32> = (0..count).map(|_| random.signed()).collect();
    // Heel: twice low-passed noise and a low knock, 2 ms attack
    let heel_noise = one_pole_lowpass(
        &one_pole_lowpass(&noise, params.heel_cutoff),
        params.heel_cutoff * 1.3,
    );
    // Toe: the noise minus its low part, softened
    let low = one_pole_lowpass(&noise, 900.0);
    let toe_noise: Vec<f32> = noise.iter().zip(&low).map(|(a, b)| a - b).collect();
    let toe_noise = one_pole_lowpass(&toe_noise, params.toe_cutoff);

    let mut samples: Vec<f32> = (0..count)
        .map(|index| {
            let time = index as f32 / SAMPLE_RATE as f32;
            let attack = (time / 0.002).min(1.0);
            let knock = (2.0 * std::f32::consts::PI * params.knock_frequency * time).sin();
            let heel = attack
                * (-time / params.heel_decay).exp()
                * (3.2 * heel_noise[index] + params.knock_level * knock);
            let toe_time = time - params.toe_delay;
            let toe = if toe_time > 0.0 {
                (toe_time / 0.004).min(1.0)
                    * (-toe_time / params.toe_decay).exp()
                    * params.toe_level
                    * toe_noise[index]
            } else {
                0.0
            };
            heel + toe
        })
        .collect();
    // 10 ms fade-out, then normalize
    let fade = ((SAMPLE_RATE as f32 * 0.01) as usize).min(count);
    for index in 0..fade {
        samples[count - 1 - index] *= index as f32 / fade as f32;
    }
    let peak = samples
        .iter()
        .fold(0.0f32, |peak, sample| peak.max(sample.abs()))
        .max(f32::EPSILON);
    samples
        .iter()
        .map(|sample| sample / peak * params.peak)
        .collect()
}

/// RIFF/WAVE, 16-bit mono PCM at `SAMPLE_RATE`.
fn wav_bytes(samples: &[f32]) -> Vec<u8> {
    let data_size = (samples.len() * 2) as u32;
    let mut bytes = Vec::with_capacity(44 + data_size as usize);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_size).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
    bytes.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    bytes.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes()); // byte rate
    bytes.extend_from_slice(&2u16.to_le_bytes()); // block align
    bytes.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_size.to_le_bytes());
    for sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_spec_uses_defaults() {
        assert_eq!(
            FootstepParams::parse("").unwrap(),
            FootstepParams::default()
        );
    }

    #[test]
    fn json_values_parsed_missing_ones_default() {
        let params =
            FootstepParams::parse(r#"{ "seed": 7, "heel_cutoff": 380, "toe_level": 0.8 }"#)
                .unwrap();
        assert_eq!(params.seed, 7);
        assert_eq!(params.heel_cutoff, 380.0);
        assert_eq!(params.toe_level, 0.8);
        assert_eq!(
            params.knock_frequency,
            FootstepParams::default().knock_frequency
        );
    }

    #[test]
    fn unknown_key_or_bad_value_rejected() {
        let unknown = FootstepParams::parse(r#"{ "seed": 1, "loudness": 3 }"#).unwrap_err();
        assert_eq!(unknown.to_string(), "unknown parameter `loudness`");
        let bad = FootstepParams::parse(r#"{ "toe_delay": "soon" }"#).unwrap_err();
        assert!(bad
            .to_string()
            .starts_with("`toe_delay`: expected a number"));
    }

    #[test]
    fn wav_header_and_length_follow_params() {
        let params = FootstepParams {
            length: 0.2,
            ..FootstepParams::default()
        };
        let bytes = wav_bytes(&footstep(&params));
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..16], b"WAVEfmt ");
        assert_eq!(
            u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
            SAMPLE_RATE
        );
        assert_eq!(bytes.len(), 44 + (SAMPLE_RATE as f32 * 0.2) as usize * 2);
    }

    #[test]
    fn peak_normalized_and_ends_silent() {
        let samples = footstep(&FootstepParams::default());
        let peak = samples.iter().fold(0.0f32, |peak, s| peak.max(s.abs()));
        assert!((peak - 0.6).abs() < 1e-4, "peak {peak}");
        assert_eq!(*samples.last().unwrap(), 0.0);
    }

    #[test]
    fn same_params_same_sound_other_seed_different() {
        let params = FootstepParams::default();
        assert_eq!(footstep(&params), footstep(&params));
        let other = FootstepParams {
            seed: 2,
            ..params.clone()
        };
        assert_ne!(footstep(&params), footstep(&other));
    }

    #[test]
    fn output_named_after_spec() {
        let output =
            ProceduralFootstep.output_for(Path::new("res/audio/footstep3.procedural_footstep"));
        assert_eq!(output, Path::new("res/audio/footstep3.cooked.wav"));
    }
}
