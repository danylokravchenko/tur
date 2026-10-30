//! Whisper's timestamp tokens, as Transformers' `WhisperTimeStampLogitsProcessor` constrains
//! them and its `_retrieve_segment` turns them into segments and the next seek.

/// Seconds per timestamp step.
pub(super) const TIME_PRECISION: f64 = 0.02;
/// Feature frames per timestamp step.
pub(super) const FRAMES_PER_STEP: usize = 2;

/// Timestamp tokens and their limits in one vocabulary.
#[derive(Clone, Copy, Debug)]
pub(super) struct TimestampRules {
    /// `<|0.00|>`: this token and every later one is a timestamp.
    pub(super) begin: u32,
    pub(super) no_timestamps: u32,
    pub(super) eos: u32,
    /// Latest step a window's first timestamp may name.
    pub(super) max_initial: Option<u32>,
}

impl TimestampRules {
    fn is_timestamp(&self, token: u32) -> bool {
        token >= self.begin
    }

    /// Restrict `row` given the tokens `sampled` after the prompt: timestamps come in
    /// non-decreasing pairs (a lone one only before end of text), a window opens with one,
    /// and when timestamps together outweigh every text token, one is chosen.
    pub(super) fn constrain(&self, row: &mut [f32], sampled: &[u32]) {
        let begin = (self.begin as usize).min(row.len());
        let eos = (self.eos as usize).min(row.len());
        if let Some(slot) = row.get_mut(self.no_timestamps as usize) {
            *slot = f32::NEG_INFINITY;
        }
        let last = sampled.last().is_some_and(|&t| self.is_timestamp(t));
        let penultimate = sampled.len() < 2 || self.is_timestamp(sampled[sampled.len() - 2]);
        if last {
            if penultimate {
                row[begin..].fill(f32::NEG_INFINITY);
            } else {
                row[..eos].fill(f32::NEG_INFINITY);
            }
        }
        if let Some(&newest) = sampled.iter().rfind(|&&t| self.is_timestamp(t)) {
            // An opening timestamp may repeat as its closing one, and nothing else may.
            let floor = if last && !penultimate {
                newest
            } else {
                newest.saturating_add(1)
            };
            let floor = (floor as usize).min(row.len());
            row[begin..floor].fill(f32::NEG_INFINITY);
        }
        if sampled.is_empty() {
            row[..begin].fill(f32::NEG_INFINITY);
            if let Some(max) = self.max_initial {
                let first_banned = (self.begin as usize)
                    .saturating_add(max as usize)
                    .saturating_add(1);
                let first_banned = first_banned.min(row.len());
                row[first_banned..].fill(f32::NEG_INFINITY);
            }
        }
        // The softmax normalizer cancels from both sides of the comparison.
        let (text, timestamps) = row.split_at(begin);
        let best_text = text.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if log_sum_exp(timestamps) > f64::from(best_text) {
            row[..begin].fill(f32::NEG_INFINITY);
        }
    }

    /// Split one window's `sequence` (without its prompt or end of text) of `frames`
    /// feature frames into segments, and return how many frames the next window advances.
    /// With consecutive timestamps, text after the last closed segment is dropped and the
    /// next window starts at that timestamp, so a word at the window's edge is decoded
    /// again whole. A lone final timestamp, or no segment break at all, consumes the window.
    pub(super) fn split(&self, sequence: &[u32], frames: usize) -> (Vec<WindowSegment>, usize) {
        let timestamp = |index: usize| self.is_timestamp(sequence[index]);
        let single_ending =
            sequence.len() >= 2 && !timestamp(sequence.len() - 2) && timestamp(sequence.len() - 1);
        let mut slices: Vec<usize> = (1..sequence.len())
            .filter(|&i| timestamp(i - 1) && timestamp(i))
            .collect();
        let step = |token: u32| token.saturating_sub(self.begin) as usize;
        let Some(last) = slices.last_mut() else {
            // Without a break the whole decoding is one segment, closed by its last timestamp.
            let end = sequence
                .iter()
                .rfind(|&&t| self.is_timestamp(t))
                .map(|&t| step(t))
                .filter(|&s| s != 0)
                .unwrap_or(frames / FRAMES_PER_STEP);
            let segment = WindowSegment {
                tokens: sequence.to_vec(),
                start: 0,
                end,
            };
            return (vec![segment], frames);
        };
        if single_ending {
            slices.push(sequence.len());
        } else {
            // Keeps the closing timestamp in the last segment.
            *last += 1;
        }
        let mut segments = Vec::with_capacity(slices.len());
        let mut from = 0;
        for (index, &to) in slices.iter().enumerate() {
            let tokens = &sequence[from..to];
            let closing = if index + 1 == slices.len() && !single_ending {
                tokens.len().saturating_sub(2)
            } else {
                tokens.len().saturating_sub(1)
            };
            segments.push(WindowSegment {
                tokens: tokens.to_vec(),
                start: step(tokens[0]),
                end: step(tokens[closing]),
            });
            from = to;
        }
        let advance = if single_ending {
            frames
        } else {
            step(sequence[from - 2]) * FRAMES_PER_STEP
        };
        // A window closing only at its first instant would decode the same frames forever.
        (segments, if advance == 0 { frames } else { advance })
    }
}

/// One segment of a decoded window, in timestamp steps from the window's start.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct WindowSegment {
    /// Text and timestamp tokens.
    pub(super) tokens: Vec<u32>,
    pub(super) start: usize,
    pub(super) end: usize,
}

fn log_sum_exp(values: &[f32]) -> f64 {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let max = f64::from(max);
    max + values
        .iter()
        .map(|&v| (f64::from(v) - max).exp())
        .sum::<f64>()
        .ln()
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "the rules leave logits untouched or replace them with infinity exactly"
)]
mod tests {
    use super::*;

    const RULES: TimestampRules = TimestampRules {
        begin: 10,
        no_timestamps: 9,
        eos: 8,
        max_initial: Some(2),
    };
    const NO: f32 = f32::NEG_INFINITY;

    /// `row` after the rules, one logit per token of a 20-token vocabulary.
    fn constrained(sampled: &[u32], row: &mut [f32; 20]) -> [f32; 20] {
        RULES.constrain(row, sampled);
        *row
    }

    #[test]
    fn test_a_window_opens_with_a_timestamp_within_the_initial_limit() {
        let row = constrained(&[], &mut [1.0; 20]);
        assert!(row[..10].iter().all(|&v| v == NO));
        assert!(row[10..13].iter().all(|&v| v == 1.0));
        assert!(row[13..].iter().all(|&v| v == NO));
    }

    #[test]
    fn test_timestamps_pair_up_and_never_decrease() {
        // Text after a timestamp: a closing timestamp at or above it, or more text.
        let row = constrained(&[12, 3], &mut [0.0; 20]);
        assert_eq!(row[12], NO);
        assert_eq!(row[13], 0.0);
        // A lone timestamp closes a segment: only end of text or a timestamp may follow,
        // and the closing timestamp may repeat the one before it.
        let mut likely_end = [0.0; 20];
        likely_end[8] = 5.0;
        let row = constrained(&[12, 3, 14], &mut likely_end);
        assert!(row[..8].iter().all(|&v| v == NO));
        assert_eq!(row[8], 5.0);
        assert_eq!(row[13], NO);
        assert_eq!(row[14], 0.0);
        // After a pair the next token is text.
        let mut heavy = [0.0; 20];
        heavy[3] = 5.0;
        let row = constrained(&[12, 3, 14, 14], &mut heavy);
        assert!(row[10..].iter().all(|&v| v == NO));
        assert_eq!(row[3], 5.0);
    }

    #[test]
    fn test_timestamps_win_when_their_total_mass_beats_every_text_token() {
        // No single timestamp beats the text token, but the eight allowed together do.
        let mut row = [-100.0; 20];
        row[3] = 1.0;
        row[11..19].fill(0.0);
        let row = constrained(&[10, 3], &mut row);
        assert_eq!(row[3], NO);
        assert_eq!(row[12], 0.0);
        // Below the text token they leave it alone.
        let mut row = [-100.0; 20];
        row[3] = 4.0;
        row[11..19].fill(0.0);
        assert_eq!(constrained(&[10, 3], &mut row)[3], 4.0);
    }

    #[test]
    fn test_pairs_become_segments_and_seek_to_the_last_closed_one() {
        // <0> a <5> <5> b <9> <9> c: `c` is unfinished, so the next window starts at step 9.
        let (segments, advance) = RULES.split(&[10, 1, 15, 15, 2, 19, 19, 3], 3000);
        let expected = [
            WindowSegment {
                tokens: vec![10, 1, 15],
                start: 0,
                end: 5,
            },
            WindowSegment {
                tokens: vec![15, 2, 19, 19],
                start: 5,
                end: 9,
            },
        ];
        assert_eq!(segments, expected);
        assert_eq!(advance, 9 * FRAMES_PER_STEP);
    }

    #[test]
    fn test_a_lone_final_timestamp_consumes_the_window() {
        let (segments, advance) = RULES.split(&[10, 1, 15, 15, 2, 19], 3000);
        assert_eq!(segments.len(), 2);
        assert_eq!((segments[1].start, segments[1].end), (5, 9));
        assert_eq!(advance, 3000);
    }

    #[test]
    fn test_windows_without_breaks_form_one_segment_and_always_advance() {
        // No timestamps at all: the segment spans the window.
        let (segments, advance) = RULES.split(&[1, 2, 3], 2400);
        assert_eq!((segments[0].start, segments[0].end), (0, 1200));
        assert_eq!(advance, 2400);
        // One timestamp names the end, and step zero does not.
        assert_eq!(RULES.split(&[1, 17], 3000).0[0].end, 7);
        assert_eq!(RULES.split(&[10, 1], 3000).0[0].end, 1500);
        // A pair at step zero must not seek back to where it began.
        assert_eq!(RULES.split(&[10, 10, 1], 3000).1, 3000);
    }
}
