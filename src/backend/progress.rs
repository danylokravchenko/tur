use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use parking_lot::RwLock;
use std::io::Write;
use std::sync::Arc;

/// Progress reporter for model loading and token generation
#[derive(Clone)]
pub struct ProgressReporter {
    multi: Arc<MultiProgress>,
    loading_bar: Arc<RwLock<Option<ProgressBar>>>,
    generation_bar: Arc<RwLock<Option<ProgressBar>>>,
    text_buffer: Arc<RwLock<String>>,
}

impl ProgressReporter {
    pub fn new() -> Self {
        Self {
            multi: Arc::new(MultiProgress::new()),
            loading_bar: Arc::new(RwLock::new(None)),
            generation_bar: Arc::new(RwLock::new(None)),
            text_buffer: Arc::new(RwLock::new(String::new())),
        }
    }

    /// Initialize loading progress bar for model weights
    pub fn init_loading(&self, total_layers: usize) {
        let style = ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:60.cyan/blue} {pos:>4}/{len:4} {msg}",
        )
        .unwrap()
        .progress_chars("##-");

        let pb = self.multi.add(ProgressBar::new(total_layers as u64));
        pb.set_style(style);
        pb.set_message("Loading model layers");

        *self.loading_bar.write() = Some(pb);
    }

    /// Update loading progress (called after each layer is loaded)
    pub fn update_loading(&self, current: usize, total: usize) {
        if let Some(pb) = self.loading_bar.read().as_ref() {
            pb.set_position(current as u64);
            if current >= total {
                pb.finish_with_message("Model loaded");
            }
        }
    }

    /// Increment loading progress by 1
    pub fn inc_loading(&self) {
        if let Some(pb) = self.loading_bar.read().as_ref() {
            pb.inc(1);
        }
    }

    /// Finish loading progress
    pub fn finish_loading(&self) {
        if let Some(pb) = self.loading_bar.write().take() {
            pb.finish_with_message("Model loaded");
        }
    }

    /// Initialize generation progress bar
    pub fn init_generation(&self, max_tokens: usize) {
        let style = ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:60.green/blue} {pos:>4}/{len:4} tokens | {per_sec:.2} tok/s | {msg}",
        )
        .unwrap()
        .progress_chars("=>-");

        let pb = self.multi.add(ProgressBar::new(max_tokens as u64));
        pb.set_style(style);
        pb.set_message("Generating");

        *self.generation_bar.write() = Some(pb);
    }

    /// Update generation progress
    pub fn update_generation(&self, tokens_generated: usize) {
        if let Some(pb) = self.generation_bar.read().as_ref() {
            pb.set_position(tokens_generated as u64);
        }
    }

    /// Increment generation progress by 1
    pub fn inc_generation(&self) {
        if let Some(pb) = self.generation_bar.read().as_ref() {
            pb.inc(1);
        }
    }

    /// Print a message without interfering with progress bars
    pub fn println(&self, msg: &str) {
        if let Some(pb) = self.generation_bar.read().as_ref() {
            pb.println(msg);
        } else if let Some(pb) = self.loading_bar.read().as_ref() {
            pb.println(msg);
        } else {
            println!("{}", msg);
        }
    }

    /// Print text without newline (for streaming tokens)
    /// Accumulates text and prints complete lines above the progress bar
    pub fn print(&self, text: &str) {
        let mut buffer = self.text_buffer.write();
        buffer.push_str(text);

        // Check if we have complete lines to print
        if let Some(last_newline) = buffer.rfind('\n') {
            let to_print = buffer[..=last_newline].to_string();
            *buffer = buffer[last_newline + 1..].to_string();

            if let Some(pb) = self.generation_bar.read().as_ref() {
                pb.println(&to_print.trim_end_matches('\n'));
            } else if let Some(pb) = self.loading_bar.read().as_ref() {
                pb.println(&to_print.trim_end_matches('\n'));
            } else {
                print!("{}", to_print);
                let _ = std::io::stdout().flush();
            }
        }
    }

    /// Flush any remaining text in the buffer
    pub fn flush_text(&self) {
        let mut buffer = self.text_buffer.write();
        if !buffer.is_empty() {
            let to_print = buffer.clone();
            buffer.clear();

            if let Some(pb) = self.generation_bar.read().as_ref() {
                pb.println(&to_print);
            } else if let Some(pb) = self.loading_bar.read().as_ref() {
                pb.println(&to_print);
            } else {
                print!("{}", to_print);
                let _ = std::io::stdout().flush();
            }
        }
    }

    /// Set a message on the progress bar
    pub fn set_message(&self, msg: String) {
        if let Some(pb) = self.generation_bar.read().as_ref() {
            pb.set_message(msg);
        } else if let Some(pb) = self.loading_bar.read().as_ref() {
            pb.set_message(msg);
        }
    }

    /// Finish generation progress
    pub fn finish_generation(&self, tokens_generated: usize, elapsed_secs: f64) {
        if let Some(pb) = self.generation_bar.write().take() {
            let tok_per_sec = tokens_generated as f64 / elapsed_secs;
            pb.finish_with_message(format!(
                "Generated {} tokens ({:.2} tok/s)",
                tokens_generated, tok_per_sec
            ));
        }
    }

    /// Clear all progress bars
    pub fn clear(&self) {
        let _ = self.multi.clear();
    }
}

impl Default for ProgressReporter {
    fn default() -> Self {
        Self::new()
    }
}

/// A no-op progress reporter that does nothing (for when progress bars are disabled)
#[derive(Clone, Default)]
pub struct NoOpProgressReporter;

impl NoOpProgressReporter {
    pub fn new() -> Self {
        Self
    }

    pub fn init_loading(&self, _total_layers: usize) {}
    pub fn update_loading(&self, _current: usize, _total: usize) {}
    pub fn inc_loading(&self) {}
    pub fn finish_loading(&self) {}
    pub fn init_generation(&self, _max_tokens: usize) {}
    pub fn update_generation(&self, _tokens_generated: usize) {}
    pub fn inc_generation(&self) {}
    pub fn finish_generation(&self, _tokens_generated: usize, _elapsed_secs: f64) {}
    pub fn clear(&self) {}
}
