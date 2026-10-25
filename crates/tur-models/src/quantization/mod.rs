//! Shared offline calibration statistics and layer-wise conversion infrastructure.

mod calibration;
pub(crate) use calibration::Recorder;
pub use calibration::{Calibration, Site};

use candle_core::{DType, Device, Tensor};
use std::{
    fs::File,
    io::{Read, Seek, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
use tur_core::{
    Result,
    execution::{checked_elements, invalid},
};

static NEXT: AtomicU64 = AtomicU64::new(0);

/// A loaded reference layer and its input-statistics recorder. Implementations own
/// model-specific execution and must fence work before releasing device resources.
pub(crate) trait LayerCalibration {
    type Statistics;

    fn forward(&self, input: Tensor) -> Result<Tensor>;
    fn finish(self) -> Result<Self::Statistics>;
}

/// Consume layers lazily, freeing each layer and consuming its statistics before
/// loading the next. Embeddings and hidden states are streamed one sample at a time.
/// The shared workflow makes no assumptions about decoder architecture or statistics.
pub(crate) fn calibrate_layers<L: LayerCalibration>(
    embeddings: impl IntoIterator<Item = Result<Tensor>>,
    layers: impl IntoIterator<Item = Result<L>>,
    device: &Device,
    scratch: &Path,
    mut consume: impl FnMut(&L::Statistics) -> Result<()>,
) -> Result<()> {
    let mut states = Activations::create(scratch)?;
    let mut shapes = Vec::new();
    for embedding in embeddings {
        let embedding = embedding?;
        shapes.push(embedding.dims2()?);
        states.write(&embedding)?;
    }
    for layer in layers {
        let layer = layer?;
        states.advance()?;
        for &(rows, cols) in &shapes {
            let input = states.read(rows, cols, device)?;
            let result = layer.forward(input);
            device.synchronize()?;
            let output = result?;
            if output.dims2()? != (rows, cols) {
                return Err(invalid("calibration layer changed hidden-state shape"));
            }
            states.write(&output)?;
        }
        let statistics = layer.finish()?;
        consume(&statistics)?;
    }
    states.remove()
}

/// Two alternating files bound disk usage to two sets of hidden states. Read and write
/// handles never refer to the same file during a layer pass.
struct Activations {
    dir: PathBuf,
    input: File,
    output: File,
}

impl Activations {
    fn create(scratch: &Path) -> Result<Self> {
        let dir = loop {
            let dir = scratch.join(format!(
                ".tur-calibration-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match std::fs::create_dir(&dir) {
                Ok(()) => break dir,
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(e) => return Err(e.into()),
            }
        };
        let files = (|| -> Result<_> {
            Ok((
                File::create_new(dir.join("a"))?,
                File::create_new(dir.join("b"))?,
            ))
        })();
        match files {
            Ok((input, output)) => Ok(Self { dir, input, output }),
            Err(e) => {
                // Preserve the creation error; no caller can use this scratch directory.
                drop(std::fs::remove_dir_all(dir));
                Err(e)
            }
        }
    }

    fn advance(&mut self) -> Result<()> {
        std::mem::swap(&mut self.input, &mut self.output);
        self.input.rewind()?;
        self.output.set_len(0)?;
        self.output.rewind()?;
        Ok(())
    }

    fn write(&mut self, x: &Tensor) -> Result<()> {
        let values = x
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.output.write_all(&bytes)?;
        Ok(())
    }

    fn read(&mut self, rows: usize, cols: usize, device: &Device) -> Result<Tensor> {
        let size = checked_elements(&[rows, cols])?
            .checked_mul(4)
            .ok_or_else(|| invalid("calibration activation size overflow"))?;
        let mut bytes = vec![0; size];
        self.input.read_exact(&mut bytes)?;
        Ok(Tensor::from_raw_buffer(
            &bytes,
            DType::F32,
            &[rows, cols],
            device,
        )?)
    }

    fn remove(self) -> Result<()> {
        std::fs::remove_dir_all(&self.dir)?;
        Ok(())
    }
}

impl Drop for Activations {
    fn drop(&mut self) {
        // Best-effort cleanup on errors; successful conversion reports cleanup failures.
        drop(std::fs::remove_dir_all(&self.dir));
    }
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;
    use std::{cell::Cell, rc::Rc};

    struct Layer {
        live: Rc<Cell<usize>>,
        seen: Cell<usize>,
        samples: Cell<usize>,
        index: usize,
    }

    impl LayerCalibration for Layer {
        type Statistics = usize;

        fn forward(&self, input: Tensor) -> Result<Tensor> {
            // Different sample lengths and repeated inputs must survive disk round trips.
            let values = input.flatten_all()?.to_vec1::<f32>()?;
            let expected =
                2.0f32.mul_add(self.index as f32, [3.0f32, 5.0, 7.0][self.samples.get()]);
            assert!(values.iter().all(|v| v.to_bits() == expected.to_bits()));
            self.samples.set(self.samples.get() + 1);
            self.seen.set(self.seen.get() + input.dim(0)?);
            Ok((input + 2.0)?)
        }

        fn finish(self) -> Result<usize> {
            Ok(self.seen.get())
        }
    }

    impl Drop for Layer {
        fn drop(&mut self) {
            self.live.set(self.live.get() - 1);
        }
    }

    #[test]
    fn test_streaming_releases_layers_before_consuming_statistics_and_loading_next() -> Result<()> {
        let scratch = Activations::create(&std::env::temp_dir())?;
        let live = Rc::new(Cell::new(0));
        let consumed = Cell::new(0);
        let embeddings = [(1, 3.0f32), (5, 5.0), (2, 7.0)]
            .into_iter()
            .map(|(rows, value)| Ok(Tensor::full(value, (rows, 4), &Device::Cpu)?));
        let layers = (0..4).map(|index| {
            assert_eq!(live.get(), 0);
            assert_eq!(consumed.get(), index);
            live.set(1);
            Ok(Layer {
                live: Rc::clone(&live),
                seen: Cell::new(0),
                samples: Cell::new(0),
                index,
            })
        });
        calibrate_layers(embeddings, layers, &Device::Cpu, &scratch.dir, |&tokens| {
            assert_eq!(tokens, 8);
            assert_eq!(live.get(), 0);
            consumed.set(consumed.get() + 1);
            Ok(())
        })?;
        assert_eq!(consumed.get(), 4);
        assert_eq!(std::fs::read_dir(&scratch.dir)?.count(), 2);
        scratch.remove()
    }

    #[test]
    fn test_streaming_cleans_activations_and_stops_loading_after_consumer_failure() -> Result<()> {
        let scratch = Activations::create(&std::env::temp_dir())?;
        let live = Rc::new(Cell::new(0));
        let loaded = Cell::new(0);
        let embeddings = [Ok(Tensor::full(3.0f32, (2, 4), &Device::Cpu)?)];
        let layers = (0..4).map(|_| {
            loaded.set(loaded.get() + 1);
            live.set(live.get() + 1);
            Ok(Layer {
                live: Rc::clone(&live),
                seen: Cell::new(0),
                samples: Cell::new(0),
                index: 0,
            })
        });
        let error = calibrate_layers(embeddings, layers, &Device::Cpu, &scratch.dir, |_| {
            Err(invalid("injected packing failure"))
        });
        assert!(matches!(
            error,
            Err(tur_core::TurError::InvalidExecution(_))
        ));
        assert_eq!(loaded.get(), 1);
        assert_eq!(live.get(), 0);
        assert_eq!(std::fs::read_dir(&scratch.dir)?.count(), 2);
        scratch.remove()
    }
}
