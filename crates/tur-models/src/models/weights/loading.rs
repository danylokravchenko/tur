//! Shared immutable weight loading and embedding storage.
use super::{Embedding, Packer, Precision, Projection, STAGING_BYTES, WeightPlan};
use crate::models::checkpoint::Checkpoint;
use candle_core::{DType, Device, Tensor};
use std::collections::BTreeMap;
use tur_core::{Result, execution::invalid};
use tur_metal::packed::PackedLinear;

/// Input embedding storage.
pub(crate) enum Embeddings {
    /// A dense F32 or BF16 table.
    Table(Tensor),
    /// A packed table, dequantized one gathered row at a time.
    Packed(PackedLinear),
    /// Rows of the tied LM head: no second table.
    Head,
}

impl Embeddings {
    pub(crate) fn rows(&self, ids: &Tensor, head: &PackedLinear) -> Result<Tensor> {
        match self {
            Self::Table(table) => Ok(table.index_select(ids, 0)?.to_dtype(DType::F32)?),
            Self::Packed(table) => table.rows(ids),
            Self::Head => head.rows(ids),
        }
    }
    pub(crate) fn bytes(&self) -> usize {
        match self {
            Self::Table(t) => t.elem_count() * t.dtype().size_in_bytes(),
            Self::Packed(t) => t.resident_bytes(),
            Self::Head => 0,
        }
    }
}
/// Validated weight source of one load: a `SafeTensors` checkpoint converted by a plan,
/// or a prepacked artifact uploaded as stored.
pub(crate) struct Tensors {
    pub(crate) checkpoint: Checkpoint,
    pub(crate) prepacked: bool,
    pub(crate) plan: WeightPlan,
    pub(crate) projections: BTreeMap<String, Projection>,
    pub(crate) device: Device,
}

impl Tensors {
    /// A small tensor (norm, gate, convolution) on the host in its stored dtype.
    pub(crate) fn take(&self, name: &str) -> Result<Tensor> {
        self.checkpoint.take(name)
    }

    /// Projection `name` on the device in its planned precision.
    pub(crate) fn projection(&self, name: &str) -> Result<PackedLinear> {
        let projection = self
            .projections
            .get(name)
            .ok_or_else(|| invalid(format!("model has no projection {name}")))?;
        let precision = self.plan.precision(name)?;
        let shape = (projection.rows, projection.cols);
        if self.prepacked {
            return self.stored(name, precision, shape);
        }
        Packer::RTN
            .pack(&self.checkpoint, projection, precision)?
            .upload(precision, shape, &self.device)
    }

    /// An artifact matrix uploaded exactly as stored; its layout was validated on open.
    pub(crate) fn stored(
        &self,
        name: &str,
        precision: Precision,
        shape: (usize, usize),
    ) -> Result<PackedLinear> {
        let (entry, bytes) = self.checkpoint.take_raw(name)?;
        match precision.ggml() {
            Some(format) => PackedLinear::from_blocks(&bytes, format, shape, &self.device),
            None => PackedLinear::dense(
                &Tensor::from_raw_buffer(
                    &bytes,
                    entry.candle_dtype(name)?,
                    &entry.shape,
                    &Device::Cpu,
                )?,
                &self.device,
            ),
        }
    }

    /// The input embedding in its planned storage. Dense tables are filled on the device
    /// a staging chunk at a time, so the host never holds the whole table.
    pub(crate) fn embedding(&self, name: &str, rows: usize, cols: usize) -> Result<Embeddings> {
        let name = name.to_owned();
        let precision = match self.plan.embedding {
            Embedding::Head => return Ok(Embeddings::Head),
            Embedding::Table(precision) => precision,
        };
        let dtype = match precision {
            Precision::F32 => DType::F32,
            Precision::Bf16 => DType::BF16,
            _ if self.prepacked => {
                return Ok(Embeddings::Packed(self.stored(
                    &name,
                    precision,
                    (rows, cols),
                )?));
            }
            _ => {
                let table = Projection {
                    name: name.clone(),
                    parts: vec![(name, rows)],
                    rows,
                    cols,
                    experts: 1,
                };
                return Ok(Embeddings::Packed(
                    Packer::RTN
                        .pack(&self.checkpoint, &table, precision)?
                        .upload(precision, (rows, cols), &self.device)?,
                ));
            }
        };
        Ok(Embeddings::Table(self.table(&name, rows, cols, dtype)?))
    }

    /// Split a wide token table into independent column blocks. Gemma's packed PLE table
    /// exceeds the maximum single Metal buffer on smaller Macs; one layer's columns fit.
    pub(crate) fn column_tables(
        &self,
        name: &str,
        rows: usize,
        columns: usize,
        parts: usize,
        dtype: DType,
    ) -> Result<Vec<Tensor>> {
        if parts == 0 || columns == 0 {
            return Err(invalid("empty column embedding table"));
        }
        let tables = (0..parts)
            .map(|_| Ok(Tensor::zeros((rows, columns), dtype, &self.device)?))
            .collect::<Result<Vec<_>>>()?;
        let width = columns
            .checked_mul(parts)
            .ok_or_else(|| invalid("embedding columns overflow"))?;
        let mut start = 0;
        for chunk in self
            .checkpoint
            .take_rows(name, (STAGING_BYTES / (width * 4)).max(1))?
        {
            let chunk = chunk?;
            for (i, table) in tables.iter().enumerate() {
                let slice = chunk
                    .narrow(1, i * columns, columns)?
                    .contiguous()?
                    .to_dtype(dtype)?
                    .to_device(&self.device)?;
                table.slice_set(&slice, 0, start)?;
            }
            start += chunk.dim(0)?;
            // Candle retains uploaded staging buffers until a synchronization. Bound
            // their device residency as well as host staging for multi-gigabyte tables.
            self.device.synchronize()?;
        }
        Ok(tables)
    }

    pub(crate) fn table(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
        dtype: DType,
    ) -> Result<Tensor> {
        let table = Tensor::zeros((rows, cols), dtype, &self.device)?;
        let mut row = 0;
        for chunk in self
            .checkpoint
            .take_rows(name, (STAGING_BYTES / (cols * 4)).max(1))?
        {
            let chunk = chunk?.to_dtype(dtype)?.to_device(&self.device)?;
            let count = chunk.dim(0)?;
            table.slice_set(&chunk, 0, row)?;
            row += count;
            drop(chunk);
            // Reclaim completed upload buffers before staging the next chunk.
            self.device.synchronize()?;
        }
        Ok(table)
    }
}
