//! Implements Compression and Decompression for `Datasets` and `Clusters`.

use crate::dataset::RowMajor;
use crate::prelude::*;

// A `Dataset` that also allows for compression and decompression.

// Instances in a `CompressibleDataset` can be encoded in terms of each other to
// produce compressed encodings represented as bytes.
// Those bytes can also be used to decode an encoded instance by using the reference.
pub trait CompressibleDataset<T: Number, U: Number>: Dataset<T, U> {
    /// Get a reference as a regular `Dataset`.
    fn as_dataset(&self) -> &dyn Dataset<T, U>;

    /// Encode one instance in terms of another.
    fn encode(&self, x: Index, y: Index) -> Result<Vec<u8>, String> {
        self.metric().encode(&self.instance(x), &self.instance(y))
    }

    /// Decode an instance from the encoded bytes and the reference.
    fn decode(&self, x: Index, y: &[u8]) -> Result<Vec<T>, String> {
        self.metric().decode(&self.instance(x), y)
    }
}

impl<T: Number, U: Number> CompressibleDataset<T, U> for RowMajor<T, U> {
    fn as_dataset(&self) -> &dyn Dataset<T, U> {
        self
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::dataset::RowMajor;
    use crate::CompressibleDataset;

    #[test]
    fn test_codec() {
        let data = vec![vec![0., 0., 0.], vec![1., 1., 1.], vec![2., 2., 2.], vec![3., 3., 3.]];
        let dataset: Arc<dyn CompressibleDataset<_, f64>> = Arc::new(RowMajor::new(data, "hamming", false).unwrap());

        let encoded = dataset.encode(0, 1).unwrap();
        let decoded = dataset.decode(0, &encoded).unwrap();

        assert_eq!(dataset.instance(1), decoded);
    }
}
