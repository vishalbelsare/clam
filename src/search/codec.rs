//! Implements Compression and Decompression for `Datasets` and `Clusters`.

use ndarray::prelude::*;

use crate::dataset::RowMajor;
use crate::prelude::*;

// A `Dataset` that also allows for compression and decompression.

// Instances in a `CompressibleDataset` can be endoced in terms of each other to
// produce compressed encodings represented as bytes.
// Those bytes can also be used to decode an encoded instance by using the reference.
pub trait CompressibleDataset<T: Number, U: Number>: Dataset<T, U> {
    /// Get a reference as a regular `Dataset`.
    fn as_dataset(&self) -> &dyn Dataset<T, U>;

    /// Encode one instance in terms of another.
    ///
    /// TODO: Think about whether to do this in terms of indices of instances.
    fn encode(&self, x: &ArrayView<T, IxDyn>, y: &ArrayView<T, IxDyn>) -> Result<Vec<u8>, String>;

    /// Decode an instance from the encoded bytes and the reference.
    fn decode(&self, x: &ArrayView<T, IxDyn>, y: &Vec<u8>) -> Result<Array<T, IxDyn>, String>;
}

impl<T: Number, U: Number> CompressibleDataset<T, U> for RowMajor<T, U> {
    fn as_dataset(&self) -> &dyn Dataset<T, U> {
        self
    }

    fn encode(&self, x: &ArrayView<T, IxDyn>, y: &ArrayView<T, IxDyn>) -> Result<Vec<u8>, String> {
        self.metric().encode(x, y)
    }

    fn decode(&self, x: &ArrayView<T, IxDyn>, y: &Vec<u8>) -> Result<Array<T, IxDyn>, String> {
        self.metric().decode(x, y)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ndarray::prelude::*;

    use crate::dataset::RowMajor;
    use crate::CompressibleDataset;

    #[test]
    fn test_codec() {
        let data: Array2<f64> = arr2(&[[0., 0., 0.], [1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]);
        let dataset: Arc<dyn CompressibleDataset<f64, f64>> = Arc::new(RowMajor::<f64, f64>::new(data, "hamming", false).unwrap());

        let row0 = dataset.instance(0);
        let row1 = dataset.instance(1);

        let encoded = dataset.encode(&row0, &row1).unwrap();
        let decoded = dataset.decode(&row0, &encoded).unwrap();
        let decoded = decoded.view();

        assert_eq!(row1, decoded);
    }
}
