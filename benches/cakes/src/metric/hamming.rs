//! The `Hamming` distance metric.

use std::sync::{Arc, RwLock};

use abd_clam::{metric::ParMetric, Metric};

use super::{CountingMetric, ParCountingMetric};

/// The `Hamming` distance metric.
pub struct Hamming(Arc<RwLock<usize>>, bool);

impl Hamming {
    /// Creates a new `Hamming` distance metric.
    pub fn new() -> Self {
        Self(Arc::new(RwLock::new(0)), true)
    }
}

impl Metric<String, u32> for Hamming {
    fn distance(&self, a: &String, b: &String) -> u32 {
        if self.1 {
            self.increment();
        }
        distances::strings::hamming(a, b)
    }

    fn name(&self) -> &str {
        "hamming"
    }

    fn has_identity(&self) -> bool {
        true
    }

    fn has_non_negativity(&self) -> bool {
        true
    }

    fn has_symmetry(&self) -> bool {
        true
    }

    fn obeys_triangle_inequality(&self) -> bool {
        true
    }

    fn is_expensive(&self) -> bool {
        false
    }
}

impl CountingMetric<String, u32> for Hamming {
    fn disable_counting(&mut self) {
        self.1 = false;
    }

    fn enable_counting(&mut self) {
        self.1 = true;
    }

    #[allow(clippy::unwrap_used)]
    fn count(&self) -> usize {
        *self.0.read().unwrap()
    }

    #[allow(clippy::unwrap_used)]
    fn reset_count(&self) -> usize {
        let mut count = self.0.write().unwrap();
        let old = *count;
        *count = 0;
        old
    }

    #[allow(clippy::unwrap_used)]
    fn increment(&self) {
        *self.0.write().unwrap() += 1;
    }
}

impl ParMetric<String, u32> for Hamming {}

impl ParCountingMetric<String, u32> for Hamming {}