use std::cmp;
use std::hash::{BuildHasher, Hash, Hasher};

use ahash::RandomState;
use bitvec::prelude::*;

// Bloom Filter struct
pub struct BloomFilter {
    bitvec: BitVec,
    m: usize,
    k: usize,
    hashes: Vec<RandomState>,
}

impl BloomFilter {
    // Create new bloom filter
    pub fn new(n: usize, false_positive_probability: f64) -> Self {
        let m = Self::optimal_m(n, false_positive_probability);
        let k = Self::optimal_k(m, n);
        let bitvec = bitvec![0; m];
        let seeds: Vec<u64> = (0..k).map(|i| i as u64).collect();
        let hashes = seeds
            .iter()
            .map(|&seed| {
                RandomState::with_seeds(seed, seed.reverse_bits(), seed, seed.reverse_bits())
            })
            .collect();
        Self {
            bitvec,
            m,
            k,
            hashes,
        }
    }

    // Calculate optimal number of bit maps
    fn optimal_m(n: usize, false_positive_probability: f64) -> usize {
        let ln_2 = std::f64::consts::LN_2;
        let m = (n as f64 * false_positive_probability.ln() / (-ln_2 * ln_2)).ceil();
        m as usize
    }

    // Calculate optimal number of hash functions
    fn optimal_k(m: usize, n: usize) -> usize {
        let k = (m as f64 / n as f64 * std::f64::consts::LN_2).ceil();
        cmp::max(k as usize, 1)
    }

    // Add an item to the bloom filter
    pub fn add<T: Hash>(&mut self, item: &T) {
        for i in 0..self.k {
            let mut hasher = self.hashes[i].build_hasher();
            item.hash(&mut hasher);
            let hash = hasher.finish() as usize;
            let index = (hash.wrapping_add(i)) % self.m;
            self.bitvec.set(index, true);
        }
    }

    // Check if an item is present in the bloom filter
    pub fn check<T: Hash>(&self, item: &T) -> bool {
        for i in 0..self.k {
            let mut hasher = self.hashes[i].build_hasher();
            item.hash(&mut hasher);
            let hash = hasher.finish() as usize;
            let index = (hash.wrapping_add(i)) % self.m;
            if !self.bitvec[index] {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_bloom_filter() {
        let mut bloom_filter = BloomFilter::new(1000, 0.01);

        // Add items to the bloom filter
        bloom_filter.add(&"hello");
        bloom_filter.add(&"world");

        // Check if items are present in the bloom filter
        assert!(bloom_filter.check(&"hello"));
        assert!(bloom_filter.check(&"world"));

        // Check if non-existent item is not present in the bloom filter
        assert!(!bloom_filter.check(&"foo"));

        // Add more items to the bloom filter
        bloom_filter.add(&"foo");
        bloom_filter.add(&"bar");

        // Check if newly added items are present in the bloom filter
        assert!(bloom_filter.check(&"foo"));
        assert!(bloom_filter.check(&"bar"));

        // Check if previously added items are still present in the bloom filter
        assert!(bloom_filter.check(&"hello"));
        assert!(bloom_filter.check(&"world"));

        // Check if non-existent item is not present in the bloom filter
        assert!(!bloom_filter.check(&"baz"));

        // Add more items to the bloom filter
        bloom_filter.add(&"baz");
        bloom_filter.add(&"qux");

        // Check if newly added items are present in the bloom filter
        assert!(bloom_filter.check(&"baz"));
        assert!(bloom_filter.check(&"qux"));

        // Check if previously added items are still present in the bloom filter
        assert!(bloom_filter.check(&"hello"));
        assert!(bloom_filter.check(&"world"));
        assert!(bloom_filter.check(&"foo"));
        assert!(bloom_filter.check(&"bar"));

        // Check if non-existent item is not present in the bloom filter
        assert!(!bloom_filter.check(&"quux"));
    }

    #[test]
    fn performance_test() {
        let n = 1_000_000;
        let false_positive_probability = 0.001;
        let mut bloom_filter = BloomFilter::new(n, false_positive_probability);

        // Performance test for `add` method
        let start = Instant::now();
        for i in 0..n {
            bloom_filter.add(&i.to_string());
        }
        let duration = start.elapsed();
        println!("Time taken to add {} items: {:?}", n, duration);

        // Performance test for `check` method
        let start = Instant::now();
        for i in 0..n {
            bloom_filter.check(&i.to_string());
        }
        let duration = start.elapsed();
        println!("Time taken to check {} items: {:?}", n, duration);
    }
}
