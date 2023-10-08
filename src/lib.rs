use std::cmp;
use std::hash::{BuildHasher, Hash, Hasher};

use ahash::RandomState;
use bitvec::prelude::*;

// Bloom Filter struct
pub struct BloomFilter {
    bitvec: BitVec,
    m: usize,
    k: usize,
    hashes: [RandomState; 2],
}

impl BloomFilter {
    // Create new bloom filter
    pub fn new(n: usize, false_positive_probability: f64) -> Self {
        let m = Self::optimal_m(n, false_positive_probability);
        let k = Self::optimal_k(m, n);
        let bitvec = bitvec![0; m];
        let hashes = [
            RandomState::with_seeds(0, 0, 0, 0),
            RandomState::with_seeds(1, 1, 1, 1),
        ];
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

    // Generate hash values for double hashing
    fn hash_values<T: Hash>(&self, item: &T) -> (usize, usize) {
        let mut hashers: Vec<_> = self.hashes.iter().map(|state| state.build_hasher()).collect();
        item.hash(&mut hashers[0]);
        item.hash(&mut hashers[1]);
        (hashers[0].finish() as usize, hashers[1].finish() as usize)
    }

    // Add an item to the bloom filter
    pub fn add<T: Hash>(&mut self, item: &T) {
        let (hash1, hash2) = self.hash_values(item);
        for i in 0..self.k {
            let combined_hash = hash1.wrapping_add(i.wrapping_mul(hash2));
            let index = combined_hash % self.m;
            self.bitvec.set(index, true);
        }
    }

    // Check if an item is present in the bloom filter
    pub fn check<T: Hash>(&self, item: &T) -> bool {
        let (hash1, hash2) = self.hash_values(item);
        (0..self.k).all(|i| {
            let combined_hash = hash1.wrapping_add(i.wrapping_mul(hash2));
            let index = combined_hash % self.m;
            self.bitvec[index]
        })
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
        let false_positive_probability = 0.01;
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
