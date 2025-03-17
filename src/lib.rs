use std::cmp;
use std::error::Error;
use std::fmt;
use std::hash::{BuildHasher, Hash, Hasher};

use ahash::RandomState;
use bitvec::prelude::*;

// Bloom Filter struct
#[derive(Debug)]
pub struct BloomFilter {
    bitvec: BitVec,
    m: usize,
    k: usize,
    hashes: [RandomState; 2],
}

impl BloomFilter {
    // Create new bloom filter
    pub fn new(n: usize, false_positive_probability: f64) -> Result<Self, BloomFilterError> {
        if false_positive_probability <= 0.0 || false_positive_probability >= 1.0 {
            return Err(BloomFilterError::InvalidProbability);
        }

        let m = Self::optimal_m(n, false_positive_probability);
        let k = Self::optimal_k(m, n);
        let bitvec = bitvec![0; m];

        if bitvec.capacity() < m {
            return Err(BloomFilterError::MemoryAllocationFailed);
        }

        let hashes = [
            RandomState::with_seeds(0, 0, 0, 0),
            RandomState::with_seeds(1, 1, 1, 1),
        ];

        Ok(Self {
            bitvec,
            m,
            k,
            hashes,
        })
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
        let mut hashers: Vec<_> = self
            .hashes
            .iter()
            .map(|state| state.build_hasher())
            .collect();
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

#[derive(Debug)]
pub enum BloomFilterError {
    InvalidProbability,
    MemoryAllocationFailed,
}

impl fmt::Display for BloomFilterError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            BloomFilterError::InvalidProbability => write!(
                f,
                "Invalid false_positive_probability provided. It should be between 0 and 1."
            ),
            BloomFilterError::MemoryAllocationFailed => {
                write!(f, "Failed to allocate memory for the BloomFilter.")
            }
        }
    }
}

impl Error for BloomFilterError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_bloom_filter() {
        let bloom_filter_result = BloomFilter::new(1000, 0.01);
        assert!(bloom_filter_result.is_ok());

        let mut bloom_filter = bloom_filter_result.unwrap();

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
    fn test_with_different_data_types() {
        let mut bloom_filter = BloomFilter::new(1000, 0.01).unwrap();

        // Test with integers
        bloom_filter.add(&42);
        bloom_filter.add(&1337);
        assert!(bloom_filter.check(&42));
        assert!(bloom_filter.check(&1337));
        assert!(!bloom_filter.check(&9999));

        // Test with tuples
        bloom_filter.add(&(1, "one"));
        bloom_filter.add(&(2, "two"));
        assert!(bloom_filter.check(&(1, "one")));
        assert!(bloom_filter.check(&(2, "two")));
        assert!(!bloom_filter.check(&(3, "three")));

        // Test with custom structs
        #[derive(Hash)]
        struct TestStruct {
            id: u32,
            name: String,
        }

        let item1 = TestStruct { id: 1, name: "test1".to_string() };
        let item2 = TestStruct { id: 2, name: "test2".to_string() };
        let item3 = TestStruct { id: 3, name: "test3".to_string() };

        bloom_filter.add(&item1);
        bloom_filter.add(&item2);
        assert!(bloom_filter.check(&item1));
        assert!(bloom_filter.check(&item2));
        assert!(!bloom_filter.check(&item3));
    }

    #[test]
    fn test_different_filter_sizes() {
        // Small filter
        let mut small_filter = BloomFilter::new(10, 0.1).unwrap();
        for i in 0..10 {
            small_filter.add(&i);
        }
        for i in 0..10 {
            assert!(small_filter.check(&i));
        }

        // Medium filter
        let mut medium_filter = BloomFilter::new(1000, 0.01).unwrap();
        for i in 0..1000 {
            medium_filter.add(&i);
        }
        for i in 0..1000 {
            assert!(medium_filter.check(&i));
        }

        // Large filter
        let mut large_filter = BloomFilter::new(10000, 0.001).unwrap();
        for i in 0..1000 {
            large_filter.add(&i);
        }
        for i in 0..1000 {
            assert!(large_filter.check(&i));
        }
    }

    #[test]
    fn test_false_positive_rate() {
        let n = 10000;
        let fpp = 0.01; // Expected false positive probability
        let mut bloom_filter = BloomFilter::new(n, fpp).unwrap();

        // Add n items
        for i in 0..n {
            bloom_filter.add(&i);
        }

        // Check for false positives with another n items that weren't added
        let mut false_positives = 0;
        for i in n..(2 * n) {
            if bloom_filter.check(&i) {
                false_positives += 1;
            }
        }

        let actual_fpp = false_positives as f64 / n as f64;
        println!("Expected FPP: {}, Actual FPP: {}", fpp, actual_fpp);

        // The actual false positive rate should be reasonably close to the expected rate
        // We use a generous margin to avoid flaky tests
        assert!(actual_fpp < fpp * 2.0, "False positive rate too high: {}", actual_fpp);
    }

    #[test]
    fn test_edge_cases() {
        // Test with minimum valid probability
        let min_prob_filter = BloomFilter::new(100, 0.000001);
        assert!(min_prob_filter.is_ok());

        // Test with maximum valid probability
        let max_prob_filter = BloomFilter::new(100, 0.999999);
        assert!(max_prob_filter.is_ok());

        // Test with very small capacity
        let small_capacity = BloomFilter::new(1, 0.01);
        assert!(small_capacity.is_ok());

        // Test adding and checking empty string
        let mut filter = BloomFilter::new(100, 0.01).unwrap();
        filter.add(&"");
        assert!(filter.check(&""));

        // Test adding and checking very long string
        let long_string = "a".repeat(10000);
        filter.add(&long_string);
        assert!(filter.check(&long_string));
    }

    #[test]
    fn test_error_conditions() {
        // Test with invalid probabilities
        assert!(BloomFilter::new(1000, -0.01).is_err());
        assert!(BloomFilter::new(1000, 0.0).is_err());
        assert!(BloomFilter::new(1000, 1.0).is_err());
        assert!(BloomFilter::new(1000, 1.1).is_err());
        
        // Test with very large capacity that might cause memory issues
        // Use a large but more reasonable value to avoid panicking the bitvec library
        let large_capacity = 1_000_000_000; // 1 billion elements
        let huge_filter = BloomFilter::new(large_capacity, 0.0001);
        // This might succeed or fail depending on available memory
        // We're just checking that it doesn't panic
        match huge_filter {
            Ok(_) => (), // It's ok if it succeeds
            Err(BloomFilterError::MemoryAllocationFailed) => (), // Expected on memory-constrained systems
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn performance_test() {
        let n = 1_000_000;
        let false_positive_probability = 0.01;

        let bloom_filter_result = BloomFilter::new(n, false_positive_probability);
        assert!(bloom_filter_result.is_ok());

        let mut bloom_filter = bloom_filter_result.unwrap();

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

    #[test]
    fn test_invalid_probability() {
        // This should fail because the false_positive_probability is not between 0 and 1
        let bloom_filter_result = BloomFilter::new(1000, -0.01);
        assert!(bloom_filter_result.is_err());
    }
}
