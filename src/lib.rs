use std::cmp;
use std::error::Error;
use std::fmt;
use std::hash::{BuildHasher, Hash, Hasher};

use ahash::RandomState;
use bitvec::prelude::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Error types that can occur when working with a bloom filter
#[derive(Debug)]
pub enum BloomFilterError {
    /// False positive probability must be > 0.0 and < 1.0
    InvalidProbability,
    /// Failed to allocate memory for the bloom filter's bit vector
    MemoryAllocationFailed,
    /// Bloom filters are not compatible for merging (different sizes or hash counts)
    IncompatibleFilters,
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
            },
            BloomFilterError::IncompatibleFilters => {
                write!(f, "Bloom filters are not compatible for merging (different sizes or hash counts).")
            }
        }
    }
}

impl Error for BloomFilterError {}

/// Configuration for hash functions
#[derive(Debug, Clone)]
struct HashConfig {
    /// First hash function seed
    h1_seed: [u64; 4],
    /// Second hash function seed
    h2_seed: [u64; 4],
}

impl Default for HashConfig {
    fn default() -> Self {
        Self {
            h1_seed: [0, 0, 0, 0],
            h2_seed: [1, 1, 1, 1],
        }
    }
}

/// A probabilistic data structure that tests whether an element is a member of a set.
///
/// False positive matches are possible, but false negatives are not.
/// Elements can be added to the set, but not removed.
/// The more elements that are added to the set, the larger the probability of false positives.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BloomFilter {
    /// The bit vector that stores the bloom filter's data
    bit_array: BitVec,
    /// The number of bits in the bloom filter
    bit_size: usize,
    /// The number of hash functions to use
    hash_count: usize,
    /// The hash functions used for double hashing
    hash_functions: [RandomState; 2],
    /// The expected number of elements to be inserted
    expected_elements: usize,
    /// The number of elements that have been added (estimated)
    inserted_elements: usize,
}

impl BloomFilter {
    /// Creates a new bloom filter with the given expected element count and false positive probability.
    ///
    /// # Arguments
    ///
    /// * `expected_elements` - The expected number of elements to be inserted
    /// * `false_positive_probability` - The desired false positive probability (between 0 and 1)
    ///
    /// # Returns
    ///
    /// A result containing the new BloomFilter or an error
    ///
    /// # Example
    ///
    /// ```
    /// use frbf::BloomFilter;
    ///
    /// let mut filter = BloomFilter::new(1000, 0.01).unwrap();
    /// filter.add(&"hello");
    /// assert!(filter.check(&"hello"));
    /// ```
    pub fn new(expected_elements: usize, false_positive_probability: f64) -> Result<Self, BloomFilterError> {
        Self::with_config(expected_elements, false_positive_probability, HashConfig::default())
    }

    /// Creates a new bloom filter with custom hash function configuration.
    ///
    /// # Arguments
    ///
    /// * `expected_elements` - The expected number of elements to be inserted
    /// * `false_positive_probability` - The desired false positive probability (between 0 and 1)
    /// * `hash_config` - Configuration for the hash functions
    fn with_config(
        expected_elements: usize,
        false_positive_probability: f64,
        hash_config: HashConfig,
    ) -> Result<Self, BloomFilterError> {
        // Validate probability range
        if false_positive_probability <= 0.0 || false_positive_probability >= 1.0 {
            return Err(BloomFilterError::InvalidProbability);
        }

        // Calculate optimal parameters
        let bit_size = Self::optimal_bit_size(expected_elements, false_positive_probability);
        let hash_count = Self::optimal_hash_count(bit_size, expected_elements);

        // Initialize bit vector
        let bit_array = bitvec![0; bit_size];

        // Check if memory allocation succeeded
        if bit_array.capacity() < bit_size {
            return Err(BloomFilterError::MemoryAllocationFailed);
        }

        // Initialize hash functions
        let hash_functions = [
            RandomState::with_seeds(
                hash_config.h1_seed[0],
                hash_config.h1_seed[1],
                hash_config.h1_seed[2],
                hash_config.h1_seed[3],
            ),
            RandomState::with_seeds(
                hash_config.h2_seed[0],
                hash_config.h2_seed[1],
                hash_config.h2_seed[2],
                hash_config.h2_seed[3],
            ),
        ];

        Ok(Self {
            bit_array,
            bit_size,
            hash_count,
            hash_functions,
            expected_elements,
            inserted_elements: 0,
        })
    }

    /// Calculate optimal number of bits for the given parameters
    fn optimal_bit_size(expected_elements: usize, false_positive_probability: f64) -> usize {
        let ln_2 = std::f64::consts::LN_2;
        let size = (expected_elements as f64 * false_positive_probability.ln() / (-ln_2 * ln_2)).ceil();
        size as usize
    }

    /// Calculate optimal number of hash functions for the given parameters
    fn optimal_hash_count(bit_size: usize, expected_elements: usize) -> usize {
        let k = (bit_size as f64 / expected_elements as f64 * std::f64::consts::LN_2).ceil();
        cmp::max(k as usize, 1)
    }

    /// Generate hash values for double hashing
    fn compute_hashes<T: Hash>(&self, item: &T) -> (usize, usize) {
        let mut hashers: Vec<_> = self
            .hash_functions
            .iter()
            .map(|state| state.build_hasher())
            .collect();

        item.hash(&mut hashers[0]);
        item.hash(&mut hashers[1]);

        (hashers[0].finish() as usize, hashers[1].finish() as usize)
    }

    /// Add an item to the bloom filter
    ///
    /// # Arguments
    ///
    /// * `item` - The item to add
    pub fn add<T: Hash>(&mut self, item: &T) {
        let (hash1, hash2) = self.compute_hashes(item);

        for i in 0..self.hash_count {
            let index = self.index_for_hash(hash1, hash2, i);
            self.bit_array.set(index, true);
        }

        self.inserted_elements += 1;
    }

    /// Clears the bloom filter by resetting all bits to 0
    ///
    /// This operation resets the filter to its initial state without reallocating memory.
    pub fn clear(&mut self) {
        self.bit_array.fill(false);
        self.inserted_elements = 0;
    }

    /// Merges another bloom filter into this one
    ///
    /// Both filters must have the same bit size and hash count to be compatible.
    ///
    /// # Arguments
    ///
    /// * `other` - The other bloom filter to merge with
    ///
    /// # Returns
    ///
    /// A result indicating success or an error if the filters are incompatible
    ///
    /// # Example
    ///
    /// ```
    /// use frbf::BloomFilter;
    ///
    /// let mut filter1 = BloomFilter::new(1000, 0.01).unwrap();
    /// let mut filter2 = BloomFilter::new(1000, 0.01).unwrap();
    ///
    /// filter1.add(&"hello");
    /// filter2.add(&"world");
    ///
    /// filter1.merge(&filter2).unwrap();
    ///
    /// assert!(filter1.check(&"hello"));
    /// assert!(filter1.check(&"world"));
    /// ```
    pub fn merge(&mut self, other: &Self) -> Result<(), BloomFilterError> {
        // Check if the filters are compatible
        if self.bit_size != other.bit_size || self.hash_count != other.hash_count {
            return Err(BloomFilterError::IncompatibleFilters);
        }

        // Merge the bit arrays using bitwise OR
        for i in 0..self.bit_array.len() {
            if other.bit_array[i] {
                self.bit_array.set(i, true);
            }
        }

        // Update the inserted elements count
        self.inserted_elements += other.inserted_elements;

        Ok(())
    }

    /// Estimates the number of elements in the bloom filter
    ///
    /// This is an approximation based on the number of bits set in the filter.
    ///
    /// # Returns
    ///
    /// The estimated number of elements in the filter
    pub fn estimate_elements(&self) -> usize {
        // Count the number of bits set to 1
        let bits_set = self.bit_array.count_ones();

        // Use the formula: n = -m * ln(1 - X/m) / k
        // where m is the bit size, X is the number of bits set to 1, and k is the number of hash functions
        let m = self.bit_size as f64;
        let x = bits_set as f64;
        let k = self.hash_count as f64;

        let estimate = -m * (1.0 - x / m).ln() / k;
        estimate.round() as usize
    }

    /// Estimates the current false positive probability
    ///
    /// This is an approximation based on the number of bits set in the filter.
    ///
    /// # Returns
    ///
    /// The estimated false positive probability
    pub fn estimate_false_positive_rate(&self) -> f64 {
        // Count the number of bits set to 1
        let bits_set = self.bit_array.count_ones();

        // Calculate the probability that a bit is still 0
        let p = 1.0 - (bits_set as f64 / self.bit_size as f64);

        // The false positive probability is (1 - p)^k
        (1.0 - p).powf(self.hash_count as f64)
    }

    /// Check if an item might be in the bloom filter
    ///
    /// Returns true if the item might be in the set, false if it definitely is not.
    ///
    /// # Arguments
    ///
    /// * `item` - The item to check
    ///
    /// # Returns
    ///
    /// `true` if the item might be in the set, `false` if it definitely is not
    pub fn check<T: Hash>(&self, item: &T) -> bool {
        let (hash1, hash2) = self.compute_hashes(item);

        (0..self.hash_count).all(|i| {
            let index = self.index_for_hash(hash1, hash2, i);
            self.bit_array[index]
        })
    }

    /// Calculate the index in the bit array for a given hash combination
    #[inline]
    fn index_for_hash(&self, hash1: usize, hash2: usize, i: usize) -> usize {
        let combined_hash = hash1.wrapping_add(i.wrapping_mul(hash2));
        combined_hash % self.bit_size
    }

    /// Returns the current size of the bit array
    pub fn bit_size(&self) -> usize {
        self.bit_size
    }

    /// Returns the number of hash functions being used
    pub fn hash_count(&self) -> usize {
        self.hash_count
    }

    /// Returns the expected number of elements the filter was configured for
    pub fn expected_elements(&self) -> usize {
        self.expected_elements
    }

    /// Returns the number of elements that have been added to the filter
    pub fn inserted_elements(&self) -> usize {
        self.inserted_elements
    }
}

/// Builder for creating a BloomFilter with custom configuration
pub struct BloomFilterBuilder {
    expected_elements: usize,
    false_positive_probability: f64,
    hash_config: HashConfig,
}

impl BloomFilterBuilder {
    /// Creates a new builder with default values
    ///
    /// # Arguments
    ///
    /// * `expected_elements` - The expected number of elements to be inserted
    pub fn new(expected_elements: usize) -> Self {
        Self {
            expected_elements,
            false_positive_probability: 0.01, // Default 1% false positive rate
            hash_config: HashConfig::default(),
        }
    }

    /// Sets the desired false positive probability
    ///
    /// # Arguments
    ///
    /// * `probability` - The desired false positive probability (between 0 and 1)
    ///
    /// # Returns
    ///
    /// The builder for method chaining
    pub fn false_positive_probability(mut self, probability: f64) -> Self {
        self.false_positive_probability = probability;
        self
    }

    /// Sets custom hash function seeds
    ///
    /// # Arguments
    ///
    /// * `h1_seed` - Seed for the first hash function
    /// * `h2_seed` - Seed for the second hash function
    ///
    /// # Returns
    ///
    /// The builder for method chaining
    pub fn hash_seeds(mut self, h1_seed: [u64; 4], h2_seed: [u64; 4]) -> Self {
        self.hash_config.h1_seed = h1_seed;
        self.hash_config.h2_seed = h2_seed;
        self
    }

    /// Builds the BloomFilter with the configured parameters
    ///
    /// # Returns
    ///
    /// A result containing the new BloomFilter or an error
    ///
    /// # Example
    ///
    /// ```
    /// use frbf::BloomFilterBuilder;
    ///
    /// let mut filter = BloomFilterBuilder::new(1000)
    ///     .false_positive_probability(0.001)
    ///     .build()
    ///     .unwrap();
    ///
    /// filter.add(&"hello");
    /// assert!(filter.check(&"hello"));
    /// ```
    pub fn build(self) -> Result<BloomFilter, BloomFilterError> {
        BloomFilter::with_config(
            self.expected_elements,
            self.false_positive_probability,
            self.hash_config,
        )
    }
}

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
    fn test_accessors() {
        let filter = BloomFilter::new(1000, 0.01).unwrap();
        assert!(filter.bit_size() > 0);
        assert!(filter.hash_count() > 0);
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

    #[test]
    fn test_clear() {
        let mut filter = BloomFilter::new(100, 0.01).unwrap();

        // Add some items
        filter.add(&"hello");
        filter.add(&"world");

        // Verify they're in the filter
        assert!(filter.check(&"hello"));
        assert!(filter.check(&"world"));

        // Clear the filter
        filter.clear();

        // Verify items are no longer in the filter
        assert!(!filter.check(&"hello"));
        assert!(!filter.check(&"world"));

        // Add items again to verify the filter still works
        filter.add(&"foo");
        filter.add(&"bar");

        assert!(filter.check(&"foo"));
        assert!(filter.check(&"bar"));
    }

    #[test]
    fn test_merge() {
        let mut filter1 = BloomFilter::new(100, 0.01).unwrap();
        let mut filter2 = BloomFilter::new(100, 0.01).unwrap();

        // Add different items to each filter
        filter1.add(&"hello");
        filter1.add(&"world");

        filter2.add(&"foo");
        filter2.add(&"bar");

        // Verify items are in their respective filters
        assert!(filter1.check(&"hello"));
        assert!(filter1.check(&"world"));
        assert!(!filter1.check(&"foo"));
        assert!(!filter1.check(&"bar"));

        assert!(!filter2.check(&"hello"));
        assert!(!filter2.check(&"world"));
        assert!(filter2.check(&"foo"));
        assert!(filter2.check(&"bar"));

        // Merge filter2 into filter1
        let merge_result = filter1.merge(&filter2);
        assert!(merge_result.is_ok());

        // Verify all items are now in filter1
        assert!(filter1.check(&"hello"));
        assert!(filter1.check(&"world"));
        assert!(filter1.check(&"foo"));
        assert!(filter1.check(&"bar"));

        // Verify filter2 is unchanged
        assert!(!filter2.check(&"hello"));
        assert!(!filter2.check(&"world"));
        assert!(filter2.check(&"foo"));
        assert!(filter2.check(&"bar"));

        // Test incompatible filters
        let filter3 = BloomFilter::new(200, 0.001).unwrap(); // Different parameters
        let merge_result = filter1.merge(&filter3);
        assert!(merge_result.is_err());
        match merge_result {
            Err(BloomFilterError::IncompatibleFilters) => (), // Expected
            _ => panic!("Expected IncompatibleFilters error"),
        }
    }

    #[test]
    fn test_estimation_methods() {
        let n = 1000;
        let mut filter = BloomFilter::new(n, 0.01).unwrap();

        // Initially the filter should be empty
        assert_eq!(filter.estimate_elements(), 0);

        // Add some elements
        for i in 0..n {
            filter.add(&i);
        }

        // The estimated count should be reasonably close to the actual count
        let estimated = filter.estimate_elements();
        let error_margin = 0.1; // Allow 10% error
        assert!(
            (estimated as f64) > (n as f64) * (1.0 - error_margin) &&
            (estimated as f64) < (n as f64) * (1.0 + error_margin),
            "Estimated count {} is too far from actual count {}",
            estimated,
            n
        );

        // The estimated FPP should be reasonable
        let estimated_fpp = filter.estimate_false_positive_rate();
        assert!(estimated_fpp > 0.0 && estimated_fpp < 0.1); // Should be positive but small
    }

    #[test]
    fn test_builder_pattern() {
        // Test basic builder usage
        let mut filter = BloomFilterBuilder::new(1000)
            .false_positive_probability(0.001)
            .build()
            .unwrap();

        filter.add(&"hello");
        assert!(filter.check(&"hello"));

        // Test with custom hash seeds
        let mut filter = BloomFilterBuilder::new(1000)
            .false_positive_probability(0.01)
            .hash_seeds([42, 43, 44, 45], [99, 98, 97, 96])
            .build()
            .unwrap();

        filter.add(&"world");
        assert!(filter.check(&"world"));

        // Test invalid probability
        let result = BloomFilterBuilder::new(1000)
            .false_positive_probability(1.5) // Invalid
            .build();

        assert!(result.is_err());
        match result {
            Err(BloomFilterError::InvalidProbability) => (), // Expected
            _ => panic!("Expected InvalidProbability error"),
        }
    }
}
