# FRBF: Fast Rust Bloom Filter

A high-performance, well-documented Bloom Filter implementation in Rust with configurable parameters, comprehensive error handling, and advanced features.

## Overview

A Bloom Filter is a space-efficient probabilistic data structure designed to test whether an element is a member of a set. It returns either "possibly in the set" or "definitely not in the set". False positive matches are possible, but false negatives are not.

## Features

- Efficient membership tests with no false negatives
- Configurable false positive probability
- Comprehensive error handling
- Optimized hash function implementation using double hashing
- Well-documented API with usage examples
- Extensive test suite
- Support for any hashable type
- Builder pattern for flexible initialization
- Serialization/deserialization support (optional)
- Ability to clear the filter without reallocation
- Merge operation for combining compatible filters
- Element count and false positive rate estimation

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
frbf = "0.1.0"
```

## Usage

### Creating a Bloom Filter

To create a new Bloom Filter:

```rust
use frbf::BloomFilter;

let bloom_filter_result = BloomFilter::new(1000, 0.01);

match bloom_filter_result {
    Ok(mut bloom_filter) => {
        // Use the bloom filter...
    },
    Err(e) => {
        println!("Error creating Bloom Filter: {}", e);
    }
}
```

This attempts to create a new Bloom Filter optimized for 1000 items and a 1% false positive probability. The implementation automatically calculates the optimal number of bits and hash functions based on these parameters.

### Adding Items

To add an item to the Bloom Filter:

```rust
bloom_filter.add(&"hello");
```

You can add any type that implements the `Hash` trait:

```rust
// Add integers
bloom_filter.add(&42);

// Add tuples
bloom_filter.add(&(1, "test"));

// Add custom structs that implement Hash
#[derive(Hash)]
struct User {
    id: u64,
    name: String,
}

let user = User { id: 1, name: "Alice".to_string() };
bloom_filter.add(&user);
```

### Checking Membership

To check if an item is present in the Bloom Filter:

```rust
if bloom_filter.check(&"hello") {
    println!("'hello' might be in the set!");
} else {
    println!("'hello' is definitely not in the set.");
}
```

### Getting Filter Properties

You can access filter properties:

```rust
// Get the bit array size
let bit_size = bloom_filter.bit_size();

// Get the number of hash functions
let hash_count = bloom_filter.hash_count();

// Get the expected number of elements
let expected = bloom_filter.expected_elements();

// Get the number of elements that have been added
let inserted = bloom_filter.inserted_elements();
```

## Performance Considerations

The Bloom Filter is optimized for space efficiency and performance:

- Time complexity for both add and check operations: O(k) where k is the number of hash functions
- Space complexity: O(m) where m is the number of bits in the filter

For optimal performance, choose the expected number of elements and false positive probability carefully:

- Higher number of expected elements or lower false positive probability will result in a larger bit array
- Very low false positive probabilities may result in a high number of hash functions, impacting performance

## Error Handling

The implementation returns a `Result` type when creating a new Bloom Filter or performing operations like merging, which can contain the following errors:

- `BloomFilterError::InvalidProbability` - When the false positive probability is not between 0 and 1
- `BloomFilterError::MemoryAllocationFailed` - When memory allocation for the bit array fails
- `BloomFilterError::IncompatibleFilters` - When attempting to merge filters with different bit sizes or hash counts

## Advanced Usage

### Using the Builder Pattern

For more flexible initialization, you can use the builder pattern:

```rust
use frbf::BloomFilterBuilder;

let mut filter = BloomFilterBuilder::new(1000)
    .false_positive_probability(0.001)
    .hash_seeds([42, 43, 44, 45], [99, 98, 97, 96]) // Custom hash seeds
    .build()
    .unwrap();

filter.add(&"hello");
assert!(filter.check(&"hello"));
```

### Clearing the Filter

You can clear the filter without reallocating memory:

```rust
// Add some items
filter.add(&"hello");
filter.add(&"world");

// Clear the filter
filter.clear();

// The filter is now empty
assert!(!filter.check(&"hello"));
assert!(!filter.check(&"world"));
```

### Merging Filters

You can merge two compatible bloom filters:

```rust
let mut filter1 = BloomFilter::new(1000, 0.01).unwrap();
let mut filter2 = BloomFilter::new(1000, 0.01).unwrap();

filter1.add(&"hello");
filter2.add(&"world");

// Merge filter2 into filter1
filter1.merge(&filter2).unwrap();

// filter1 now contains items from both filters
assert!(filter1.check(&"hello"));
assert!(filter1.check(&"world"));
```

### Estimating Elements and False Positive Rate

You can estimate the number of elements in the filter and the current false positive rate:

```rust
// Add some elements
for i in 0..1000 {
    filter.add(&i);
}

// Estimate the number of elements
let estimated_count = filter.estimate_elements();
println!("Estimated element count: {}", estimated_count);

// Estimate the current false positive rate
let estimated_fpp = filter.estimate_false_positive_rate();
println!("Estimated false positive rate: {}", estimated_fpp);
```

### Serialization and Deserialization

To enable serialization support, add the `serde-support` feature to your `Cargo.toml`:

```toml
[dependencies]
frbf = { version = "0.1.0", features = ["serde-support"] }
```

Then you can serialize and deserialize the bloom filter:

```rust
use serde_json;

let mut filter = BloomFilter::new(1000, 0.01).unwrap();
filter.add(&"hello");

// Serialize
let serialized = serde_json::to_string(&filter).unwrap();

// Deserialize
let deserialized: BloomFilter = serde_json::from_str(&serialized).unwrap();

// The deserialized filter contains the same elements
assert!(deserialized.check(&"hello"));
```

### Calculating the Actual False Positive Rate

You can verify the actual false positive rate of your filter:

```rust
let n = 10000;
let expected_fpp = 0.01;
let mut bloom_filter = BloomFilter::new(n, expected_fpp).unwrap();

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
println!("Expected FPP: {}, Actual FPP: {}", expected_fpp, actual_fpp);
```

## Limitations

- Once an element is added to the Bloom Filter, it cannot be removed
- The false positive rate increases as more elements are added to the filter
- Very large expected element counts may result in memory allocation failures

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
