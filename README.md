# Bloom Filter in Rust

A simple, robust, and efficient implementation of the Bloom Filter data structure in Rust.

## Overview

A Bloom Filter is a probabilistic data structure that can test whether an element is a member of a set. It returns either "possibly in the set" or "definitely not in the set". False positive matches are possible, but false negatives are not.

## Features

- Efficient membership tests with no false negatives.
- Configurable false positive probability.

## Usage

### Creating a Bloom Filter

To create a new Bloom Filter:

```rust
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

This attempts to create a new Bloom Filter optimized for 1000 items and a 1% false positive probability. Make sure to handle the potential errors.

### Adding Items

To add an item to the Bloom Filter:

```rust
bloom_filter.add(&"hello");
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

## Limitations

- The current implementation uses double hashing determined during the Bloom Filter's creation. The number of hash functions is based on the desired false positive rate and the expected number of items.
- While the false positive rate can be minimized by adjusting the parameters, it cannot be completely eliminated.
- Errors such as invalid false positive probabilities or memory allocation failures should be handled appropriately.
