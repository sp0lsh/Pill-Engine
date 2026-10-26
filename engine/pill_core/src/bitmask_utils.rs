use std::{
    fmt::Binary,
    ops::{Add, Not, Shl, Sub},
};

// Creates bitmasks of any size with bits in range passed as parameter set to 1
// Where range order is: from MSB to LSB
// e.g. let x: u32 = create_bitmask_from_range::<u32>(4..8);
// will return mask: 0000_1111_1000_0000_0000_0000_0000_0000
pub fn create_bitmask_from_range<T>(mask_range: &core::ops::Range<T>) -> T
where
    T: Copy
        + Default
        + Binary
        + From<u8>
        + Ord
        + Shl<Output = T>
        + Sub<Output = T>
        + Add<Output = T>
        + Not<Output = T>,
{
    let mask_size = T::from(std::mem::size_of::<T>() as u8 * 8);

    if mask_range.end >= mask_size {
        panic!("Provided mask range exceeds mask size");
    }

    let range_length: T = mask_range.end - mask_range.start + T::from(1);
    let mask_shift = mask_size - mask_range.end - T::from(1);

    match range_length == mask_size {
        true => !(T::from(0)) << mask_shift, // Prevent overflow (when shifting by same value as number of bits in type)
        false => !(!T::from(0) << range_length) << mask_shift,
    }
}

/// A set of component types (one bit per type registered in a scene).
pub type ComponentMask = u64;

/// Component types a scene can register: the bits of [`ComponentMask`].
pub const MAX_COMPONENT_TYPES: usize = ComponentMask::BITS as usize;

// From 0 to MAX_COMPONENT_TYPES - 1, from MSB to LSB
// e.g. create_bitmask_with_one(3) sets the fourth highest bit
pub fn create_bitmask_with_one(index: usize) -> ComponentMask {
    assert!(
        index < MAX_COMPONENT_TYPES,
        "component type {index} exceeds the {MAX_COMPONENT_TYPES} a scene can register"
    );
    (1 as ComponentMask) << (MAX_COMPONENT_TYPES - 1 - index)
}

// Indices of the set bits, from MSB to LSB (the inverse of create_bitmask_with_one)
// e.g. bits 0, 3 and 9 set → [0, 3, 9]
pub fn get_indices_of_set_elements(bitmask: ComponentMask) -> Vec<usize> {
    (0..MAX_COMPONENT_TYPES)
        .filter(|&index| bitmask & create_bitmask_with_one(index) != 0)
        .collect()
}

#[cfg(test)]
mod component_mask_tests {
    use super::*;

    #[test]
    fn component_bits_round_trip_past_sixteen_types() {
        let indices = [0, 3, 15, 16, 40, MAX_COMPONENT_TYPES - 1];
        let mask = indices
            .iter()
            .fold(0, |mask, &index| mask | create_bitmask_with_one(index));
        assert_eq!(get_indices_of_set_elements(mask), indices);
    }

    #[test]
    #[should_panic(expected = "exceeds")]
    fn component_type_past_the_mask_panics() {
        create_bitmask_with_one(MAX_COMPONENT_TYPES);
    }
}
