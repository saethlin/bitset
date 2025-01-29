use std::{
    fmt,
    ops::{BitAnd, BitAndAssign, BitOrAssign, Not, Range, Shl},
};

/// Integral type used to represent the bit set.
pub trait FiniteBitSetTy:
    BitAnd<Output = Self>
    + BitAndAssign
    + BitOrAssign
    + Clone
    + Copy
    + Shl
    + Not<Output = Self>
    + PartialEq
    + Sized
{
    /// Size of the domain representable by this type, e.g. 64 for `u64`.
    const DOMAIN_SIZE: u32;

    /// Value which represents the `FiniteBitSet` having every bit set.
    const FILLED: Self;
    /// Value which represents the `FiniteBitSet` having no bits set.
    const EMPTY: Self;

    /// Value for one as the integral type.
    const ONE: Self;
    /// Value for zero as the integral type.
    const ZERO: Self;

    /// Perform a checked left shift on the integral type.
    fn checked_shl(self, rhs: u32) -> Option<Self>;
    /// Perform a checked right shift on the integral type.
    fn checked_shr(self, rhs: u32) -> Option<Self>;
}

impl FiniteBitSetTy for u32 {
    const DOMAIN_SIZE: u32 = 32;

    const FILLED: Self = Self::MAX;
    const EMPTY: Self = Self::MIN;

    const ONE: Self = 1u32;
    const ZERO: Self = 0u32;

    fn checked_shl(self, rhs: u32) -> Option<Self> {
        self.checked_shl(rhs)
    }

    fn checked_shr(self, rhs: u32) -> Option<Self> {
        self.checked_shr(rhs)
    }
}

impl std::fmt::Debug for FiniteBitSet<u32> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:032b}", self.0)
    }
}

impl FiniteBitSetTy for u64 {
    const DOMAIN_SIZE: u32 = 64;

    const FILLED: Self = Self::MAX;
    const EMPTY: Self = Self::MIN;

    const ONE: Self = 1u64;
    const ZERO: Self = 0u64;

    fn checked_shl(self, rhs: u32) -> Option<Self> {
        self.checked_shl(rhs)
    }

    fn checked_shr(self, rhs: u32) -> Option<Self> {
        self.checked_shr(rhs)
    }
}

impl std::fmt::Debug for FiniteBitSet<u64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:064b}", self.0)
    }
}

impl FiniteBitSetTy for u128 {
    const DOMAIN_SIZE: u32 = 128;

    const FILLED: Self = Self::MAX;
    const EMPTY: Self = Self::MIN;

    const ONE: Self = 1u128;
    const ZERO: Self = 0u128;

    fn checked_shl(self, rhs: u32) -> Option<Self> {
        self.checked_shl(rhs)
    }

    fn checked_shr(self, rhs: u32) -> Option<Self> {
        self.checked_shr(rhs)
    }
}

impl std::fmt::Debug for FiniteBitSet<u128> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:0128b}", self.0)
    }
}

/// A fixed-sized bitset type represented by an integer type. Indices outwith
/// than the range representable by `T` are considered set.
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct FiniteBitSet<T: FiniteBitSetTy>(pub T);

impl<T: FiniteBitSetTy> FiniteBitSet<T> {
    /// Creates a new, empty bitset.
    pub fn new_empty() -> Self {
        Self(T::EMPTY)
    }

    /// Sets the `index`th bit.
    pub fn set(&mut self, index: u32) {
        self.0 |= T::ONE.checked_shl(index).unwrap_or(T::ZERO);
    }

    /// Unsets the `index`th bit.
    pub fn clear(&mut self, index: u32) {
        self.0 &= !T::ONE.checked_shl(index).unwrap_or(T::ZERO);
    }

    /// Sets the `i`th to `j`th bits.
    pub fn set_range(&mut self, range: Range<u32>) {
        let bits = T::FILLED
            .checked_shl(range.end - range.start)
            .unwrap_or(T::ZERO)
            .not()
            .checked_shl(range.start)
            .unwrap_or(T::ZERO);
        self.0 |= bits;
    }

    /// Is the set empty?
    pub fn is_empty(&self) -> bool {
        self.0 == T::EMPTY
    }

    /// Returns the domain size of the bitset.
    pub fn within_domain(&self, index: u32) -> bool {
        index < T::DOMAIN_SIZE
    }

    /// Returns if the `index`th bit is set.
    pub fn contains(&self, index: u32) -> Option<bool> {
        self.within_domain(index)
            .then(|| ((self.0.checked_shr(index).unwrap_or(T::ONE)) & T::ONE) == T::ONE)
    }
}

impl<T: FiniteBitSetTy> Default for FiniteBitSet<T> {
    fn default() -> Self {
        Self::new_empty()
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn every_bit_u32() {
        let mut set: FiniteBitSet<u32> = FiniteBitSet::new_empty();
        assert!(set.is_empty());
        assert_eq!(set, FiniteBitSet::default());
        for bit in 0..32 {
            assert!(set.within_domain(bit));
            set.set(bit);
            assert_eq!(set.contains(bit), Some(true));
            assert!(!set.is_empty());

            assert_eq!(set, set.clone());

            set.clear(bit);
            assert_eq!(set.contains(bit), Some(false));
            assert!(set.is_empty());

            assert_eq!(set, FiniteBitSet::new_empty());
        }
    }

    #[test]
    fn every_bit_u64() {
        let mut set: FiniteBitSet<u64> = FiniteBitSet::new_empty();
        assert!(set.is_empty());
        assert_eq!(set, FiniteBitSet::default());
        for bit in 0..64 {
            assert!(set.within_domain(bit));
            set.set(bit);
            assert_eq!(set.contains(bit), Some(true));
            assert!(!set.is_empty());

            assert_eq!(set, set.clone());

            set.clear(bit);
            assert_eq!(set.contains(bit), Some(false));
            assert!(set.is_empty());

            assert_eq!(set, FiniteBitSet::new_empty());
        }
    }

    #[test]
    fn every_bit_u128() {
        let mut set: FiniteBitSet<u128> = FiniteBitSet::new_empty();
        assert!(set.is_empty());
        assert_eq!(set, FiniteBitSet::default());
        for bit in 0..128 {
            assert!(set.within_domain(bit));
            set.set(bit);
            assert_eq!(set.contains(bit), Some(true));
            assert!(!set.is_empty());

            assert_eq!(set, set.clone());

            set.clear(bit);
            assert_eq!(set.contains(bit), Some(false));
            assert!(set.is_empty());

            assert_eq!(set, FiniteBitSet::new_empty());
        }
    }

    #[test]
    fn insert_range() {
        let mut set: FiniteBitSet<u64> = FiniteBitSet::new_empty();
        set.set_range(1..7);
        assert!(!set.is_empty());
        assert_eq!(set.contains(0), Some(false));
        for bit in 1..7 {
            assert_eq!(set.contains(bit), Some(true));
        }
    }
}
