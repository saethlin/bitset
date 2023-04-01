//! The inline representation of a small-size-optimized bitset.
//!
//! This type maintains a number of tricky invariants.
//! It stores its length in one byte, but as an enum so that its representation
//! has a niche that a surrounding `repr(Rust)` enum can wrap it without any
//! size overhead.

use crate::{nonmaxu8::NonMaxU8, num_words, Word};

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
#[repr(C)]
pub struct DenseBitSet {
    words: [Word; 3],
    tail: [u8; 7],
    domain_size: NonMaxU8,
}

const INLINE_CAPACITY_BYTES: usize = 31;
const INLINE_CAPACITY_BITS: usize = INLINE_CAPACITY_BYTES * 8;

impl DenseBitSet {
    /// # Panics
    ///
    /// Panics if `domain_size > 248`.
    #[inline]
    pub fn new_empty(domain_size: usize) -> Self {
        assert!(domain_size <= INLINE_CAPACITY_BITS);
        Self {
            words: [0; 3],
            tail: [0; 7],
            domain_size: NonMaxU8::new(domain_size as u8).unwrap(),
        }
    }

    /// Returns the number of bits this can hold.
    #[inline]
    pub fn domain_size(&self) -> usize {
        self.domain_size.get() as usize
    }

    /// Provides mutable access to the all used `Word`s.
    ///
    /// The intended use of this function is to implement set algorithms that
    /// benefit from operating on multiple bits at once.
    ///
    /// This exposes the inner representation of `DenseBitSet` to safe code. To
    /// ensure that the invariants of the type are not broken, this method
    /// writes the previous value of `domain_size` after the provided
    /// function returns. Note that this only ensures soundness, callers can
    /// still mistakenly read through the `&mut [Word]` and find unexpected
    /// bits.
    #[inline]
    pub fn words_mut<R, F: FnOnce(&mut [Word]) -> R>(&mut self, f: F) -> R {
        struct ResetGuard<'a> {
            set: &'a mut DenseBitSet,
            original_size: NonMaxU8,
        }

        impl<'a, 'b> ResetGuard<'a> {
            unsafe fn get(&'b mut self) -> &'b mut [u64] {
                let size = num_words(self.original_size.get() as usize);
                &mut self.set.as_words_mut()[..size]
            }
        }

        impl<'a> Drop for ResetGuard<'a> {
            fn drop(&mut self) {
                self.set.domain_size = self.original_size;
            }
        }

        // SAFETY: We expose mutable access to the last byte in our tail, but as soon as
        // `f` returns, we set it back to whatever it was previously.
        unsafe {
            let mut guard = ResetGuard {
                original_size: self.domain_size,
                set: self,
            };
            f(guard.get())
        }
    }

    #[inline]
    pub fn raw_parts(&self) -> (&[u8], usize) {
        let domain_size = self.domain_size.get() as usize;
        (&self.as_bytes()[..], domain_size)
    }

    #[inline]
    pub fn raw_parts_mut(&mut self) -> (&mut [u8], usize) {
        let domain_size = self.domain_size.get() as usize;
        (&mut self.as_bytes_mut()[..], domain_size)
    }

    #[inline]
    fn as_bytes(&self) -> &[u8; 31] {
        // SAFETY: Reinterpreting u64 as u8 is valid, and our repr(C) ensures we have no
        // padding.
        unsafe { &*(self as *const Self as *const [u8; 31]) }
    }

    #[inline]
    fn as_bytes_mut(&mut self) -> &mut [u8; 31] {
        // SAFETY: Reinterpreting u64 as u8 is valid, and our repr(C) ensures we have no
        // padding.
        unsafe { &mut *(self as *mut Self as *mut [u8; 31]) }
    }

    #[inline]
    fn as_words(&self) -> &[u64; 4] {
        // SAFETY: Reinterpreting u64 as u8 is valid, and our repr(C) ensures we have no
        // padding. This also provides access to our NonMaxU8 as if it is part
        // of a u64. This is also valid, and we don't provide mutation, so the
        // niche is preserved.
        unsafe { &*(self as *const Self as *const [u64; 4]) }
    }

    /// SAFETY: Callers must ensure that if the last 8 bits are modified, that
    /// they are reset to a nonzero value before `self` is accessed as a
    /// `DenseBitSet` again.
    #[inline]
    unsafe fn as_words_mut(&mut self) -> &mut [u64; 4] {
        unsafe { &mut *(self as *mut Self as *mut [u64; 4]) }
    }

    #[inline]
    pub fn union(&mut self, other: &Self) -> bool {
        assert_eq!(self.domain_size, other.domain_size);
        // Since domain_size of self and other must be equal, we can just or all the
        // bits, and the domain size bits will not be changed.
        unsafe {
            let mut changed = 0;
            for (this, other) in self.as_words_mut().iter_mut().zip(other.as_words().iter()) {
                let old = *this;
                let new = old | *other;
                *this = new;
                changed |= old ^ new;
            }
            changed != 0
        }
    }

    #[inline]
    pub fn intersect(&mut self, other: &Self) -> bool {
        assert_eq!(self.domain_size, other.domain_size);
        // Since domain_size of self and other must be equal, we can just or all the
        // bits, and the domain size bits will not be changed.
        unsafe {
            let mut changed = 0;
            for (this, other) in self.as_words_mut().iter_mut().zip(other.as_words().iter()) {
                let old = *this;
                let new = old & *other;
                *this = new;
                changed |= old ^ new;
            }
            changed != 0
        }
    }

    #[inline]
    pub fn words(&self) -> impl Iterator<Item = Word> + '_ {
        DenseBitSetWordIter {
            state: 0,
            set: self,
        }
    }
}

pub struct DenseBitSetWordIter<'a> {
    state: u8,
    set: &'a DenseBitSet,
}

const STATE_TAIL: u8 = 3;
const STATE_DONE: u8 = 4;

impl Iterator for DenseBitSetWordIter<'_> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.state >= STATE_DONE {
            return None;
        }
        let mut word = self.set.as_words()[self.state as usize];
        if self.state == STATE_TAIL {
            word &= u64::MAX >> 8;
        }
        self.state += 1;
        Some(word)
    }
}
