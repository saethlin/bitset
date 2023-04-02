use crate::{
    byte_index_and_mask, fixed::BitSetImpl, num_words, word_index_and_mask, BitIter, BitSet, Idx,
};
use std::{iter, marker::PhantomData};

/// A resizable bitset type with a dense representation.
///
/// `T` is an index type, typically a newtyped `usize` wrapper, but it can also
/// just be `usize`.
///
/// All operations that involve an element will panic if the element is equal
/// to or greater than the domain size.
#[derive(Clone, Debug, PartialEq)]
pub struct GrowableBitSet<T: Idx> {
    bit_set: BitSet<T>,
}

impl<T: Idx> Default for GrowableBitSet<T> {
    fn default() -> Self {
        GrowableBitSet::new_empty()
    }
}

impl<T: Idx> GrowableBitSet<T> {
    /// Ensure that the set can hold at least `min_domain_size` elements.
    pub fn ensure(&mut self, min_domain_size: usize) {
        let (domain_size, current_len) = match &self.bit_set.inner {
            BitSetImpl::Inline(inline) => {
                if inline.ensure(min_domain_size).is_ok() {
                    return;
                } else {
                    (inline.domain_size(), 4)
                }
            }
            BitSetImpl::Heap { domain_size, words } => {
                if *domain_size >= min_domain_size {
                    return;
                }
                (*domain_size, words.len())
            }
        };
        // We need to do a heap resize. :'(
        let new_words = num_words(min_domain_size) - current_len;
        let words = self
            .bit_set
            .words()
            .chain(iter::repeat(0).take(new_words))
            .collect();
        self.bit_set = BitSet {
            inner: BitSetImpl::Heap {
                domain_size: min_domain_size,
                words,
            },
            marker: PhantomData,
        };
    }

    pub fn new_empty() -> GrowableBitSet<T> {
        GrowableBitSet {
            bit_set: BitSet::new_empty(0),
        }
    }

    pub fn with_capacity(capacity: usize) -> GrowableBitSet<T> {
        GrowableBitSet {
            bit_set: BitSet::new_empty(capacity),
        }
    }

    /// Returns `true` if the set has changed.
    #[inline]
    pub fn insert(&mut self, elem: T) -> bool {
        self.ensure(elem.index() + 1);
        self.bit_set.insert(elem)
    }

    /// Returns `true` if the set has changed.
    #[inline]
    pub fn remove(&mut self, elem: T) -> bool {
        self.ensure(elem.index() + 1);
        self.bit_set.remove(elem)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bit_set.is_empty()
    }

    #[inline]
    pub fn contains(&self, elem: T) -> bool {
        let (word_index, mask) = byte_index_and_mask(elem.index());
        let (words, domain_size) = self.bit_set.raw_parts();
        words
            .get(word_index)
            .map_or(false, |word| (word & mask) != 0)
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        self.bit_set.iter()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.bit_set.count()
    }
}

impl<T: Idx> From<BitSet<T>> for GrowableBitSet<T> {
    fn from(bit_set: BitSet<T>) -> Self {
        Self { bit_set }
    }
}
