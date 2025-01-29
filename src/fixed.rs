use std::{fmt, marker::PhantomData, ops::RangeBounds};

use crate::{
    bit_relations_inherent_impls, bitwise, byte_index_and_mask, dense::DenseBitSet,
    inclusive_start_end, num_bytes, num_words, word_index_and_mask, BitIter, BitRelations,
    EitherIter, Idx, Word, WORD_BITS
};
/*
use crate::{
    sequential_update, Chunk, ChunkedBitSet, GrowableBitSet, SparseBitSet, CHUNK_WORDS,
};
*/

/// A fixed-size bitset type with a dense representation.
///
/// NOTE: Use [`GrowableBitSet`] if you need support for resizing after
/// creation.
///
/// `T` is an index type, typically a newtyped `usize` wrapper, but it can also
/// just be `usize`.
///
/// All operations that involve an element will panic if the element is equal
/// to or greater than the domain size. All operations that involve two bitsets
/// will panic if the bitsets have differing domain sizes.
#[derive(Eq, PartialEq, Hash)]
pub struct BitSet<T> {
    pub(crate) inner: BitSetImpl,
    pub(crate) marker: PhantomData<T>,
}

#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub(crate) enum BitSetImpl {
    Inline(DenseBitSet),
    Heap {
        domain_size: usize,
        words: Box<[u64]>,
    },
}

const INLINE_BITSET_BYTES: usize = 31;
const INLINE_BITSET_BITS: usize = INLINE_BITSET_BYTES * 8;

impl<T: Idx> BitSet<T> {
    /// Creates a new, empty bitset with a given `domain_size`.
    #[inline]
    pub fn new_empty(domain_size: usize) -> BitSet<T> {
        let inner = if domain_size <= INLINE_BITSET_BITS {
            BitSetImpl::Inline(DenseBitSet::new_empty(domain_size))
        } else {
            let num_words = num_words(domain_size);
            BitSetImpl::Heap {
                domain_size,
                words: vec![0; num_words].into_boxed_slice(),
            }
        };
        BitSet {
            inner,
            marker: PhantomData,
        }
    }

    /// Creates a new, filled bitset with a given `domain_size`.
    #[inline]
    pub fn new_filled(domain_size: usize) -> BitSet<T> {
        let mut this = if domain_size <= INLINE_BITSET_BITS {
            let mut inner = DenseBitSet::new_empty(domain_size);
            inner.words_mut(|words| words.fill(!0));
            BitSet {
                inner: BitSetImpl::Inline(inner),
                marker: PhantomData,
            }
        } else {
            let num_words = num_words(domain_size);
            BitSet {
                inner: BitSetImpl::Heap {
                    domain_size,
                    words: vec![!0; num_words].into_boxed_slice(),
                },
                marker: PhantomData,
            }
        };
        this.clear_excess_bits();
        this
    }

    /// Gets the domain size.
    #[inline]
    pub fn domain_size(&self) -> usize {
        match &self.inner {
            BitSetImpl::Inline(inline) => inline.domain_size(),
            BitSetImpl::Heap { domain_size, .. } => *domain_size,
        }
    }

    #[inline]
    fn words_mut<R, F: FnOnce(&mut [Word]) -> R>(&mut self, f: F) -> R {
        match &mut self.inner {
            BitSetImpl::Inline(inline) => inline.words_mut(f),
            BitSetImpl::Heap { words, domain_size } => {
                let used_words = num_words(*domain_size);
                f(&mut words[..used_words])
            }
        }
    }

    #[inline]
    pub(crate) fn raw_parts(&self) -> (&[u8], usize) {
        match &self.inner {
            BitSetImpl::Inline(inline) => inline.raw_parts(),
            BitSetImpl::Heap { domain_size, words } => {
                let bytes =
                    unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), words.len() * 8) };
                (bytes, *domain_size)
            }
        }
    }

    #[inline]
    fn raw_parts_mut(&mut self) -> (&mut [u8], usize) {
        match &mut self.inner {
            BitSetImpl::Inline(inline) => inline.raw_parts_mut(),
            BitSetImpl::Heap { domain_size, words } => {
                let bytes = unsafe {
                    std::slice::from_raw_parts_mut(words.as_mut_ptr().cast(), words.len() * 8)
                };
                (bytes, *domain_size)
            }
        }
    }

    /// Clear all elements.
    #[inline]
    pub fn clear(&mut self) {
        self.words_mut(|words| words.fill(0));
    }

    /// Clear excess bits in the final word.
    #[inline]
    fn clear_excess_bits(&mut self) {
        let domain_size = self.domain_size();
        let num_bits_in_final_word = domain_size % WORD_BITS;
        let mask = if num_bits_in_final_word > 0 {
            (1 << num_bits_in_final_word) - 1
        } else {
            return;
        };
        self.words_mut(|words| {
            *words.last_mut().unwrap() &= mask;
        });
    }

    /// Count the number of set bits in the set.
    #[inline]
    pub fn count(&self) -> usize {
        let (bytes, domain_length) = self.raw_parts();
        bytes[..num_bytes(domain_length)]
            .iter()
            .map(|e| e.count_ones() as usize)
            .sum()
    }

    /// Returns `true` if `self` contains `elem`.
    #[inline]
    pub fn contains(&self, elem: T) -> bool {
        let (index, mask) = byte_index_and_mask(elem.index());
        let (words, domain_size) = self.raw_parts();
        assert!(elem.index() < domain_size);
        debug_assert!(index < words.len());
        unsafe { (words.get_unchecked(index) & mask) != 0 }
    }

    /// Is `self` is a (non-strict) superset of `other`?
    #[inline]
    pub fn superset(&self, other: &BitSet<T>) -> bool {
        assert_eq!(self.domain_size(), other.domain_size());
        self.words()
            .zip(other.words())
            .all(|(a, b)| (a & b) == b)
    }

    /// Is the set empty?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.iter().next().is_none()
    }

    /// Insert `elem`. Returns whether the set has changed.
    #[inline(never)]
    pub fn insert(&mut self, elem: T) -> bool {
        let bit = elem.index();
        let (words, domain_size) = self.raw_parts_mut();
        assert!(bit < domain_size);
        let (index, mask) = byte_index_and_mask(bit);
        debug_assert!(index < words.len());
        let word_ref = unsafe { words.get_unchecked_mut(index) };
        let word = *word_ref;
        let new_word = word | mask;
        *word_ref = new_word;
        new_word != word
    }

    #[inline]
    pub fn insert_range(&mut self, elems: impl RangeBounds<T>) {
        let Some((start, end)) = inclusive_start_end(elems, self.domain_size()) else {
            return;
        };

        let (start_word_index, start_mask) = word_index_and_mask(start);
        let (end_word_index, end_mask) = word_index_and_mask(end);

        self.words_mut(|words| {
            // Set all words in between start and end (exclusively of both).
            words[start_word_index + 1..end_word_index].fill(!0);

            if start_word_index != end_word_index {
                // Start and end are in different words, so we handle each in turn.
                //
                // We set all leading bits. This includes the start_mask bit.
                words[start_word_index] |= !(start_mask - 1);
                // And all trailing bits (i.e. from 0..=end) in the end word,
                // including the end.
                words[end_word_index] |= end_mask | (end_mask - 1);
            } else {
                words[start_word_index] |= end_mask | (end_mask - start_mask);
            }
        });
    }

    /// Sets all bits to true.
    #[inline]
    pub fn insert_all(&mut self) {
        self.words_mut(|words| {
            words.fill(!0);
        });
        self.clear_excess_bits();
    }

    /// Returns `true` if the set has changed.
    #[inline]
    pub fn remove(&mut self, elem: T) -> bool {
        let bit = elem.index();
        let (words, domain_size) = self.raw_parts_mut();
        assert!(bit < domain_size);
        let (index, mask) = byte_index_and_mask(bit);
        let word_ref = &mut words[index];
        let word = *word_ref;
        let new_word = word & !mask;
        *word_ref = new_word;
        new_word != word
    }

    /// Iterates over the indices of set bits in a sorted order.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        BitIter::new(self.words())
    }

    #[inline]
    pub fn words(&self) -> impl Iterator<Item = Word> + '_ {
        match &self.inner {
            BitSetImpl::Inline(inline) => EitherIter::Left(inline.words()),
            BitSetImpl::Heap { words, .. } => EitherIter::Right(words.iter().cloned()),
        }
    }

    /*
    /// Set `self = self | other`. In contrast to `union` returns `true` if the set contains at
    /// least one bit that is not in `other` (i.e. `other` is not a superset of `self`).
    ///
    /// This is an optimization for union of a hybrid bitset.
    pub fn reverse_union_sparse(&mut self, sparse: &SparseBitSet<T>) -> bool {
        assert!(sparse.domain_size == self.domain_size());
        self.clear_excess_bits();

        let mut not_already = false;
        // Index of the current word not yet merged.
        let mut current_index = 0;
        // Mask of bits that came from the sparse set in the current word.
        let mut new_bit_mask = 0;
        for (word_index, mask) in sparse.iter().map(|x| word_index_and_mask(*x)) {
            // Next bit is in a word not inspected yet.
            if word_index > current_index {
                self.words_mut()[current_index] |= new_bit_mask;
                // Were there any bits in the old word that did not occur in the sparse set?
                not_already |= (self.words()[current_index] ^ new_bit_mask) != 0;
                // Check all words we skipped for any set bit.
                not_already |= self.words()[current_index + 1..word_index].iter().any(|&x| x != 0);
                // Update next word.
                current_index = word_index;
                // Reset bit mask, no bits have been merged yet.
                new_bit_mask = 0;
            }
            // Add bit and mark it as coming from the sparse set.
            // self.words[word_index] |= mask;
            new_bit_mask |= mask;
        }
        self.words_mut()[current_index] |= new_bit_mask;
        // Any bits in the last inspected word that were not in the sparse set?
        not_already |= (self.words()[current_index] ^ new_bit_mask) != 0;
        // Any bits in the tail? Note `clear_excess_bits` before.
        not_already |= self.words()[current_index + 1..].iter().any(|&x| x != 0);

        not_already
    }
    */

    // FIXME: This is an iterating algorithm, but I'm implementing this with byte access right now
    // instead of word access.
    pub fn last_set_in(&self, range: impl RangeBounds<T>) -> Option<T> {

        #[inline]
        fn max_bit(word: u8) -> usize {
            8 - 1 - word.leading_zeros() as usize
        }

        let (words, domain_size) = self.raw_parts();
        let (start, end) = inclusive_start_end(range, domain_size)?;
        let (start_word_index, _) = byte_index_and_mask(start);
        let (end_word_index, end_mask) = byte_index_and_mask(end);

        let end_word = words[end_word_index] & (end_mask | (end_mask - 1));
        if end_word != 0 {
            let pos = max_bit(end_word) + 8 * end_word_index;
            if start <= pos {
                return Some(T::new(pos));
            }
        }

        // We exclude end_word_index from the range here, because we don't want
        // to limit ourselves to *just* the last word: the bits set it in may be
        // after `end`, so it may not work out.
        if let Some(offset) = words[start_word_index..end_word_index]
            .iter()
            .rposition(|&w| w != 0)
        {
            let word_idx = start_word_index + offset;
            let start_word = words[word_idx];
            let pos = max_bit(start_word) + 8 * word_idx;
            if start <= pos {
                return Some(T::new(pos));
            }
        }

        None
    }

    bit_relations_inherent_impls! {}
}

// dense REL dense
impl<T: Idx> BitRelations<BitSet<T>> for BitSet<T> {
    fn union(&mut self, other: &BitSet<T>) -> bool {
        match (&mut self.inner, &other.inner) {
            (BitSetImpl::Inline(s), BitSetImpl::Inline(o)) => {
                assert_eq!(s.domain_size(), o.domain_size());
                s.union(o)
            }
            (
                BitSetImpl::Heap {
                    words: s,
                    domain_size: s_domain_size,
                },
                BitSetImpl::Heap {
                    words: o,
                    domain_size: o_domain_size,
                },
            ) => {
                assert_eq!(s_domain_size, o_domain_size);
                bitwise(&mut s[..], o.iter().copied(), |a, b| a | b)
            }
            _ => unreachable!(),
        }
    }

    fn subtract(&mut self, other: &BitSet<T>) -> bool {
        match (&mut self.inner, &other.inner) {
            (BitSetImpl::Inline(s), BitSetImpl::Inline(o)) => {
                assert_eq!(s.domain_size(), o.domain_size());
                s.words_mut(|words| bitwise(words, o.words(), |a, b| a & !b))
            }
            (
                BitSetImpl::Heap {
                    words: s,
                    domain_size: s_domain_size,
                },
                BitSetImpl::Heap {
                    words: o,
                    domain_size: o_domain_size,
                },
            ) => {
                assert_eq!(s_domain_size, o_domain_size);
                bitwise(&mut s[..], o.iter().copied(), |a, b| a & !b)
            }
            _ => unreachable!(),
        }
    }

    fn intersect(&mut self, other: &BitSet<T>) -> bool {
        match (&mut self.inner, &other.inner) {
            (BitSetImpl::Inline(s), BitSetImpl::Inline(o)) => {
                assert_eq!(s.domain_size(), o.domain_size());
                s.intersect(o)
            }
            (
                BitSetImpl::Heap {
                    words: s,
                    domain_size: s_domain_size,
                },
                BitSetImpl::Heap {
                    words: o,
                    domain_size: o_domain_size,
                },
            ) => {
                assert_eq!(s_domain_size, o_domain_size);
                bitwise(&mut s[..], o.iter().copied(), |a, b| a & b)
            }
            _ => unreachable!(),
        }
    }
}

/*
impl<T: Idx> From<GrowableBitSet<T>> for BitSet<T> {
    fn from(bit_set: GrowableBitSet<T>) -> Self {
        let mut new = BitSet::new_empty(bit_set.domain_size);
        for bit in bit_set.iter() {
            new.insert(bit);
        }
        new
    }
}
*/

impl<T> Clone for BitSet<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            marker: PhantomData,
        }
    }

    #[inline]
    fn clone_from(&mut self, from: &Self) {
        #[inline]
        fn clone_from_heap<T>(this: &mut BitSet<T>, from: &BitSet<T>) {
            *this = from.clone();
        }

        match &from.inner {
            BitSetImpl::Inline(inline) => {
                *self = BitSet {
                    inner: BitSetImpl::Inline(*inline),
                    marker: PhantomData,
                };
            }
            BitSetImpl::Heap { .. } => clone_from_heap(self, from),
        }
    }
}

use std::fmt::Debug;
impl<T: Idx + Debug> Debug for BitSet<T> {
    fn fmt(&self, w: &mut fmt::Formatter<'_>) -> fmt::Result {
        w.debug_list().entries(self.iter()).finish()
    }
}

/*
impl<T: Idx> BitRelations<ChunkedBitSet<T>> for BitSet<T> {
    fn union(&mut self, other: &ChunkedBitSet<T>) -> bool {
        sequential_update(|elem| self.insert(elem), other.iter())
    }

    fn subtract(&mut self, _other: &ChunkedBitSet<T>) -> bool {
        unimplemented!("implement if/when necessary");
    }

    fn intersect(&mut self, other: &ChunkedBitSet<T>) -> bool {
        assert_eq!(self.domain_size(), other.domain_size);
        let mut changed = false;
        for (i, chunk) in other.chunks.iter().enumerate() {
            let mut words = &mut self.words_mut()[i * CHUNK_WORDS..];
            if words.len() > CHUNK_WORDS {
                words = &mut words[..CHUNK_WORDS];
            }
            match chunk {
                Chunk::Zeros(..) => {
                    for word in words {
                        if *word != 0 {
                            changed = true;
                            *word = 0;
                        }
                    }
                }
                Chunk::Ones(..) => (),
                Chunk::Mixed(_, _, _data) => {
                    unimplemented!("Stop being lazy");
                    /*
                    for (i, word) in words.iter_mut().enumerate() {
                        let new_val = *word & data[i];
                        if new_val != *word {
                            changed = true;
                            *word = new_val;
                        }
                    }
                    */
                }
            }
        }
        changed
    }
}
*/



#[cfg(test)]
mod dense_tests {
    use super::*;

    #[test]
    fn new_filled() {
        for size in 0..300 {
            let mut set: BitSet<usize> = BitSet::new_filled(size);
            assert_eq!(size, set.domain_size());
            assert_eq!(size, set.count());
            assert_eq!(size, set.iter().count());
            set.clear();
            assert_eq!(set, BitSet::new_empty(size));
        }
    }

    #[test]
    fn every_bit_works() {
        for size in 0..300 {
            let mut set = BitSet::new_empty(size);
            assert_eq!(size, set.domain_size());
            assert!(set.is_empty());
            for bit in 0..size {
                set.insert(bit);
                assert!(set.contains(bit));
                assert_eq!(set.count(), 1);
                {
                    let mut it = set.iter();
                    assert_eq!(it.next(), Some(bit));
                    assert_eq!(it.next(), None);
                }
                assert_eq!(size, set.domain_size());

                assert_eq!(set, set.clone());

                set.remove(bit);
                assert!(!set.contains(bit));
                assert_eq!(set.count(), 0);
                assert_eq!(set.iter().next(), None);
                assert_eq!(size, set.domain_size());

                assert_eq!(set, BitSet::new_empty(size));
            }
        }
    }

    #[test]
    fn union_two_sets() {
        let mut set1: BitSet<usize> = BitSet::new_empty(65);
        let mut set2: BitSet<usize> = BitSet::new_empty(65);
        assert!(set1.insert(3));
        assert!(!set1.insert(3));
        assert!(set2.insert(5));
        assert!(set2.insert(64));
        assert!(set1.union(&set2));
        assert!(!set1.union(&set2));
        assert!(set1.contains(3));
        assert!(!set1.contains(4));
        assert!(set1.contains(5));
        assert!(!set1.contains(63));
        assert!(set1.contains(64));
    }

    #[test]
    fn soundness() {
        let size = 31 * 8;
        let mut set = BitSet::new_empty(size);
        for i in 0..size {
            set.insert(i);
        }
        let _payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            set.words_mut(|f| {
                for word in f {
                    *word = 0;
                }
                panic!("ouch");
            })
        }));
        println!("{set:?}");
    }
}
