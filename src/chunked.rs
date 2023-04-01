use crate::{
    bit_relations_inherent_impls, bitwise, num_words, word_index_and_mask, BitIter, BitRelations,
    Idx, Word, WORD_BITS,
};
use std::{marker::PhantomData, rc::Rc};

use Chunk::*;

// The choice of chunk size has some trade-offs.
//
// A big chunk size tends to favour cases where many large `ChunkedBitSet`s are
// present, because they require fewer `Chunk`s, reducing the number of
// allocations and reducing peak memory usage. Also, fewer chunk operations are
// required, though more of them might be `Mixed`.
//
// A small chunk size tends to favour cases where many small `ChunkedBitSet`s
// are present, because less space is wasted at the end of the final chunk (if
// it's not full).
const CHUNK_WORDS: usize = 32;
const CHUNK_BITS: usize = CHUNK_WORDS * WORD_BITS; // 2048 bits

/// ChunkSize is small to keep `Chunk` small. The static assertion ensures it's
/// not too small.
type ChunkSize = u16;
const _: () = assert!(CHUNK_BITS <= ChunkSize::MAX as usize);

/// A fixed-size bitset type with a partially dense, partially sparse
/// representation. The bitset is broken into chunks, and chunks that are all
/// zeros or all ones are represented and handled very efficiently.
///
/// This type is especially efficient for sets that typically have a large
/// `domain_size` with significant stretches of all zeros or all ones, and also
/// some stretches with lots of 0s and 1s mixed in a way that causes trouble
/// for `IntervalSet`.
///
/// `T` is an index type, typically a newtyped `usize` wrapper, but it can also
/// just be `usize`.
///
/// All operations that involve an element will panic if the element is equal
/// to or greater than the domain size. All operations that involve two bitsets
/// will panic if the bitsets have differing domain sizes.
#[derive(Debug, PartialEq, Eq)]
pub struct ChunkedBitSet<T> {
    domain_size: usize,

    /// The chunks. Each one contains exactly CHUNK_BITS values, except the
    /// last one which contains 1..=CHUNK_BITS values.
    chunks: Box<[Chunk]>,

    marker: PhantomData<T>,
}

// Note: the chunk domain size is duplicated in each variant. This is a bit
// inconvenient, but it allows the type size to be smaller than if we had an
// outer struct containing a chunk domain size plus the `Chunk`, because the
// compiler can place the chunk domain size after the tag.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Chunk {
    /// A chunk that is all zeros; we don't represent the zeros explicitly.
    Zeros(ChunkSize),

    /// A chunk that is all ones; we don't represent the ones explicitly.
    Ones(ChunkSize),

    /// A chunk that has a mix of zeros and ones, which are represented
    /// explicitly and densely. It never has all zeros or all ones.
    ///
    /// If this is the final chunk there may be excess, unused words. This
    /// turns out to be both simpler and have better performance than
    /// allocating the minimum number of words, largely because we avoid having
    /// to store the length, which would make this type larger. These excess
    /// words are always be zero, as are any excess bits in the final in-use
    /// word.
    ///
    /// The second field is the count of 1s set in the chunk, and must satisfy
    /// `0 < count < chunk_domain_size`.
    ///
    /// The words are within an `Rc` because it's surprisingly common to
    /// duplicate an entire chunk, e.g. in `ChunkedBitSet::clone_from()`, or
    /// when a `Mixed` chunk is union'd into a `Zeros` chunk. When we do need
    /// to modify a chunk we use `Rc::make_mut`.
    Mixed(ChunkSize, ChunkSize, Rc<[Word; CHUNK_WORDS]>),
}

/*
// This type is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
crate::static_assert_size!(Chunk, 16);
*/

impl<T> ChunkedBitSet<T> {
    pub fn domain_size(&self) -> usize {
        self.domain_size
    }

    #[cfg(test)]
    fn assert_valid(&self) {
        if self.domain_size == 0 {
            assert!(self.chunks.is_empty());
            return;
        }

        assert!((self.chunks.len() - 1) * CHUNK_BITS <= self.domain_size);
        assert!(self.chunks.len() * CHUNK_BITS >= self.domain_size);
        for chunk in self.chunks.iter() {
            chunk.assert_valid();
        }
    }
}

impl<T: Idx> ChunkedBitSet<T> {
    /// Creates a new bitset with a given `domain_size` and chunk kind.
    fn new(domain_size: usize, is_empty: bool) -> Self {
        let chunks = if domain_size == 0 {
            Box::new([])
        } else {
            // All the chunks have a chunk_domain_size of `CHUNK_BITS` except
            // the final one.
            let final_chunk_domain_size = {
                let n = domain_size % CHUNK_BITS;
                if n == 0 {
                    CHUNK_BITS
                } else {
                    n
                }
            };
            let mut chunks =
                vec![Chunk::new(CHUNK_BITS, is_empty); num_chunks(domain_size)].into_boxed_slice();
            *chunks.last_mut().unwrap() = Chunk::new(final_chunk_domain_size, is_empty);
            chunks
        };
        ChunkedBitSet {
            domain_size,
            chunks,
            marker: PhantomData,
        }
    }

    /// Creates a new, empty bitset with a given `domain_size`.
    #[inline]
    pub fn new_empty(domain_size: usize) -> Self {
        ChunkedBitSet::new(domain_size, /* is_empty */ true)
    }

    /// Creates a new, filled bitset with a given `domain_size`.
    #[inline]
    pub fn new_filled(domain_size: usize) -> Self {
        ChunkedBitSet::new(domain_size, /* is_empty */ false)
    }

    #[cfg(test)]
    fn chunks(&self) -> &[Chunk] {
        &self.chunks
    }

    /// Count the number of bits in the set.
    pub fn count(&self) -> usize {
        self.chunks.iter().map(|chunk| chunk.count()).sum()
    }

    /// Returns `true` if `self` contains `elem`.
    #[inline]
    pub fn contains(&self, elem: T) -> bool {
        assert!(elem.index() < self.domain_size);
        let chunk = &self.chunks[chunk_index(elem)];
        match &chunk {
            Zeros(_) => false,
            Ones(_) => true,
            Mixed(_, _, words) => {
                let (word_index, mask) = chunk_word_index_and_mask(elem);
                (words[word_index] & mask) != 0
            }
        }
    }

    #[inline]
    pub fn iter(&self) -> ChunkedBitIter<'_, T> {
        ChunkedBitIter::new(self)
    }

    /// Insert `elem`. Returns whether the set has changed.
    pub fn insert(&mut self, elem: T) -> bool {
        assert!(elem.index() < self.domain_size);
        let chunk_index = chunk_index(elem);
        let chunk = &mut self.chunks[chunk_index];
        match *chunk {
            Zeros(chunk_domain_size) => {
                if chunk_domain_size > 1 {
                    // We take some effort to avoid copying the words.
                    let words = Rc::<[Word; CHUNK_WORDS]>::new_zeroed();
                    // SAFETY: `words` can safely be all zeroes.
                    let mut words = unsafe { words.assume_init() };
                    let words_ref = Rc::get_mut(&mut words).unwrap();

                    let (word_index, mask) = chunk_word_index_and_mask(elem);
                    words_ref[word_index] |= mask;
                    *chunk = Mixed(chunk_domain_size, 1, words);
                } else {
                    *chunk = Ones(chunk_domain_size);
                }
                true
            }
            Ones(_) => false,
            Mixed(chunk_domain_size, ref mut count, ref mut words) => {
                // We skip all the work if the bit is already set.
                let (word_index, mask) = chunk_word_index_and_mask(elem);
                if (words[word_index] & mask) == 0 {
                    *count += 1;
                    if *count < chunk_domain_size {
                        let words = Rc::make_mut(words);
                        words[word_index] |= mask;
                    } else {
                        *chunk = Ones(chunk_domain_size);
                    }
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Sets all bits to true.
    pub fn insert_all(&mut self) {
        for chunk in self.chunks.iter_mut() {
            *chunk = match *chunk {
                Zeros(chunk_domain_size)
                | Ones(chunk_domain_size)
                | Mixed(chunk_domain_size, ..) => Ones(chunk_domain_size),
            }
        }
    }

    /// Returns `true` if the set has changed.
    pub fn remove(&mut self, elem: T) -> bool {
        assert!(elem.index() < self.domain_size);
        let chunk_index = chunk_index(elem);
        let chunk = &mut self.chunks[chunk_index];
        match *chunk {
            Zeros(_) => false,
            Ones(chunk_domain_size) => {
                if chunk_domain_size > 1 {
                    // We take some effort to avoid copying the words.
                    let words = Rc::<[Word; CHUNK_WORDS]>::new_zeroed();
                    // SAFETY: `words` can safely be all zeroes.
                    let mut words = unsafe { words.assume_init() };
                    let words_ref = Rc::get_mut(&mut words).unwrap();

                    // Set only the bits in use.
                    let num_words = num_words(chunk_domain_size as usize);
                    words_ref[..num_words].fill(!0);
                    clear_excess_bits_in_final_word(
                        chunk_domain_size as usize,
                        &mut words_ref[..num_words],
                    );
                    let (word_index, mask) = chunk_word_index_and_mask(elem);
                    words_ref[word_index] &= !mask;
                    *chunk = Mixed(chunk_domain_size, chunk_domain_size - 1, words);
                } else {
                    *chunk = Zeros(chunk_domain_size);
                }
                true
            }
            Mixed(chunk_domain_size, ref mut count, ref mut words) => {
                // We skip all the work if the bit is already clear.
                let (word_index, mask) = chunk_word_index_and_mask(elem);
                if (words[word_index] & mask) != 0 {
                    *count -= 1;
                    if *count > 0 {
                        let words = Rc::make_mut(words);
                        words[word_index] &= !mask;
                    } else {
                        *chunk = Zeros(chunk_domain_size);
                    }
                    true
                } else {
                    false
                }
            }
        }
    }

    bit_relations_inherent_impls! {}
}

impl<T: Idx> BitRelations<ChunkedBitSet<T>> for ChunkedBitSet<T> {
    fn union(&mut self, other: &ChunkedBitSet<T>) -> bool {
        assert_eq!(self.domain_size, other.domain_size);
        debug_assert_eq!(self.chunks.len(), other.chunks.len());

        let mut changed = false;
        for (mut self_chunk, other_chunk) in self.chunks.iter_mut().zip(other.chunks.iter()) {
            match (&mut self_chunk, &other_chunk) {
                (_, Zeros(_)) | (Ones(_), _) => {}
                (Zeros(self_chunk_domain_size), Ones(other_chunk_domain_size))
                | (Mixed(self_chunk_domain_size, ..), Ones(other_chunk_domain_size))
                | (Zeros(self_chunk_domain_size), Mixed(other_chunk_domain_size, ..)) => {
                    // `other_chunk` fully overwrites `self_chunk`
                    debug_assert_eq!(self_chunk_domain_size, other_chunk_domain_size);
                    *self_chunk = other_chunk.clone();
                    changed = true;
                }
                (
                    Mixed(
                        self_chunk_domain_size,
                        ref mut self_chunk_count,
                        ref mut self_chunk_words,
                    ),
                    Mixed(_other_chunk_domain_size, _other_chunk_count, other_chunk_words),
                ) => {
                    // First check if the operation would change
                    // `self_chunk.words`. If not, we can avoid allocating some
                    // words, and this happens often enough that it's a
                    // performance win. Also, we only need to operate on the
                    // in-use words, hence the slicing.
                    let op = |a, b| a | b;
                    let num_words = num_words(*self_chunk_domain_size as usize);
                    if bitwise_changes(
                        &self_chunk_words[0..num_words],
                        &other_chunk_words[0..num_words],
                        op,
                    ) {
                        let self_chunk_words = Rc::make_mut(self_chunk_words);
                        let has_changed = bitwise(
                            &mut self_chunk_words[0..num_words],
                            other_chunk_words[0..num_words].iter().copied(),
                            op,
                        );
                        debug_assert!(has_changed);
                        *self_chunk_count = self_chunk_words[0..num_words]
                            .iter()
                            .map(|w| w.count_ones() as ChunkSize)
                            .sum();
                        if *self_chunk_count == *self_chunk_domain_size {
                            *self_chunk = Ones(*self_chunk_domain_size);
                        }
                        changed = true;
                    }
                }
            }
        }
        changed
    }

    fn subtract(&mut self, _other: &ChunkedBitSet<T>) -> bool {
        unimplemented!("implement if/when necessary");
    }

    fn intersect(&mut self, _other: &ChunkedBitSet<T>) -> bool {
        unimplemented!("implement if/when necessary");
    }
}

/*
impl<T: Idx> BitRelations<HybridBitSet<T>> for ChunkedBitSet<T> {
    fn union(&mut self, other: &HybridBitSet<T>) -> bool {
        // FIXME: This is slow if `other` is dense, but it hasn't been a problem
        // in practice so far.
        // If a faster implementation of this operation is required, consider
        // reopening https://github.com/rust-lang/rust/pull/94625
        assert_eq!(self.domain_size, other.domain_size());
        sequential_update(|elem| self.insert(elem), other.iter())
    }

    fn subtract(&mut self, other: &HybridBitSet<T>) -> bool {
        // FIXME: This is slow if `other` is dense, but it hasn't been a problem
        // in practice so far.
        // If a faster implementation of this operation is required, consider
        // reopening https://github.com/rust-lang/rust/pull/94625
        assert_eq!(self.domain_size, other.domain_size());
        sequential_update(|elem| self.remove(elem), other.iter())
    }

    fn intersect(&mut self, _other: &HybridBitSet<T>) -> bool {
        unimplemented!("implement if/when necessary");
    }
}
*/

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
            let mut words = &mut self.words[i * CHUNK_WORDS..];
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
                Chunk::Mixed(_, _, data) => {
                    for (i, word) in words.iter_mut().enumerate() {
                        let new_val = *word & data[i];
                        if new_val != *word {
                            changed = true;
                            *word = new_val;
                        }
                    }
                }
            }
        }
        changed
    }
}
*/

impl<T> Clone for ChunkedBitSet<T> {
    fn clone(&self) -> Self {
        ChunkedBitSet {
            domain_size: self.domain_size,
            chunks: self.chunks.clone(),
            marker: PhantomData,
        }
    }

    /// WARNING: this implementation of clone_from will panic if the two
    /// bitsets have different domain sizes. This constraint is not inherent to
    /// `clone_from`, but it works with the existing call sites and allows a
    /// faster implementation, which is important because this function is hot.
    fn clone_from(&mut self, from: &Self) {
        assert_eq!(self.domain_size, from.domain_size);
        debug_assert_eq!(self.chunks.len(), from.chunks.len());

        self.chunks.clone_from(&from.chunks)
    }
}

pub struct ChunkedBitIter<'a, T: Idx> {
    index: usize,
    bitset: &'a ChunkedBitSet<T>,
}

impl<'a, T: Idx> ChunkedBitIter<'a, T> {
    #[inline]
    fn new(bitset: &'a ChunkedBitSet<T>) -> ChunkedBitIter<'a, T> {
        ChunkedBitIter { index: 0, bitset }
    }
}

impl<'a, T: Idx> Iterator for ChunkedBitIter<'a, T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        while self.index < self.bitset.domain_size() {
            let elem = T::new(self.index);
            let chunk = &self.bitset.chunks[chunk_index(elem)];
            match &chunk {
                Zeros(chunk_domain_size) => {
                    self.index += *chunk_domain_size as usize;
                }
                Ones(_chunk_domain_size) => {
                    self.index += 1;
                    return Some(elem);
                }
                Mixed(_chunk_domain_size, _, words) => loop {
                    let elem = T::new(self.index);
                    self.index += 1;
                    let (word_index, mask) = chunk_word_index_and_mask(elem);
                    if (words[word_index] & mask) != 0 {
                        return Some(elem);
                    }
                    if self.index % CHUNK_BITS == 0 {
                        break;
                    }
                },
            }
        }
        None
    }

    fn fold<B, F>(mut self, mut init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        // If `next` has already been called, we may not be at the start of a chunk, so
        // we first advance the iterator to the start of the next chunk, before
        // proceeding in chunk sized steps.
        while self.index % CHUNK_BITS != 0 {
            let Some(item) = self.next() else {
                return init
            };
            init = f(init, item);
        }
        let start_chunk = self.index / CHUNK_BITS;
        let chunks = &self.bitset.chunks[start_chunk..];
        for (i, chunk) in chunks.iter().enumerate() {
            let base = (start_chunk + i) * CHUNK_BITS;
            match chunk {
                Chunk::Zeros(_) => (),
                Chunk::Ones(limit) => {
                    for j in 0..(*limit as usize) {
                        init = f(init, T::new(base + j));
                    }
                }
                Chunk::Mixed(_, _, words) => {
                    init = BitIter::new(words.iter().copied()).fold(init, |val, mut item: T| {
                        item.increment_by(base);
                        f(val, item)
                    });
                }
            }
        }
        init
    }
}

impl Chunk {
    #[cfg(test)]
    fn assert_valid(&self) {
        match *self {
            Zeros(chunk_domain_size) | Ones(chunk_domain_size) => {
                assert!(chunk_domain_size as usize <= CHUNK_BITS);
            }
            Mixed(chunk_domain_size, count, ref words) => {
                assert!(chunk_domain_size as usize <= CHUNK_BITS);
                assert!(0 < count && count < chunk_domain_size);

                // Check the number of set bits matches `count`.
                assert_eq!(
                    words
                        .iter()
                        .map(|w| w.count_ones() as ChunkSize)
                        .sum::<ChunkSize>(),
                    count
                );

                // Check the not-in-use words are all zeroed.
                let num_words = num_words(chunk_domain_size as usize);
                if num_words < CHUNK_WORDS {
                    assert_eq!(
                        words[num_words..]
                            .iter()
                            .map(|w| w.count_ones() as ChunkSize)
                            .sum::<ChunkSize>(),
                        0
                    );
                }
            }
        }
    }

    fn new(chunk_domain_size: usize, is_empty: bool) -> Self {
        debug_assert!(chunk_domain_size <= CHUNK_BITS);
        let chunk_domain_size = chunk_domain_size as ChunkSize;
        if is_empty {
            Zeros(chunk_domain_size)
        } else {
            Ones(chunk_domain_size)
        }
    }

    /// Count the number of 1s in the chunk.
    fn count(&self) -> usize {
        match *self {
            Zeros(_) => 0,
            Ones(chunk_domain_size) => chunk_domain_size as usize,
            Mixed(_, count, _) => count as usize,
        }
    }
}

// Applies a function to mutate a bitset, and returns true if any
// of the applications return true
fn sequential_update<T: Idx>(
    mut self_update: impl FnMut(T) -> bool,
    it: impl Iterator<Item = T>,
) -> bool {
    it.fold(false, |changed, elem| self_update(elem) | changed)
}

/*
// Optimization of intersection for SparseBitSet that's generic
// over the RHS
fn sparse_intersect<T: Idx>(
    set: &mut SparseBitSet<T>,
    other_contains: impl Fn(&T) -> bool,
) -> bool {
    let size = set.elems.len();
    set.elems.retain(|elem| other_contains(elem));
    set.elems.len() != size
}

// Optimization of dense/sparse intersection. The resulting set is
// guaranteed to be at most the size of the sparse set, and hence can be
// represented as a sparse set. Therefore the sparse set is copied and filtered,
// then returned as the new set.
fn dense_sparse_intersect<T: Idx>(
    dense: &BitSet<T>,
    sparse: &SparseBitSet<T>,
) -> (SparseBitSet<T>, bool) {
    let mut sparse_copy = sparse.clone();
    sparse_intersect(&mut sparse_copy, |el| dense.contains(*el));
    let n = sparse_copy.len();
    (sparse_copy, n != dense.count())
}
*/

#[inline]
fn num_chunks<T: Idx>(domain_size: T) -> usize {
    assert!(domain_size.index() > 0);
    (domain_size.index() + CHUNK_BITS - 1) / CHUNK_BITS
}

#[inline]
fn chunk_index<T: Idx>(elem: T) -> usize {
    elem.index() / CHUNK_BITS
}

#[inline]
fn chunk_word_index_and_mask<T: Idx>(elem: T) -> (usize, Word) {
    let chunk_elem = elem.index() % CHUNK_BITS;
    word_index_and_mask(chunk_elem)
}
