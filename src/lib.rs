#![feature(new_uninit)]

use std::{
    fmt::Debug,
    hash::Hash,
    ops::{Bound, RangeBounds},
};

mod chunked;
mod dense;
mod finite;
mod fixed;
mod growable;
mod hybrid;
mod iter;
mod nonmaxu8;
mod sparse;
//mod matrix;
//mod sparse_matrix;

pub use chunked::ChunkedBitSet;
pub use finite::FiniteBitSet;
pub use fixed::BitSet;
pub use growable::GrowableBitSet;
pub use hybrid::HybridBitSet;
use iter::*;
//pub use matrix::BitMatrix;
//pub use sparse_matrix::SparseBitMatrix;

pub trait Idx: Copy + 'static + Eq + PartialEq + Debug + Hash {
    fn new(_: usize) -> Self;
    fn index(&self) -> usize;
    #[inline]
    fn increment_by(&mut self, amount: usize) {
        *self = self.plus(amount);
    }

    #[inline]
    fn plus(self, amount: usize) -> Self {
        Self::new(self.index() + amount)
    }
}

impl Idx for usize {
    #[inline]
    fn new(s: usize) -> Self {
        s
    }
    #[inline]
    fn index(&self) -> usize {
        *self
    }
}

pub trait BitRelations<Rhs> {
    fn union(&mut self, other: &Rhs) -> bool;
    fn subtract(&mut self, other: &Rhs) -> bool;
    fn intersect(&mut self, other: &Rhs) -> bool;
}

macro_rules! bit_relations_inherent_impls {
    () => {
        /// Sets `self = self | other` and returns `true` if `self` changed
        /// (i.e., if new bits were added).
        pub fn union<Rhs>(&mut self, other: &Rhs) -> bool
        where
            Self: BitRelations<Rhs>,
        {
            <Self as BitRelations<Rhs>>::union(self, other)
        }

        /// Sets `self = self - other` and returns `true` if `self` changed.
        /// (i.e., if any bits were removed).
        pub fn subtract<Rhs>(&mut self, other: &Rhs) -> bool
        where
            Self: BitRelations<Rhs>,
        {
            <Self as BitRelations<Rhs>>::subtract(self, other)
        }

        /// Sets `self = self & other` and return `true` if `self` changed.
        /// (i.e., if any bits were removed).
        pub fn intersect<Rhs>(&mut self, other: &Rhs) -> bool
        where
            Self: BitRelations<Rhs>,
        {
            <Self as BitRelations<Rhs>>::intersect(self, other)
        }
    };
}
pub(crate) use bit_relations_inherent_impls;

#[inline]
fn inclusive_start_end<T: Idx>(
    range: impl RangeBounds<T>,
    domain: usize,
) -> Option<(usize, usize)> {
    // Both start and end are inclusive.
    let start = match range.start_bound().cloned() {
        Bound::Included(start) => start.index(),
        Bound::Excluded(start) => start.index() + 1,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound().cloned() {
        Bound::Included(end) => end.index(),
        Bound::Excluded(end) => end.index().checked_sub(1)?,
        Bound::Unbounded => domain - 1,
    };
    assert!(end < domain);
    if start > end {
        return None;
    }
    Some((start, end))
}

type Word = u64;
const WORD_BITS: usize = std::mem::size_of::<Word>() * 8;

#[inline]
fn num_words(domain_size: usize) -> usize {
    (domain_size.index() + WORD_BITS - 1) / WORD_BITS
}

#[inline]
fn num_bytes(domain_size: usize) -> usize {
    (domain_size.index() + 8 - 1) / 8
}

#[inline]
fn word_index_and_mask<T: Idx>(index: T) -> (usize, Word) {
    let index = index.index();
    let word_index = index / WORD_BITS;
    let mask = 1 << (index % WORD_BITS);
    (word_index, mask)
}

#[inline]
fn byte_index_and_mask(index: usize) -> (usize, u8) {
    let word_index = index / 8;
    let mask = 1 << (index % 8);
    (word_index, mask)
}

/*
#[inline]
fn max_bit(word: Word) -> usize {
    WORD_BITS - 1 - word.leading_zeros() as usize
}
*/

#[inline]
fn clear_excess_bits_in_final_word(domain_size: usize, words: &mut [Word]) {
    let num_bits_in_final_word = domain_size % WORD_BITS;
    if num_bits_in_final_word > 0 {
        let mask = (1 << num_bits_in_final_word) - 1;
        words[words.len() - 1] &= mask;
    }
}

#[inline]
fn bitwise<Op>(out_vec: &mut [Word], in_vec: impl IntoIterator<Item = Word>, op: Op) -> bool
where
    Op: Fn(Word, Word) -> Word,
{
    let mut changed = 0;
    for (out_elem, in_elem) in std::iter::zip(out_vec, in_vec) {
        let old_val = *out_elem;
        let new_val = op(old_val, in_elem);
        *out_elem = new_val;
        // This is essentially equivalent to a != with changed being a bool, but
        // in practice this code gets auto-vectorized by the compiler for most
        // operators. Using != here causes us to generate quite poor code as the
        // compiler tries to go back to a boolean on each loop iteration.
        changed |= old_val ^ new_val;
    }
    changed != 0
}

enum EitherIter<L, R> {
    Left(L),
    Right(R),
}

impl<L, R, T> Iterator for EitherIter<L, R>
where
    L: Iterator<Item = T>,
    R: Iterator<Item = T>,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            EitherIter::Left(l) => l.next(),
            EitherIter::Right(r) => r.next(),
        }
    }
}
