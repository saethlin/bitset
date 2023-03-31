#![warn(clippy::pedantic)]
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Bound;
use std::ops::RangeBounds;

mod dense;
mod fixed;
mod iter;
mod nonmaxu8;

pub use dense::*;
pub use fixed::*;
use iter::*;

pub trait Idx: Copy + 'static + Eq + PartialEq + Debug + Hash {
    fn new(_: usize) -> Self;
    fn index(&self) -> usize;
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
fn word_index_and_mask(index: usize) -> (usize, Word) {
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

#[inline]
fn max_bit(word: Word) -> usize {
    WORD_BITS - 1 - word.leading_zeros() as usize
}
