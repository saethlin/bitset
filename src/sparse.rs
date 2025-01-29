use crate::{bit_relations_inherent_impls, BitRelations, BitSet, Idx};
use arrayvec::ArrayVec;
use std::ops::RangeBounds;

pub const SPARSE_MAX: usize = 8;

/// A fixed-size bitset type with a sparse representation and a maximum of
/// `SPARSE_MAX` elements. The elements are stored as a sorted `ArrayVec` with
/// no duplicates.
///
/// This type is used by `HybridBitSet`; do not use directly.
#[derive(Clone, Debug)]
pub struct SparseBitSet<T> {
    domain_size: usize,
    elems: ArrayVec<T, SPARSE_MAX>,
}

impl<T: Idx> SparseBitSet<T> {
    pub fn new_empty(domain_size: usize) -> Self {
        SparseBitSet {
            domain_size,
            elems: ArrayVec::new(),
        }
    }

    pub fn domain_size(&self) -> usize {
        self.domain_size
    }

    pub fn len(&self) -> usize {
        self.elems.len()
    }

    pub fn is_empty(&self) -> bool {
        self.elems.len() == 0
    }

    pub fn contains(&self, elem: T) -> bool {
        assert!(elem.index() < self.domain_size);
        self.elems.contains(&elem)
    }

    pub fn insert(&mut self, elem: T) -> bool {
        assert!(elem.index() < self.domain_size);
        let changed = if let Some(i) = self.elems.iter().position(|&e| e.index() >= elem.index()) {
            if self.elems[i] == elem {
                // `elem` is already in the set.
                false
            } else {
                // `elem` is smaller than one or more existing elements.
                self.elems.insert(i, elem);
                true
            }
        } else {
            // `elem` is larger than all existing elements.
            self.elems.push(elem);
            true
        };
        assert!(self.len() <= SPARSE_MAX);
        changed
    }

    pub fn remove(&mut self, elem: T) -> bool {
        assert!(elem.index() < self.domain_size);
        if let Some(i) = self.elems.iter().position(|&e| e == elem) {
            self.elems.remove(i);
            true
        } else {
            false
        }
    }

    pub fn to_dense(&self) -> BitSet<T> {
        let mut dense = BitSet::new_empty(self.domain_size);
        for elem in self.elems.iter() {
            dense.insert(*elem);
        }
        dense
    }

    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        self.elems.iter().copied()
    }

    bit_relations_inherent_impls! {}
}

impl<T: Idx + Ord> SparseBitSet<T> {
    pub fn last_set_in(&self, range: impl RangeBounds<T>) -> Option<T> {
        let mut last_leq = None;
        for e in self.iter() {
            if range.contains(&e) {
                last_leq = Some(e);
            }
        }
        last_leq
    }
}
