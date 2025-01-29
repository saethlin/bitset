use crate::{
    bit_relations_inherent_impls,
    sparse::{SparseBitSet, SPARSE_MAX},
    BitRelations, BitSet, EitherIter, Idx,
};
use std::{
    fmt,
    ops::{Bound, RangeBounds},
};

/// A fixed-size bitset type with a hybrid representation: sparse when there
/// are up to a `SPARSE_MAX` elements in the set, but dense when there are more
/// than `SPARSE_MAX`.
///
/// This type is especially efficient for sets that typically have a small
/// number of elements, but a large `domain_size`, and are cleared frequently.
///
/// `T` is an index type, typically a newtyped `usize` wrapper, but it can also
/// just be `usize`.
///
/// All operations that involve an element will panic if the element is equal
/// to or greater than the domain size. All operations that involve two bitsets
/// will panic if the bitsets have differing domain sizes.
#[derive(Clone)]
pub enum HybridBitSet<T> {
    Sparse(SparseBitSet<T>),
    Dense(BitSet<T>),
}

impl<T: Idx> fmt::Debug for HybridBitSet<T> {
    fn fmt(&self, w: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sparse(b) => b.fmt(w),
            Self::Dense(b) => b.fmt(w),
        }
    }
}

impl<T: Idx> HybridBitSet<T> {
    pub fn new_empty(domain_size: usize) -> Self {
        HybridBitSet::Sparse(SparseBitSet::new_empty(domain_size))
    }

    pub fn domain_size(&self) -> usize {
        match self {
            HybridBitSet::Sparse(sparse) => sparse.domain_size(),
            HybridBitSet::Dense(dense) => dense.domain_size(),
        }
    }

    pub fn clear(&mut self) {
        let domain_size = self.domain_size();
        *self = HybridBitSet::new_empty(domain_size);
    }

    pub fn contains(&self, elem: T) -> bool {
        match self {
            HybridBitSet::Sparse(sparse) => sparse.contains(elem),
            HybridBitSet::Dense(dense) => dense.contains(elem),
        }
    }

    pub fn superset(&self, other: &HybridBitSet<T>) -> bool {
        match (self, other) {
            (HybridBitSet::Dense(self_dense), HybridBitSet::Dense(other_dense)) => {
                self_dense.superset(other_dense)
            }
            _ => {
                assert!(self.domain_size() == other.domain_size());
                other.iter().all(|elem| self.contains(elem))
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            HybridBitSet::Sparse(sparse) => sparse.is_empty(),
            HybridBitSet::Dense(dense) => dense.is_empty(),
        }
    }

    /// Returns the previous element present in the bitset from `elem`,
    /// inclusively of elem. That is, will return `Some(elem)` if elem is in the
    /// bitset.
    pub fn last_set_in(&self, range: impl RangeBounds<T>) -> Option<T>
    where
        T: Ord,
    {
        match self {
            HybridBitSet::Sparse(sparse) => sparse.last_set_in(range),
            HybridBitSet::Dense(dense) => dense.last_set_in(range),
        }
    }

    pub fn insert(&mut self, elem: T) -> bool {
        // No need to check `elem` against `self.domain_size` here because all
        // the match cases check it, one way or another.
        match self {
            HybridBitSet::Sparse(sparse) if sparse.len() < SPARSE_MAX => {
                // The set is sparse and has space for `elem`.
                sparse.insert(elem)
            }
            HybridBitSet::Sparse(sparse) if sparse.contains(elem) => {
                // The set is sparse and does not have space for `elem`, but
                // that doesn't matter because `elem` is already present.
                false
            }
            HybridBitSet::Sparse(sparse) => {
                // The set is sparse and full. Convert to a dense set.
                let mut dense = sparse.to_dense();
                let changed = dense.insert(elem);
                assert!(changed);
                *self = HybridBitSet::Dense(dense);
                changed
            }
            HybridBitSet::Dense(dense) => dense.insert(elem),
        }
    }

    pub fn insert_range(&mut self, elems: impl RangeBounds<T>) {
        // No need to check `elem` against `self.domain_size` here because all
        // the match cases check it, one way or another.
        let start = match elems.start_bound().cloned() {
            Bound::Included(start) => start.index(),
            Bound::Excluded(start) => start.index() + 1,
            Bound::Unbounded => 0,
        };
        let end = match elems.end_bound().cloned() {
            Bound::Included(end) => end.index() + 1,
            Bound::Excluded(end) => end.index(),
            Bound::Unbounded => self.domain_size() - 1,
        };
        let Some(len) = end.checked_sub(start) else { return };
        match self {
            HybridBitSet::Sparse(sparse) if sparse.len() + len < SPARSE_MAX => {
                // The set is sparse and has space for `elems`.
                for elem in start..end {
                    sparse.insert(T::new(elem));
                }
            }
            HybridBitSet::Sparse(sparse) => {
                // The set is sparse and full. Convert to a dense set.
                let mut dense = sparse.to_dense();
                dense.insert_range(elems);
                *self = HybridBitSet::Dense(dense);
            }
            HybridBitSet::Dense(dense) => dense.insert_range(elems),
        }
    }

    pub fn insert_all(&mut self) {
        let domain_size = self.domain_size();
        match self {
            HybridBitSet::Sparse(_) => {
                *self = HybridBitSet::Dense(BitSet::new_filled(domain_size));
            }
            HybridBitSet::Dense(dense) => dense.insert_all(),
        }
    }

    pub fn remove(&mut self, elem: T) -> bool {
        // Note: we currently don't bother going from Dense back to Sparse.
        match self {
            HybridBitSet::Sparse(sparse) => sparse.remove(elem),
            HybridBitSet::Dense(dense) => dense.remove(elem),
        }
    }

    /// Converts to a dense set, consuming itself in the process.
    pub fn to_dense(self) -> BitSet<T> {
        match self {
            HybridBitSet::Sparse(sparse) => sparse.to_dense(),
            HybridBitSet::Dense(dense) => dense,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        match self {
            HybridBitSet::Sparse(sparse) => EitherIter::Left(sparse.iter()),
            HybridBitSet::Dense(dense) => EitherIter::Right(dense.iter()),
        }
    }

    bit_relations_inherent_impls! {}
}
