use crate::{
    num_words, word_index_and_mask, BitIter, BitRelations, BitSet, HybridBitSet, Idx, Word,
    WORD_BITS,
};
use std::{fmt, marker::PhantomData};

/// A fixed-column-size, variable-row-size 2D bit matrix with a moderately
/// sparse representation.
///
/// Initially, every row has no explicit representation. If any bit within a
/// row is set, the entire row is instantiated as `Some(<HybridBitSet>)`.
/// Furthermore, any previously uninstantiated rows prior to it will be
/// instantiated as `None`. Those prior rows may themselves become fully
/// instantiated later on if any of their bits are set.
///
/// `R` and `C` are index types used to identify rows and columns respectively;
/// typically newtyped `usize` wrappers, but they can also just be `usize`.
#[derive(Clone, Debug)]
pub struct SparseBitMatrix<R, C>
where
    R: Idx,
    C: Idx,
{
    num_columns: usize,
    rows: IndexVec<R, Option<HybridBitSet<C>>>,
}

impl<R: Idx, C: Idx> SparseBitMatrix<R, C> {
    /// Creates a new empty sparse bit matrix with no rows or columns.
    pub fn new(num_columns: usize) -> Self {
        Self {
            num_columns,
            rows: IndexVec::new(),
        }
    }

    fn ensure_row(&mut self, row: R) -> &mut HybridBitSet<C> {
        // Instantiate any missing rows up to and including row `row` with an empty
        // HybridBitSet. Then replace row `row` with a full HybridBitSet if
        // necessary.
        self.rows
            .get_or_insert_with(row, || HybridBitSet::new_empty(self.num_columns))
    }

    /// Sets the cell at `(row, column)` to true. Put another way, insert
    /// `column` to the bitset for `row`.
    ///
    /// Returns `true` if this changed the matrix.
    pub fn insert(&mut self, row: R, column: C) -> bool {
        self.ensure_row(row).insert(column)
    }

    /// Sets the cell at `(row, column)` to false. Put another way, delete
    /// `column` from the bitset for `row`. Has no effect if `row` does not
    /// exist.
    ///
    /// Returns `true` if this changed the matrix.
    pub fn remove(&mut self, row: R, column: C) -> bool {
        match self.rows.get_mut(row) {
            Some(Some(row)) => row.remove(column),
            _ => false,
        }
    }

    /// Sets all columns at `row` to false. Has no effect if `row` does
    /// not exist.
    pub fn clear(&mut self, row: R) {
        if let Some(Some(row)) = self.rows.get_mut(row) {
            row.clear();
        }
    }

    /// Do the bits from `row` contain `column`? Put another way, is
    /// the matrix cell at `(row, column)` true?  Put yet another way,
    /// if the matrix represents (transitive) reachability, can
    /// `row` reach `column`?
    pub fn contains(&self, row: R, column: C) -> bool {
        self.row(row).map_or(false, |r| r.contains(column))
    }

    /// Adds the bits from row `read` to the bits from row `write`, and
    /// returns `true` if anything changed.
    ///
    /// This is used when computing transitive reachability because if
    /// you have an edge `write -> read`, because in that case
    /// `write` can reach everything that `read` can (and
    /// potentially more).
    pub fn union_rows(&mut self, read: R, write: R) -> bool {
        if read == write || self.row(read).is_none() {
            return false;
        }

        self.ensure_row(write);
        if let (Some(read_row), Some(write_row)) = self.rows.pick2_mut(read, write) {
            write_row.union(read_row)
        } else {
            unreachable!()
        }
    }

    /// Insert all bits in the given row.
    pub fn insert_all_into_row(&mut self, row: R) {
        self.ensure_row(row).insert_all();
    }

    pub fn rows(&self) -> impl Iterator<Item = R> {
        self.rows.indices()
    }

    /// Iterates through all the columns set to true in a given row of
    /// the matrix.
    pub fn iter<'a>(&'a self, row: R) -> impl Iterator<Item = C> + 'a {
        self.row(row).into_iter().flat_map(|r| r.iter())
    }

    pub fn row(&self, row: R) -> Option<&HybridBitSet<C>> {
        self.rows.get(row)?.as_ref()
    }

    /// Intersects `row` with `set`. `set` can be either `BitSet` or
    /// `HybridBitSet`. Has no effect if `row` does not exist.
    ///
    /// Returns true if the row was changed.
    pub fn intersect_row<Set>(&mut self, row: R, set: &Set) -> bool
    where
        HybridBitSet<C>: BitRelations<Set>,
    {
        match self.rows.get_mut(row) {
            Some(Some(row)) => row.intersect(set),
            _ => false,
        }
    }

    /// Subtracts `set` from `row`. `set` can be either `BitSet` or
    /// `HybridBitSet`. Has no effect if `row` does not exist.
    ///
    /// Returns true if the row was changed.
    pub fn subtract_row<Set>(&mut self, row: R, set: &Set) -> bool
    where
        HybridBitSet<C>: BitRelations<Set>,
    {
        match self.rows.get_mut(row) {
            Some(Some(row)) => row.subtract(set),
            _ => false,
        }
    }

    /// Unions `row` with `set`. `set` can be either `BitSet` or
    /// `HybridBitSet`.
    ///
    /// Returns true if the row was changed.
    pub fn union_row<Set>(&mut self, row: R, set: &Set) -> bool
    where
        HybridBitSet<C>: BitRelations<Set>,
    {
        self.ensure_row(row).union(set)
    }
}
