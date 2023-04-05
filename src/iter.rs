use crate::{Idx, Word, WORD_BITS};
use std::marker::PhantomData;

pub struct BitIter<T: Idx, I> {
    /// A copy of the current word, but with any already-visited bits cleared.
    /// (This lets us use `trailing_zeros()` to find the next set bit.) When it
    /// is reduced to 0, we move onto the next word.
    word: Word,

    /// The offset (measured in bits) of the current word.
    offset: usize,

    /// Underlying iterator over the words.
    iter: I,

    marker: PhantomData<T>,
}

impl<T: Idx, I> BitIter<T, I> {
    #[inline]
    pub(crate) fn new(iter: I) -> BitIter<T, I> {
        // We initialize `word` and `offset` to degenerate values. On the first
        // call to `next()` we will fall through to getting the first word from
        // `iter`, which sets `word` to the first word (if there is one) and
        // `offset` to 0. Doing it this way saves us from having to maintain
        // additional state about whether we have started.
        BitIter {
            word: 0,
            offset: usize::MAX - (WORD_BITS - 1),
            iter,
            marker: PhantomData,
        }
    }
}

impl<T: Idx, I: Iterator<Item = Word>> Iterator for BitIter<T, I> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        loop {
            if self.word != 0 {
                // Get the position of the next set bit in the current word,
                // then clear the bit.
                let bit_pos = self.word.trailing_zeros() as usize;
                let bit = 1 << bit_pos;
                self.word ^= bit;
                return Some(T::new(bit_pos + self.offset));
            }

            // Move onto the next word. `wrapping_add()` is needed to handle
            // the degenerate initial value given to `offset` in `new()`.
            self.word = self.iter.next()?;
            self.offset = self.offset.wrapping_add(WORD_BITS);
        }
    }
}
