extern crate bitset;
extern crate criterion;

use criterion::{criterion_group, criterion_main, Criterion};

type BitSet = bitset::BitSet<usize>;

#[inline]
fn iter_ones_using_contains<F: FnMut(usize)>(set: &BitSet, f: &mut F) {
    for bit in 0..set.domain_size() {
        if set.contains(bit) {
            f(bit);
        }
    }
}

fn iter_ones_using_contains_all_zeros(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let set = BitSet::new_empty(N);

    c.bench_function("iter_ones/contains_all_zeros", |b| {
        b.iter(|| {
            let mut count = 0;
            iter_ones_using_contains(&set, &mut |_bit| count += 1);
            count
        })
    });
}

fn iter_ones_using_contains_all_ones(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let set = BitSet::new_filled(N);
    //set.insert_range(..);

    c.bench_function("iter_ones/contains_all_ones", |b| {
        b.iter(|| {
            let mut count = 0;
            iter_ones_using_contains(&set, &mut |_bit| count += 1);
            count
        })
    });
}

fn iter_ones_all_zeros(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let set = BitSet::new_empty(N);

    c.bench_function("iter_ones/all_zeros", |b| {
        b.iter(|| {
            let mut count = 0;
            for _ in set.iter() {
                count += 1;
            }
            count
        })
    });
}

fn iter_ones_all_ones(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let set = BitSet::new_filled(N);

    c.bench_function("iter_ones/all_ones", |b| {
        b.iter(|| {
            let mut count = 0;
            for _ in set.iter() {
                count += 1;
            }
            count
        })
    });
}

/*
fn insert_range(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut set = BitSet::new_empty(N);

    c.bench_function("insert_range/1m", |b| b.iter(|| set.insert_range(..)));
}
*/

fn insert(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut set = BitSet::new_empty(N);

    c.bench_function("insert/1m", |b| {
        b.iter(|| {
            for i in 0..N {
                set.insert(i);
            }
        })
    });
}

fn insert_small(c: &mut Criterion) {
    const N: usize = 248;
    let mut set = BitSet::new_empty(N);

    c.bench_function("insert/248", |b| {
        b.iter(|| {
            for i in 0..N {
                set.insert(i);
            }
        })
    });
}

/*
fn grow_and_insert(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut set = BitSet::new_empty(N);

    c.bench_function("grow_and_insert", |b| {
        b.iter(|| {
            for i in 0..N {
                set.grow_and_insert(i);
            }
        })
    });
}

fn union_with(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut set_a = BitSet::new_empty(N);
    let set_b = BitSet::new_empty(N);

    c.bench_function("union_with/1m", |b| b.iter(|| set_a.union_with(&set_b)));
}

fn intersect_with(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut set_a = BitSet::new_empty(N);
    let set_b = BitSet::new_empty(N);

    c.bench_function("intersect_with/1m", |b| {
        b.iter(|| set_a.intersect_with(&set_b))
    });
}

fn difference_with(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut set_a = BitSet::new_empty(N);
    let set_b = BitSet::new_empty(N);

    c.bench_function("difference_with/1m", |b| {
        b.iter(|| set_a.difference_with(&set_b))
    });
}

fn symmetric_difference_with(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut set_a = BitSet::new_empty(N);
    let set_b = BitSet::new_empty(N);

    c.bench_function("symmetric_difference_with/1m", |b| {
        b.iter(|| set_a.symmetric_difference_with(&set_b))
    });
}
*/

fn clear(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut set_a = BitSet::new_empty(N);

    c.bench_function("clear/1m", |b| b.iter(|| set_a.clear()));
}

/*
fn count_ones(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let set_a = BitSet::new_empty(N);

    c.bench_function("count_ones/1m", |b| {
        b.iter(|| black_box(set_a.count_ones(..)))
    });
}
*/

criterion_group!(
    benches,
    iter_ones_using_contains_all_zeros,
    iter_ones_using_contains_all_ones,
    iter_ones_all_zeros,
    iter_ones_all_ones,
    //insert_range,
    insert,
    insert_small,
    //intersect_with,
    //difference_with,
    //union_with,
    //symmetric_difference_with,
    //count_ones,
    clear,
    //grow_and_insert,
);
criterion_main!(benches);
