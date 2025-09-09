use std::ops::Range;

pub fn slice_size(slice: &Range<usize>, size: &usize) -> usize {
    let size_ = *size;

    if slice.start >= size_ {
        0
    } else {
        slice.end.min(size_).saturating_sub(slice.start)
    }
}
