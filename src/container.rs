use ndarray::Array1;
use std::ops::Range;

pub trait SparseContainer<T> {
    fn shape(&self) -> &[usize];

    fn decompose(self) -> (Vec<usize>, Array1<T>, Vec<Array1<usize>>);

    // orthogonal indexing
    fn oindex(&self, slices: &[Range<usize>]) -> Self
    where
        Self: Sized;

    // concatenation
    // fn concat(parts: Vec<Container<T>>, axis: u8) -> Result<Container<T>, ConcatError>;

    // number of dimensions
    fn ndim(&self) -> usize {
        self.shape().len()
    }
}
