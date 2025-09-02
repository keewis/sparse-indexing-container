use crate::errors::IndexError;
use ndarray::{Array1, IxDyn, SliceArg};

trait SparseContainer<T> {
    fn fill_value(&self) -> T;
    fn shape(&self) -> Vec<u64>;

    fn data(&self) -> &Array1<T>;
    fn coords(&self) -> Vec<Array1<u64>>;

    // orthogonal indexing
    fn oindex<I>(&self, indexers: I) -> Result<Self, IndexError>
    where
        I: SliceArg<IxDyn>,
        Self: Sized;

    // fn concat(parts: Vec<Container<T>>, axis: u8) -> Result<Container<T>, ConcatError>;

    // number of dimensions
    fn ndim(&self) -> usize {
        self.shape().len()
    }
}
