use ndarray::Array1;

pub trait SparseContainer<T> {
    fn fill_value(&self) -> &T;
    fn shape(&self) -> &Vec<usize>;

    fn decompose(self) -> (Vec<usize>, T, Array1<T>, Vec<Array1<usize>>);

    // orthogonal indexing
    // fn oindex<I>(&self, indexers: I) -> Result<Self, IndexError>
    // where
    //     I: SliceArg<IxDyn>,
    //     Self: Sized;

    // concatenation
    // fn concat(parts: Vec<Container<T>>, axis: u8) -> Result<Container<T>, ConcatError>;

    // number of dimensions
    fn ndim(&self) -> usize {
        self.shape().len()
    }
}
