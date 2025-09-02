use ndarray::{SliceArg, Array1};


trait SparseContainer<T> {
    fn data(&self) -> Array1<T>;
    fn coords(&self) -> Vec<Array1<uint64>>;

    fn oindex(&self, indexers: SliceArg) -> Self;

    fn concat(parts: Vec<Container<T>>) -> Container<T>;
}
