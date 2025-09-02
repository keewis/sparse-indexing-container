use ndarray::SliceArg;

trait OIndex {
    fn oindex(indexers: SliceArg) -> Self;
}
