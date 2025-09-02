use crate::container::SparseContainer;
use crate::errors::IndexError;
use ndarray::{Array1, IxDyn, SliceArg};

#[derive(Debug)]
struct COO<T> {
    data: Array1<T>,
    coords: Vec<Array1<u64>>,

    fill_value: T,
    shape: Vec<u64>,
}

impl<T> COO<T> {
    fn new(shape: Vec<u64>, data: Array1<T>, coords: Vec<Array1<u64>>, fill_value: T) -> Self {
        // unchecked consistency for increased efficiency

        COO {
            data,
            coords,
            fill_value,
            shape,
        }
    }
}

impl<T> SparseContainer<T> for COO<T> {
    fn shape(&self) -> &Vec<u64> {
        &self.shape
    }
    fn fill_value(&self) -> &T {
        &self.fill_value
    }
    fn data(&self) -> &Array1<T> {
        &self.data
    }

    fn coords(&self) -> &Vec<Array1<u64>> {
        &self.coords
    }

    // fn oindex<I>(&self, indexers: I) -> Result<Self, IndexError>
    // where
    //     I: SliceArg<IxDyn>,
    // {
    // }
    // fn concat(parts: Vec<Container<T>>, axis: u8) -> Result<Container<T>, ConcatError> {}
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_init() {
        let shape: Vec<u64> = vec![10, 10];
        let data: Array1<f64> = array![1.3, 4.7, 2.6];
        let coords: Vec<Array1<u64>> = vec![array![2, 7, 4], array![4, 0, 9]];
        let fill_value: f64 = 0.0;

        let obj = COO::new(
            shape.clone(),
            data.clone(),
            coords.clone(),
            fill_value.clone(),
        );

        assert_eq!(obj.shape, shape);
        assert_eq!(obj.fill_value, fill_value);

        assert_eq!(obj.data, data);
        assert_eq!(obj.coords, coords);
    }

    #[test]
    fn test_ndim() {
        let shape: Vec<u64> = vec![10, 10];
        let data: Array1<f64> = array![1.3, 4.7, 2.6];
        let coords: Vec<Array1<u64>> = vec![array![2, 7, 4], array![4, 0, 9]];
        let fill_value: f64 = 0.0;

        let obj = COO::new(
            shape.clone(),
            data.clone(),
            coords.clone(),
            fill_value.clone(),
        );

        assert_eq!(obj.ndim(), 2);
    }
}
