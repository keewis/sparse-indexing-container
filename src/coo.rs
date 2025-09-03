use crate::container::SparseContainer;
use crate::errors::IndexError;
use ndarray::{Array1, Array2, IxDyn, SliceArg, SliceInfoElem, Zip};

#[derive(Debug)]
struct COO<T> {
    data: Array1<T>,
    coords: Array2<usize>,

    fill_value: T,
    shape: Vec<usize>,
}

impl<T> COO<T> {
    fn new(shape: Vec<usize>, data: Array1<T>, coords: Array2<usize>, fill_value: T) -> Self {
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
    fn shape(&self) -> &Vec<usize> {
        &self.shape
    }
    fn fill_value(&self) -> &T {
        &self.fill_value
    }
    fn data(&self) -> &Array1<T> {
        &self.data
    }

    fn coords(&self) -> &Array2<usize> {
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

    fn create_coo_2d_f64() -> COO<f64> {
        let shape: Vec<usize> = vec![10, 10];
        let data: Array1<f64> = array![1.3, 4.7, 2.6];
        let coords: Array2<usize> = array![[2, 4], [7, 0], [4, 9]];
        let fill_value: f64 = 0.0;

        COO::new(shape, data, coords, fill_value)
    }

    fn create_coo_3d_f64() -> COO<f64> {
        let shape: Vec<usize> = vec![10, 10, 15];
        let data: Array1<f64> = array![1.3, 4.7, 2.6, 1.2];
        let coords: Array2<usize> = array![[2, 4, 2], [7, 0, 14], [4, 9, 5], [9, 2, 8]];
        let fill_value: f64 = 0.0;

        COO::new(shape, data, coords, fill_value)
    }

    #[test]
    fn test_init() {
        let shape: Vec<usize> = vec![10, 10];
        let data: Array1<f64> = array![1.3, 4.7, 2.6];
        let coords: Array2<usize> = array![[2, 4], [7, 0], [4, 9]];
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
    fn test_ndim_2d() {
        let obj = create_coo_2d_f64();

        assert_eq!(obj.ndim(), 2);
    }

    #[test]
    fn test_ndim_3d() {
        let obj = create_coo_3d_f64();

        assert_eq!(obj.ndim(), 3);
    }
}
