use crate::container::SparseContainer;
use ndarray::{Array1, Array2, IxDyn, SliceArg, SliceInfoElem};
use rayon::{iter::ParallelIterator, prelude::ParallelBridge};
use std::iter::Iterator;
use std::ops::Range;

#[derive(Debug, PartialEq, Clone)]
struct COO<T>
where
    T: Copy + Send + Sync,
{
    data: Array1<T>,
    coords: Vec<Array1<usize>>,

    fill_value: T,
    shape: Vec<usize>,
}

impl<T: Copy + Send + Sync> COO<T> {
    fn new(shape: Vec<usize>, data: Array1<T>, coords: Vec<Array1<usize>>, fill_value: T) -> Self {
        // unchecked consistency for increased efficiency

        COO {
            data,
            coords,
            fill_value,
            shape,
        }
    }
}

fn slice_size(slice: &Range<usize>, size: &usize) -> usize {
    let size_ = *size;

    if slice.start >= size_ {
        0
    } else {
        slice.end.min(size_).saturating_sub(slice.start)
    }
}

impl<T: Copy + Send + Sync + std::fmt::Debug> SparseContainer<T> for COO<T> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn decompose(self) -> (Vec<usize>, T, Array1<T>, Vec<Array1<usize>>) {
        (self.shape, self.fill_value, self.data, self.coords)
    }

    fn oindex<I>(&self, indexers: I) -> Self
    where
        I: SliceArg<IxDyn> + std::fmt::Debug,
        Self: Sized,
    {
        // idea:
        // - convert indexers to a vec of ranges (panic if not a concrete slice)
        // - zip data and coords
        // - filtermap:
        //   - for each indexer:
        //     - if range:
        //       - if coords not in range: None
        //       - else: remove range.start
        //     - else:
        //       - raise (not implemented, not necessary for zarr)
        // - figure out the new data shape (using range intersections)
        let slices = indexers
            .as_ref()
            .iter()
            .zip(self.shape.iter())
            .map(|(indexer, size)| match indexer {
                SliceInfoElem::Slice { start, end, step } => {
                    let start_ = if *start < 0 {
                        panic!("negative start values are not supported")
                    } else {
                        *start as usize
                    };

                    if *step != 1 {
                        panic!("only step sizes of 1 are supported, got {:?}", step);
                    }

                    let end_ = end.map_or(*size, |v| v as usize);

                    Range {
                        start: start_,
                        end: end_,
                    }
                }
                _ => panic!("unsupported indexer type: {:?}", indexer),
            })
            .collect::<Vec<_>>();

        let new_shape = slices
            .iter()
            .zip(self.shape.iter())
            .map(|(slice, size)| slice_size(slice, size))
            .collect::<Vec<_>>();

        let (filtered_data, filtered_coords): (Vec<T>, Vec<Vec<usize>>) = self
            .iter()
            .par_bridge()
            .filter_map(|(data, coords)| {
                let matched = slices
                    .iter()
                    .zip(coords.iter())
                    .all(|(s, &c)| s.contains(c));

                if matched {
                    Some((
                        data,
                        slices
                            .iter()
                            .zip(coords.iter())
                            .map(|(s, &c)| c - s.start)
                            .collect::<Vec<_>>(),
                    ))
                } else {
                    None
                }
            })
            .unzip();

        let sliced_data: Array1<T> = filtered_data.into();
        let sliced_coords: Array2<usize> =
            Array1::from(filtered_coords.into_iter().flatten().collect::<Vec<_>>())
                .into_shape_with_order((sliced_data.len(), self.ndim()))
                .unwrap();

        Self {
            shape: new_shape,
            fill_value: self.fill_value.clone(),
            data: sliced_data,
            coords: sliced_coords
                .columns()
                .into_iter()
                .map(|c| c.into_owned())
                .collect::<Vec<_>>(),
        }
    }

    // fn concat(parts: Vec<Container<T>>, axis: u8) -> Result<Container<T>, ConcatError> {}
}

struct CooIterator<'a, T>
where
    T: Copy + Send + Sync,
{
    index: usize,

    data: &'a Array1<T>,
    coords: &'a [Array1<usize>],
}

impl<'a, T: Copy + Send + Sync> Iterator for CooIterator<'a, T> {
    type Item = (T, Vec<&'a usize>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.data.len() {
            None
        } else {
            let result = Some((
                self.data[self.index],
                self.coords
                    .iter()
                    .map(|c| &c[self.index])
                    .collect::<Vec<_>>(),
            ));
            self.index += 1;

            result
        }
    }
}

impl<T: Copy + Send + Sync> COO<T> {
    fn iter(&self) -> CooIterator<'_, T> {
        CooIterator::<T> {
            index: 0,
            data: &self.data,
            coords: &self.coords,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::{array, s};

    fn create_coo_2d_f64() -> COO<f64> {
        let shape: Vec<usize> = vec![10, 10];
        let data: Array1<f64> = array![1.3, 4.7, 2.6];
        let coords: Array2<usize> = array![[2, 4], [7, 0], [4, 9]];
        let fill_value: f64 = 0.0;

        COO::new(
            shape,
            data,
            coords
                .columns()
                .into_iter()
                .map(|c| c.into_owned())
                .collect::<Vec<_>>(),
            fill_value,
        )
    }

    fn create_coo_3d_f64() -> COO<f64> {
        let shape: Vec<usize> = vec![10, 10, 15];
        let data: Array1<f64> = array![1.3, 4.7, 2.6, 1.2];
        let coords: Array2<usize> = array![[2, 4, 2], [7, 0, 14], [4, 9, 5], [9, 2, 8]];
        let fill_value: f64 = 0.0;

        COO::new(
            shape,
            data,
            coords
                .columns()
                .into_iter()
                .map(|c| c.into_owned())
                .collect::<Vec<_>>(),
            fill_value,
        )
    }

    #[test]
    fn test_init() {
        let shape: Vec<usize> = vec![10, 10];
        let data: Array1<f64> = array![1.3, 4.7, 2.6];
        let coords: Vec<Array1<usize>> = array![[2, 4], [7, 0], [4, 9]]
            .columns()
            .into_iter()
            .map(|c| c.into_owned())
            .collect::<Vec<_>>();
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

        let data2: Array1<bool> = array![false, true, false];
        let fill_value2 = false;
        let obj2 = COO::new(
            shape.clone(),
            data2.clone(),
            coords.clone(),
            fill_value2.clone(),
        );

        assert_eq!(obj2.shape, shape);
        assert_eq!(obj2.fill_value, fill_value2);

        assert_eq!(obj2.data, data2);
        assert_eq!(obj2.coords, coords);
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

    #[test]
    fn test_oindex_2d_some() {
        let obj = create_coo_2d_f64();

        let actual = obj.oindex(s![..5, ..5]);
        let expected = COO::<f64>::new(vec![5, 5], array![1.3], vec![array![2], array![4]], 0.0);

        // assert!(actual_.is_ok(), "{}", actual_.unwrap_err().to_string());
        // let actual = actual_.unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_oindex_2d_none() {
        let obj = create_coo_2d_f64();

        let actual = obj.oindex(s![10..15, 10..15]);
        let expected = COO::<f64>::new(vec![0, 0], array![], vec![array![], array![]], 0.0);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_oindex_2d_empty() {
        let obj = create_coo_2d_f64();

        let actual = obj.oindex(s![5..5, 5..5]);
        let expected = COO::<f64>::new(vec![0, 0], array![], vec![array![], array![]], 0.0);

        assert_eq!(actual, expected);
    }
}
