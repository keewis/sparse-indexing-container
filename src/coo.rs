use crate::container::SparseContainer;
use crate::slices::slice_size;

use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, IxDyn, SliceArg, SliceInfoElem, Zip};

use numpy::{
    dtype, PyArray1, PyArrayDescrMethods, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods,
};

use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyComplex, PyFloat, PyInt, PySlice};

use rayon::iter::ParallelIterator;

use std::iter::Iterator;
use std::ops::Range;

#[derive(Debug, PartialEq, Clone)]
struct Coo<T>
where
    T: Copy + Send + Sync,
{
    data: Array1<T>,
    coords: Vec<Array1<usize>>,

    shape: Vec<usize>,
}

impl<T: Copy + Send + Sync> Coo<T> {
    fn new(shape: Vec<usize>, data: Array1<T>, coords: Vec<Array1<usize>>) -> Self {
        // unchecked consistency for increased efficiency

        Coo {
            data,
            coords,
            shape,
        }
    }
}

impl<T: Copy + Send + Sync + std::fmt::Debug> SparseContainer<T> for Coo<T> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn decompose(self) -> (Vec<usize>, Array1<T>, Vec<Array1<usize>>) {
        (self.shape, self.data, self.coords)
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

        let coords2d: Array2<usize> =
            Array1::from(self.coords.iter().flatten().copied().collect::<Vec<_>>())
                .into_shape_with_order((self.ndim(), self.data.len()))
                .unwrap();
        let (filtered_data, filtered_coords): (Vec<T>, Vec<Vec<usize>>) = Zip::from(&self.data)
            .and(coords2d.columns())
            .into_par_iter()
            .filter_map(|(data, coords)| {
                let matched = slices.iter().zip(coords.iter()).all(|(s, c)| s.contains(c));

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

enum Container {
    Bool(Coo<bool>),
    Int8(Coo<i8>),
    Int16(Coo<i16>),
    Int32(Coo<i32>),
    Int64(Coo<i64>),
    UInt8(Coo<u8>),
    UInt16(Coo<u16>),
    UInt32(Coo<u32>),
    UInt64(Coo<u64>),
    Float32(Coo<f32>),
    Float64(Coo<f64>),
    Complex32(Coo<num_complex::Complex32>),
    Complex64(Coo<num_complex::Complex64>),
}

impl Container {
    pub fn decompose<'py>(
        self,
        py: Python<'py>,
    ) -> (
        Vec<usize>,
        Bound<'py, PyUntypedArray>,
        Vec<Bound<'py, PyArray1<usize>>>,
    ) {
        match self {
            Self::Bool(c) => {
                let (shape, data, coords) = c.decompose();
                let bound_data = PyArray1::from_owned_array(py, data);
                let bound_coords = coords
                    .into_iter()
                    .map(|arr| PyArray1::from_owned_array(py, arr))
                    .collect();
                (
                    shape,
                    unsafe { bound_data.cast_into_unchecked() },
                    bound_coords,
                )
            }
            Self::Int8(c) => {
                let (shape, data, coords) = c.decompose();
                let bound_data = PyArray1::from_owned_array(py, data);
                let bound_coords = coords
                    .into_iter()
                    .map(|arr| PyArray1::from_owned_array(py, arr))
                    .collect();
                (
                    shape,
                    unsafe { bound_data.cast_into_unchecked() },
                    bound_coords,
                )
            }
            Self::Int16(c) => {
                let (shape, data, coords) = c.decompose();
                let bound_data = PyArray1::from_owned_array(py, data);
                let bound_coords = coords
                    .into_iter()
                    .map(|arr| PyArray1::from_owned_array(py, arr))
                    .collect();
                (
                    shape,
                    unsafe { bound_data.cast_into_unchecked() },
                    bound_coords,
                )
            }
            Self::Int32(c) => {
                let (shape, data, coords) = c.decompose();
                let bound_data = PyArray1::from_owned_array(py, data);
                let bound_coords = coords
                    .into_iter()
                    .map(|arr| PyArray1::from_owned_array(py, arr))
                    .collect();
                (
                    shape,
                    unsafe { bound_data.cast_into_unchecked() },
                    bound_coords,
                )
            }
            Self::Int64(c) => {
                let (shape, data, coords) = c.decompose();
                let bound_data = PyArray1::from_owned_array(py, data);
                let bound_coords = coords
                    .into_iter()
                    .map(|arr| PyArray1::from_owned_array(py, arr))
                    .collect();
                (
                    shape,
                    unsafe { bound_data.cast_into_unchecked() },
                    bound_coords,
                )
            }
            Self::UInt8(c) => {
                let (shape, data, coords) = c.decompose();
                let bound_data = PyArray1::from_owned_array(py, data);
                let bound_coords = coords
                    .into_iter()
                    .map(|arr| PyArray1::from_owned_array(py, arr))
                    .collect();
                (
                    shape,
                    unsafe { bound_data.cast_into_unchecked() },
                    bound_coords,
                )
            }
            Self::UInt16(c) => {
                let (shape, data, coords) = c.decompose();
                let bound_data = PyArray1::from_owned_array(py, data);
                let bound_coords = coords
                    .into_iter()
                    .map(|arr| PyArray1::from_owned_array(py, arr))
                    .collect();
                (
                    shape,
                    unsafe { bound_data.cast_into_unchecked() },
                    bound_coords,
                )
            }
            Self::UInt32(c) => {
                let (shape, data, coords) = c.decompose();
                let bound_data = PyArray1::from_owned_array(py, data);
                let bound_coords = coords
                    .into_iter()
                    .map(|arr| PyArray1::from_owned_array(py, arr))
                    .collect();
                (
                    shape,
                    unsafe { bound_data.cast_into_unchecked() },
                    bound_coords,
                )
            }
            Self::UInt64(c) => {
                let (shape, data, coords) = c.decompose();
                let bound_data = PyArray1::from_owned_array(py, data);
                let bound_coords = coords
                    .into_iter()
                    .map(|arr| PyArray1::from_owned_array(py, arr))
                    .collect();
                (
                    shape,
                    unsafe { bound_data.cast_into_unchecked() },
                    bound_coords,
                )
            }
            Self::Float32(c) => {
                let (shape, data, coords) = c.decompose();
                let bound_data = PyArray1::from_owned_array(py, data);
                let bound_coords = coords
                    .into_iter()
                    .map(|arr| PyArray1::from_owned_array(py, arr))
                    .collect();
                (
                    shape,
                    unsafe { bound_data.cast_into_unchecked() },
                    bound_coords,
                )
            }
            Self::Float64(c) => {
                let (shape, data, coords) = c.decompose();
                let bound_data = PyArray1::from_owned_array(py, data);
                let bound_coords = coords
                    .into_iter()
                    .map(|arr| PyArray1::from_owned_array(py, arr))
                    .collect();
                (
                    shape,
                    unsafe { bound_data.cast_into_unchecked() },
                    bound_coords,
                )
            }
            Self::Complex32(c) => {
                let (shape, data, coords) = c.decompose();
                let bound_data = PyArray1::from_owned_array(py, data);
                let bound_coords = coords
                    .into_iter()
                    .map(|arr| PyArray1::from_owned_array(py, arr))
                    .collect();
                (
                    shape,
                    unsafe { bound_data.cast_into_unchecked() },
                    bound_coords,
                )
            }
            Self::Complex64(c) => {
                let (shape, data, coords) = c.decompose();
                let bound_data = PyArray1::from_owned_array(py, data);
                let bound_coords = coords
                    .into_iter()
                    .map(|arr| PyArray1::from_owned_array(py, arr))
                    .collect();
                (
                    shape,
                    unsafe { bound_data.cast_into_unchecked() },
                    bound_coords,
                )
            }
        }
    }

    fn oindex(&self, indexers: Vec<(usize, usize)>) -> Self {
        let slices = indexers
            .into_iter()
            .map(|(start, end)| SliceInfoElem::Slice {
                start: start as isize,
                end: Some(end as isize),
                step: 1,
            })
            .collect::<Vec<_>>();

        match self {
            Self::Bool(c) => Self::Bool(c.oindex(slices.as_slice())),
            Self::Int8(c) => Self::Int8(c.oindex(slices.as_slice())),
            Self::Int16(c) => Self::Int16(c.oindex(slices.as_slice())),
            Self::Int32(c) => Self::Int32(c.oindex(slices.as_slice())),
            Self::Int64(c) => Self::Int64(c.oindex(slices.as_slice())),
            Self::UInt8(c) => Self::UInt8(c.oindex(slices.as_slice())),
            Self::UInt16(c) => Self::UInt16(c.oindex(slices.as_slice())),
            Self::UInt32(c) => Self::UInt32(c.oindex(slices.as_slice())),
            Self::UInt64(c) => Self::UInt64(c.oindex(slices.as_slice())),
            Self::Float32(c) => Self::Float32(c.oindex(slices.as_slice())),
            Self::Float64(c) => Self::Float64(c.oindex(slices.as_slice())),
            Self::Complex32(c) => Self::Complex32(c.oindex(slices.as_slice())),
            Self::Complex64(c) => Self::Complex64(c.oindex(slices.as_slice())),
        }
    }
}

#[pyclass]
#[pyo3(name = "COO")]
pub struct PyCoo {
    container: Container,
    fill_value: Py<PyAny>,
}

#[pymethods]
impl PyCoo {
    #[new]
    fn new<'py>(
        py: Python<'py>,
        data: Bound<'py, PyUntypedArray>,
        coords: Vec<Bound<'py, PyArray1<usize>>>,
        shape: Vec<usize>,
        fill_value: Bound<'py, PyAny>,
    ) -> PyResult<Self> {
        let element_type = data.dtype();
        let coords_: Vec<Array1<usize>> = coords
            .into_iter()
            .map(|c| unsafe { c.as_array() }.into_owned())
            .collect();

        let container = if element_type.is_equiv_to(&dtype::<bool>(py)) {
            let cast_data = data.cast_into::<PyArray1<bool>>()?;
            let array = unsafe { cast_data.as_array() }.into_owned();

            Ok(Container::Bool(Coo::<bool>::new(shape, array, coords_)))
        } else if element_type.is_equiv_to(&dtype::<i8>(py)) {
            let cast_data = data.cast_into::<PyArray1<i8>>()?;
            let array = unsafe { cast_data.as_array() }.into_owned();

            Ok(Container::Int8(Coo::<i8>::new(shape, array, coords_)))
        } else if element_type.is_equiv_to(&dtype::<i16>(py)) {
            let cast_data = data.cast_into::<PyArray1<i16>>()?;
            let array = unsafe { cast_data.as_array() }.into_owned();

            Ok(Container::Int16(Coo::<i16>::new(shape, array, coords_)))
        } else if element_type.is_equiv_to(&dtype::<i32>(py)) {
            let cast_data = data.cast_into::<PyArray1<i32>>()?;
            let array = unsafe { cast_data.as_array() }.into_owned();

            Ok(Container::Int32(Coo::<i32>::new(shape, array, coords_)))
        } else if element_type.is_equiv_to(&dtype::<i64>(py)) {
            let cast_data = data.cast_into::<PyArray1<i64>>()?;
            let array = unsafe { cast_data.as_array() }.into_owned();

            Ok(Container::Int64(Coo::<i64>::new(shape, array, coords_)))
        } else if element_type.is_equiv_to(&dtype::<u8>(py)) {
            let cast_data = data.cast_into::<PyArray1<u8>>()?;
            let array = unsafe { cast_data.as_array() }.into_owned();

            Ok(Container::UInt8(Coo::<u8>::new(shape, array, coords_)))
        } else if element_type.is_equiv_to(&dtype::<u16>(py)) {
            let cast_data = data.cast_into::<PyArray1<u16>>()?;
            let array = unsafe { cast_data.as_array() }.into_owned();

            Ok(Container::UInt16(Coo::<u16>::new(shape, array, coords_)))
        } else if element_type.is_equiv_to(&dtype::<u32>(py)) {
            let cast_data = data.cast_into::<PyArray1<u32>>()?;
            let array = unsafe { cast_data.as_array() }.into_owned();

            Ok(Container::UInt32(Coo::<u32>::new(shape, array, coords_)))
        } else if element_type.is_equiv_to(&dtype::<u64>(py)) {
            let cast_data = data.cast_into::<PyArray1<u64>>()?;
            let array = unsafe { cast_data.as_array() }.into_owned();

            Ok(Container::UInt64(Coo::<u64>::new(shape, array, coords_)))
        } else if element_type.is_equiv_to(&dtype::<f32>(py)) {
            let cast_data = data.cast_into::<PyArray1<f32>>()?;
            let array = unsafe { cast_data.as_array() }.into_owned();

            Ok(Container::Float32(Coo::<f32>::new(shape, array, coords_)))
        } else if element_type.is_equiv_to(&dtype::<f64>(py)) {
            let cast_data = data.cast_into::<PyArray1<f64>>()?;
            let array = unsafe { cast_data.as_array() }.into_owned();

            Ok(Container::Float64(Coo::<f64>::new(shape, array, coords_)))
        } else if element_type.is_equiv_to(&dtype::<num_complex::Complex32>(py)) {
            let cast_data = data.cast_into::<PyArray1<num_complex::Complex32>>()?;
            let array = unsafe { cast_data.as_array() }.into_owned();

            Ok(Container::Complex32(Coo::<num_complex::Complex32>::new(
                shape, array, coords_,
            )))
        } else if element_type.is_equiv_to(&dtype::<num_complex::Complex64>(py)) {
            let cast_data = data.cast_into::<PyArray1<num_complex::Complex64>>()?;
            let array = unsafe { cast_data.as_array() }.into_owned();

            Ok(Container::Complex64(Coo::<num_complex::Complex64>::new(
                shape, array, coords_,
            )))
        } else {
            Err(PyValueError::new_err("unknown dtype: {element_type:?}"))
        }?;

        Ok(Self {
            container,
            fill_value: fill_value.unbind(),
        })
    }

    fn oindex<'py>(&self, py: Python<'py>, indexers: Vec<Bound<'py, PySlice>>) -> PyResult<Self> {
        let converted_indexers: Vec<_> = indexers
            .into_iter()
            .map(|s| -> PyResult<(usize, usize)> {
                Ok((
                    s.getattr("start")?.extract::<usize>()?,
                    s.getattr("stop")?.extract::<usize>()?,
                ))
            })
            .collect::<PyResult<Vec<_>>>()?;

        Ok(Self {
            container: self.container.oindex(converted_indexers),
            fill_value: self.fill_value.clone_ref(py),
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::{array, s};

    fn create_coo_2d_f64() -> Coo<f64> {
        let shape: Vec<usize> = vec![10, 10];
        let data: Array1<f64> = array![1.3, 4.7, 2.6];
        let coords: Array2<usize> = array![[2, 4], [7, 0], [4, 9]];

        Coo::new(
            shape,
            data,
            coords
                .columns()
                .into_iter()
                .map(|c| c.into_owned())
                .collect::<Vec<_>>(),
        )
    }

    fn create_coo_3d_i32() -> Coo<i32> {
        let shape: Vec<usize> = vec![10, 10, 15];
        let data: Array1<i32> = array![1, -1, 50, 2000];
        let coords: Array2<usize> = array![[2, 4, 2], [7, 0, 14], [4, 9, 5], [9, 2, 8]];

        Coo::new(
            shape,
            data,
            coords
                .columns()
                .into_iter()
                .map(|c| c.into_owned())
                .collect::<Vec<_>>(),
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

        let obj = Coo::new(shape.clone(), data.clone(), coords.clone());

        assert_eq!(obj.shape, shape);

        assert_eq!(obj.data, data);
        assert_eq!(obj.coords, coords);

        let data2: Array1<bool> = array![false, true, false];
        let obj2 = Coo::new(shape.clone(), data2.clone(), coords.clone());

        assert_eq!(obj2.shape, shape);

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
        let obj = create_coo_3d_i32();

        assert_eq!(obj.ndim(), 3);
    }

    #[test]
    fn test_oindex_2d_some() {
        let obj = create_coo_2d_f64();

        let actual = obj.oindex(s![..5, ..5]);
        let expected = Coo::<f64>::new(vec![5, 5], array![1.3], vec![array![2], array![4]]);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_oindex_2d_none() {
        let obj = create_coo_2d_f64();

        let actual = obj.oindex(s![10..15, 10..15]);
        let expected = Coo::<f64>::new(vec![0, 0], array![], vec![array![], array![]]);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_oindex_2d_empty() {
        let obj = create_coo_2d_f64();

        let actual = obj.oindex(s![5..5, 5..5]);
        let expected = Coo::<f64>::new(vec![0, 0], array![], vec![array![], array![]]);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_oindex_3d_some() {
        let obj = create_coo_3d_i32();

        let actual = obj.oindex(s![5..10, 0..5, 5..15]);
        let expected = Coo::<i32>::new(
            vec![5, 5, 10],
            array![-1, 2000],
            vec![array![2, 4], array![0, 2], array![9, 3]],
        );

        assert_eq!(actual, expected);
    }
}
