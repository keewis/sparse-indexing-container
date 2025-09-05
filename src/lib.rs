use pyo3::prelude::*;

mod container;
mod coo;
mod errors;

#[pymodule]
mod sparse_indexing_container {
    #[pymodule_export]
    use crate::coo::PyCoo;
}
