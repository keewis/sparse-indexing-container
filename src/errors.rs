use std::error::Error;

#[derive(Debug)]
pub struct IndexError {
    message: String,
}

impl std::fmt::Display for IndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for IndexError {}

#[derive(Debug)]
pub struct ConcatError {
    message: String,
}

impl std::fmt::Display for ConcatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for ConcatError {}
