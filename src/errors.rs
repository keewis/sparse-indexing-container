use std::error::Error;

#[derive(Debug)]
pub struct IndexError {
    message: String,
}

impl IndexError {
    pub fn new_err(message: String) -> Self {
        Self { message }
    }
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

impl ConcatError {
    pub fn new_err(message: String) -> Self {
        Self { message }
    }
}

impl std::fmt::Display for ConcatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for ConcatError {}
