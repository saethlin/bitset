pub struct WriteOnly<T> {
    inner: T,
}

impl<T> WriteOnly<T> where T: Copy {
    fn write(&mut self, new: T) {
        self.inner = new;
    }
}
