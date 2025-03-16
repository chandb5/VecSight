import numpy as np
import pickle

class Reader:
    """
    A reusable class to read .fvecs, .bvecs, and .ivecs files,
    convert them to numpy arrays, and store the data in memory.
    """
    def __init__(self, filename: str):
        self.filename = filename
        self.data = None
        self.read()

    def read(self):
        """Reads the file based on its extension and stores the data."""
        if self.filename.endswith(".fvecs"):
            self.data = self._fvecs_read(self.filename)
        elif self.filename.endswith(".ivecs"):
            self.data = self._ivecs_read(self.filename)
        else:
            raise ValueError(f"Unsupported file type: {self.filename}")
        
    def save_as_pickle(self, output_filename):
        with open(output_filename, 'wb') as f:
            pickle.dump(self.data, f)

    @staticmethod
    def _fvecs_read(filename: str, c_contiguous: bool = True) -> np.ndarray:
        """Reads .fvecs file and returns a numpy array."""
        fv = np.fromfile(filename, dtype=np.float32)
        if fv.size == 0:
            return np.zeros((0, 0))
        
        dim = fv.view(np.int32)[0]
        if dim <= 0:
            raise ValueError(f"Invalid dimension size in {filename}")
        
        fv = fv.reshape(-1, 1 + dim)
        if not np.all(fv.view(np.int32)[:, 0] == dim):
            raise IOError(f"Non-uniform vector sizes in {filename}")
        
        fv = fv[:, 1:]  # Remove the first column containing dimension info
        return fv.copy() if c_contiguous else fv

    @staticmethod
    def _ivecs_read(filename: str) -> np.ndarray:
        """Reads .ivecs file and returns a numpy array"""
        return Reader._fvecs_read(filename).view(np.float32)
