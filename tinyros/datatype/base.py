"""Base class for all data types in TinyROS."""

import pickle
from dataclasses import dataclass
from typing import Any, Tuple

import cupy as cp
from cupy.cuda import MemoryPointer, UnownedMemory


@dataclass
class SerializedCuPyArray:
    """Serialized representation of a CuPy array."""

    ptr: int
    ipc_handle: bytes
    shape: Tuple[int, ...]
    dtype: cp.dtype
    strides: Tuple[int, ...]
    device: int


class DataType:
    """Base class for all data types in TinyROS."""

    def can_serialize(self, data: Any) -> bool:
        """Check if the data type can be serialized.

        Returns:
            bool: True if the data type can be serialized, False otherwise.
        """
        return False

    def can_deserialize(self, data: Any) -> bool:
        """Check if the data type can be deserialized.

        Returns:
            bool: True if the data type can be deserialized, False otherwise.
        """
        return False

    def serialize_data(self, data: Any) -> Any:
        """Serialize the data type into a format suitable for transmission.

        Args:
            data (Any): The data to serialize.

        Returns:
            Any: Serialized data.
        """
        raise NotImplementedError("Subclasses must implement serialize_data method.")

    def deserialize_data(self, data: Any) -> Any:
        """Deserialize the data type from a transmitted format.

        Args:
            data (Any): The serialized data to deserialize.

        Returns:
            Any: Deserialized data.
        """
        raise NotImplementedError("Subclasses must implement deserialize_data method.")

    def serialize(self) -> bytes:
        """Serialize the data type into a dictionary.

        CuPy arrays are serialized into IPC handles to avoid copying data,
        while other attributes are copied as-is, unless subclasses override:
        - can_serialize
        - serialize_data

        NOTE: This means that GPU arrays will not be unloaded!!!! :D
        TODO: Improve efficiency also for boring numpy arrays.

        Returns:
            bytes: Pickled dictionary containing serialized data.
        """
        serialized_data = {}
        for key in self.__dict__:
            if isinstance(self.__dict__[key], cp.ndarray):
                serialized_data[key] = self.serialize_cupy(self.__dict__[key])
            elif self.can_serialize(self.__dict__[key]):
                serialized_data[key] = self.serialize_data(self.__dict__[key])
            else:
                # Sorry, copy may happen
                serialized_data[key] = self.__dict__[key]

        return pickle.dumps(serialized_data)

    def serialize_cupy(self, arr_cp: cp.ndarray) -> SerializedCuPyArray:
        """Serialize a CuPy array into a SerializedCuPyArray.

        This method is used to serialize CuPy arrays specifically, extracting
        the IPC handle, shape, dtype, and strides.

        Args:
            arr_cp (cp.ndarray): The CuPy array to serialize.

        Returns:
            SerializedCuPyArray: An object containing serialized CuPy array data.
        """
        ptr = int(arr_cp.data.ptr)
        ipc_handle = cp.cuda.runtime.ipcGetMemHandle(ptr)
        return SerializedCuPyArray(
            ptr=ptr,
            ipc_handle=ipc_handle,
            shape=arr_cp.shape,
            dtype=arr_cp.dtype,
            strides=arr_cp.strides,
            device=arr_cp.device.id,
        )

    def deserialize_cupy(self, serialized: SerializedCuPyArray) -> cp.ndarray:
        """Deserialize a SerializedCuPyArray into a CuPy ndarray.

        This method reconstructs a CuPy array from the serialized data,
        including the IPC handle, shape, dtype, and strides.

        Args:
            serialized (SerializedCuPyArray): The serialized CuPy array data.

        Returns:
            cp.ndarray: The reconstructed CuPy ndarray.
        """
        shape = serialized.shape
        dtype = serialized.dtype
        strides = serialized.strides
        dev_id = serialized.device

        # Switch to the correct device/context
        with cp.cuda.Device(dev_id):
            try:
                # Try opening IPC handle (for cross-process)
                ptr = cp.cuda.runtime.ipcOpenMemHandle(
                    serialized.ipc_handle,
                    cp.cuda.runtime.cudaIpcMemLazyEnablePeerAccess,
                )
            except cp.cuda.runtime.CUDARuntimeError:
                # Fallback: same-context, raw pointer
                ptr = serialized.ptr

            # Wrap in UnownedMemory + MemoryPointer
            size = int(
                cp.dtype(dtype).itemsize
                * shape[0]
                * (strides[0] // cp.dtype(dtype).itemsize)
            )
            mem = UnownedMemory(ptr, size, owner=None)
            mp = MemoryPointer(mem, 0)
            return cp.ndarray(shape, dtype=dtype, memptr=mp, strides=strides)

    def deserialize(self, data: bytes) -> "DataType":
        """Deserialize the data type from a dictionary.

        Reconstructs CuPy arrays from serialized IPC handles and restores
        other attributes as-is.

        Args:
            data (bytes): The pickled serialized data dictionary.

        Returns:
            DataType: An instance of the DataType with restored attributes.
        """
        data = pickle.loads(data)
        for key in data:
            if isinstance(data[key], SerializedCuPyArray):
                # Reconstruct CuPy array from serialized data
                self.__dict__[key] = self.deserialize_cupy(data[key])
            elif self.can_deserialize(data[key]):
                # Use custom deserialization method if available
                self.__dict__[key] = self.deserialize_data(data[key])
            else:
                self.__dict__[key] = data[key]
        return self
