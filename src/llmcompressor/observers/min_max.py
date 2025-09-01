from typing import Any, Optional, Tuple

import torch
from compressed_tensors.quantization.quant_args import QuantizationArgs
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam
from compressed_tensors.utils import deprecated

from llmcompressor.observers.base import Observer

__all__ = ["MinMaxObserver", "MovingAverageMinMaxObserver"]


@Observer.register("minmax")
class MinMaxObserver(Observer):
    """
    Implements a quantization observer that calculates scale and zero point based on the
    minimum and maximum values of the tensor being observed.
    averaging_constant is used to trigger moving average of min and max values,
    following the formula:
    new_val = tracked_val + averaging_constant * (observed_val - tracked_val)
    Default behavior is to disable averaging, and return the observed absolute
    min and max values.
    """

    def __init__(
        self,
        quantization_args: QuantizationArgs,
        averaging_constant: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(quantization_args=quantization_args)

        self.min_val = {}
        self.max_val = {}
        self.averaging_constant = averaging_constant

    def calculate_updated_min_max(
        self,
        observed: torch.Tensor,
        reduce_dims: Optional[Tuple[int]] = None,
        tensor_id: Optional[Any] = None,
    ):
        """
        Updates the observed min and max by either tracking the observed absolute
        min and max values (averaging_constant = None) or by using a moving average
        smoothed by the averaging_constant.

        :param observed: observed tensor to calculate quantization parameters for
        :param reduce_dims: optional tuple of dimensions to reduce along,
            returned scale and zero point will be shaped (1,) along the
            reduced dimensions
        :param tensor_id: Optional id if different ranges of observed tensors are
            passed, useful for sharding tensors by group_size
        :return: updated min and max values
        """
        tensor_id = tensor_id or "default"

        if not reduce_dims:
            observed_min_val, observed_max_val = torch.aminmax(observed)
        else:
            observed_min_val = torch.amin(observed, dim=reduce_dims, keepdims=True)
            observed_max_val = torch.amax(observed, dim=reduce_dims, keepdims=True)

        tracked_min_val = self.min_val.get(tensor_id, None)
        tracked_max_val = self.max_val.get(tensor_id, None)

        if tracked_min_val is None or tracked_max_val is None:
            updated_min_val = observed_min_val
            updated_max_val = observed_max_val
        elif self.averaging_constant is None:  # tracking absolute min and max
            updated_min_val = torch.minimum(observed_min_val, tracked_min_val)
            updated_max_val = torch.maximum(observed_max_val, tracked_max_val)
        else:  # tracking moving average of min and max
            updated_min_val = tracked_min_val + self.averaging_constant * (
                observed_min_val - tracked_min_val
            )
            updated_max_val = tracked_max_val + self.averaging_constant * (
                observed_max_val - tracked_max_val
            )

        self.min_val[tensor_id] = updated_min_val
        self.max_val[tensor_id] = updated_max_val
        return updated_min_val, updated_max_val

    def calculate_gparam(self, observed: torch.Tensor) -> torch.Tensor:
        """
        Generate a global scale using the observed min and max.

        :param observed: observed tensor to calculate quantization parameters for
        :return: updated global scale derived from the observed tensor
        """

        updated_min_val, updated_max_val = self.calculate_updated_min_max(
            observed=observed
        )
        return generate_gparam(
            updated_min_val=updated_min_val, updated_max_val=updated_max_val
        )

    def calculate_qparams(
        self,
        observed: torch.Tensor,
        reduce_dims: Optional[Tuple[int]] = None,
        tensor_id: Optional[Any] = None,
        global_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.IntTensor]:
        """
        Generate a scale and zero-point using the observed min and max.

        :param observed: observed tensor to calculate quantization parameters for
        :param reduce_dims: optional tuple of dimensions to reduce along,
            returned scale and zero point will be shaped (1,) along the
            reduced dimensions
        :param tensor_id: Optional id if different ranges of observed tensors are
            passed, useful for sharding tensors by group_size
        :param global_scale: optional scale to further scale local quantization scales
        :return: tuple of scale and zero point derived from the observed tensor
        """

        updated_min_val, updated_max_val = self.calculate_updated_min_max(
            observed=observed, tensor_id=tensor_id, reduce_dims=reduce_dims
        )
        return calculate_qparams(
            min_vals=updated_min_val,
            max_vals=updated_max_val,
            quantization_args=self.quantization_args,
            global_scale=global_scale,
        )

    def get_qparams_along_dim(
        self,
        observed: torch.Tensor,
        dim: int,
        tensor_id: Optional[Any] = None,
        global_scale: Optional[torch.Tensor] = None,
    ):
        """
        Calculate quantization parameters along the specified dimension
        """
        reduce_dims = tuple(idx for idx in range(observed.ndim) if idx != dim)
        return self.calculate_qparams(
            observed,
            reduce_dims=reduce_dims,
            tensor_id=tensor_id,
            global_scale=global_scale,
        )

    def reset(self):
        """
        Reset the state of the observer, including min and maximum values
        """
        super().reset()
        self.min_val = {}
        self.max_val = {}


class MovingAverageMinMaxObserver(MinMaxObserver):
    @deprecated(
        message=(
            "The class name `MovingAverageMinMaxObserver` has been deprecated, please "
            "initialize with `MinMaxObserver` in the future"
        )
    )
    def __new__(cls, *args, **kwargs):
        return super().__new__(MinMaxObserver, *args, **kwargs)
