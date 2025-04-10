# Copyright 2025 The AI Edge Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# A listing of types describes the K and V tensors in KV caches.

import enum
from enum import Enum, auto
from typing import Tuple
from torch import nn


@enum.unique
class TensorDims(Enum):
  BATCH = enum.auto()
  SEQUENCE = enum.auto()
  NUM_HEADS = enum.auto()
  HEAD_DIM = enum.auto()
  MODEL_DIM = enum.auto()  # often num_heads * head_dim


DIM_TO_LETTER = {
    TensorDims.BATCH: 'B',
    TensorDims.SEQUENCE: 'T',
    TensorDims.NUM_HEADS: 'N',
    TensorDims.HEAD_DIM: 'H',
    TensorDims.MODEL_DIM: 'D',
}


class TensorDimensionMeta(type):
  """Metaclass to create classes representing an order of tensor dimensions."""

  def __new__(cls, name, bases, attrs, dimensions: Tuple[TensorDims]):
    """Creates a new class with the given name and tensor dimension order.

    Args:
      name: Name of the new class.
      bases: Base classes for the new class.
      attrs: Attributes for the new class.
      dimensions: A tuple of TensorDims defining the order.
    """

    attrs['dimensions'] = (
        dimensions  # Store the dimensions as a class attribute
    )
    return super().__new__(cls, name, bases, attrs)

  def __init__(cls, name, bases, attrs, dimensions: Tuple[TensorDims]):
    super().__init__(name, bases, attrs)

  def __repr__(cls):
    return f'{cls.__name__}'

  def __iter__(cls):
    return iter(getattr(cls, 'dimensions'))


def create_tensor_dimension_order_class(dims: Tuple[TensorDims]):
  """Creates a TensorDimensionMeta class with the specified dimensions.

  Args:
    dimensions: A tuple of TensorDims.

  Returns:
    A new class representing the tensor dimension order.
  """
  name = ''.join(DIM_TO_LETTER[d] for d in dims)
  # Derive from nn.Module for torch tracing compatiblity.
  return TensorDimensionMeta(name, (nn.Module,), {}, dimensions=dims)


BTNH = create_tensor_dimension_order_class((
    TensorDims.BATCH,
    TensorDims.SEQUENCE,
    TensorDims.NUM_HEADS,
    TensorDims.HEAD_DIM,
))
BNTH = create_tensor_dimension_order_class((
    TensorDims.BATCH,
    TensorDims.NUM_HEADS,
    TensorDims.SEQUENCE,
    TensorDims.HEAD_DIM,
))
BNHT = create_tensor_dimension_order_class((
    TensorDims.BATCH,
    TensorDims.NUM_HEADS,
    TensorDims.HEAD_DIM,
    TensorDims.SEQUENCE,
))
