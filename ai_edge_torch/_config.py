# Copyright 2024 The AI Edge Torch Authors.
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

"""Provides a configuration for the ai-edge-torch."""

import functools
import logging
import os

__all__ = ["config"]


class _Config:
  """ai-edge-torch global configs."""

  @property
  @functools.cache  # pylint: disable=method-cache-max-size-none
  def use_torch_xla(self) -> bool:
    """True if using torch_xla to lower torch ops to StableHLO.

    To use torch_xla as the lowering backend, set environment variable
    `USE_TORCH_XLA` to "true".
    """
    var = os.environ.get("USE_TORCH_XLA", "false")
    var = var.lower().strip()
    if var in ("y", "yes", "t", "true", "on", "1"):
      return True
    elif var in ("n", "no", "f", "false", "off", "0"):
      return False
    else:
      logging.warning("Invalid USE_TORCH_XLA value is ignored: %s.", var)
      return False

  @property
  def in_oss(self) -> bool:
    """True if the code is not running in google internal environment."""
    return True


config = _Config()
