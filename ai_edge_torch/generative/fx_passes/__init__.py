import torch

from ai_edge_torch.convert.fx_passes import CanonicalizePass
from ai_edge_torch.convert.fx_passes import run_passes
from ai_edge_torch.generative.fx_passes.remove_sdpa_zero_mask_pass import RemoveSDPACompositeZeroMaskPass  # NOQA


def run_generative_passes(
    exported_program: torch.export.ExportedProgram,
) -> torch.export.ExportedProgram:
  return run_passes(
      exported_program,
      [
          RemoveSDPACompositeZeroMaskPass(),
          CanonicalizePass(),
      ],
  )
