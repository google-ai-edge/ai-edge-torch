import unittest

import torch

from ai_edge_torch.debug import _search_model


class TestSearchModel(unittest.TestCase):

  def test_search_model_with_ops(self):
    class MultipleOpsModel(torch.nn.Module):

      def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sub_0 = x - 1
        add_0 = y + 1
        mul_0 = x * y
        add_1 = sub_0 + add_0
        mul_1 = add_0 * mul_0
        sub_1 = add_1 - mul_1
        return sub_1

    model = MultipleOpsModel().eval()
    args = (torch.rand(10), torch.rand(10))

    def find_subgraph_with_sub(fx_gm, inputs):
      return torch.ops.aten.sub.Tensor in [n.target for n in fx_gm.graph.nodes]

    results = list(_search_model(find_subgraph_with_sub, model, args))
    self.assertEqual(len(results), 2)
    self.assertIn(
        torch.ops.aten.sub.Tensor, list([n.target for n in results[0].graph.nodes])
    )


if __name__ == "__main__":
  unittest.main()
