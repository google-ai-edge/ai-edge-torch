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
"""Script to update the nightly dependency versions in `requirements.txt`.

Usage (host bash): python ci/update_nightly_versions.py
"""
import argparse
from datetime import datetime
from datetime import timedelta
import functools
import json
from pathlib import Path
import re
import time
import urllib.request

REPO_PATH = Path(__file__).parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--nightly-date", default=None)
args = parser.parse_args()


@functools.cache
def torch_nightly_index():
  with urllib.request.urlopen(
      "https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html"
  ) as response:
    return response.read().decode("utf-8")


def torch_version(package, nightly_date_str):
  versions = re.findall(
      f"{package}-([0-9.]+\.dev{nightly_date_str})+%2Bcpu", torch_nightly_index()
  )
  if not versions:
    raise Exception(
        f"{package} {nightly_date_str} nightly does not exist in the index."
    )
  return sorted(versions)[-1]


def tf_version(nightly_date_str):
  with urllib.request.urlopen("https://pypi.org/pypi/tf-nightly/json") as response:
    tf_index = json.loads(response.read())

  releases = tf_index["releases"]
  versions = [ver for ver in releases.keys() if nightly_date_str in ver]
  if not versions:
    raise Exception(
        f"tf-nightly {nightly_date_str} nightly does not exist in the index."
    )
  return sorted(versions)[-1]


def torch_xla_wheel(nightly_date_str, cpver):
  url = f"https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly+{nightly_date_str}-{cpver}-{cpver}-linux_x86_64.whl"
  with urllib.request.urlopen(url) as response:
    assert response.getcode() == 200
  return url


def main():
  if args.nightly_date is None:
    nightly_date = datetime.now()
  else:
    nightly_date = datetime.strptime(args.nightly_date, "%Y%m%d")

  nightly_date_str = nightly_date.strftime("%Y%m%d")
  requirements_file = REPO_PATH / "requirements.txt"
  requirements = requirements_file.read_text()

  def sub(k, v, suffix=""):
    nonlocal requirements
    pattern = f"{k}\s*(==|>=|@)\s*[^\n;]+{suffix}"
    assert re.findall(pattern, requirements, re.MULTILINE)
    requirements = re.sub(
        pattern,
        f"{k}\\g<1>{v}" + suffix,
        requirements,
        flags=re.MULTILINE,
        count=1,
    )

  sub("tf-nightly", tf_version(nightly_date_str))

  sub("torch", torch_version("torch", nightly_date_str) + "+cpu")
  sub("torchvision", torch_version("torchvision", nightly_date_str) + "+cpu")
  sub("torchaudio", torch_version("torchaudio", nightly_date_str) + "+cpu")

  sub("torch_xla", torch_xla_wheel(nightly_date_str, "cp39"), '; python_version=="3.9"')
  sub(
      "torch_xla",
      torch_xla_wheel(nightly_date_str, "cp310"),
      '; python_version=="3.10"',
  )
  sub(
      "torch_xla",
      torch_xla_wheel(nightly_date_str, "cp311"),
      '; python_version=="3.11"',
  )

  requirements_file.write_text(requirements)

  readme_file = REPO_PATH / "README.md"
  readme = readme_file.read_text()

  readme = re.sub(
      "badge/torch-[^-]+",
      f"badge/torch-{torch_version('torch', nightly_date_str)}",
      readme,
  )
  readme_file.write_text(readme)


if __name__ == "__main__":
  main()
