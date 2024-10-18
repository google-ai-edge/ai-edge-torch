"""Usage: python dedup_composite_subgraph.py -i <infile> -o <outfile>"""

import argparse
import pathlib
import flatbuffers
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--input", "-i")
parser.add_argument("--output", "-o")
parser.add_argument(
    "--large", action=argparse.BooleanOptionalAction, default=False
)


def convert_object_to_bytearray(model_object, extra_buffer=b""):
  _TFLITE_FILE_IDENTIFIER = b"TFL3"
  # Initial size of the buffer, which will grow automatically if needed
  builder = flatbuffers.Builder(1024)
  model_offset = model_object.Pack(builder)
  builder.Finish(model_offset, file_identifier=_TFLITE_FILE_IDENTIFIER)
  model_bytearray = bytes(builder.Output())
  model_bytearray = model_bytearray + extra_buffer
  return model_bytearray


def process_constant_map(model) -> int:
  buffer_size = 0
  constant_map = []
  for buffer in model.buffers:
    if buffer.data is None:
      constant_map.append(buffer.data)
    elif isinstance(buffer.data, np.ndarray):
      constant_map.append(buffer.data.tobytes())
      buffer_size += len(buffer.data.tobytes())
    else:
      constant_map.append(buffer.data)
      buffer_size += len(buffer.data)
  return buffer_size, constant_map


def serialize_large_model(model, constant_map) -> bytearray:
  # remove all the constant from the model.
  for buffer in model.buffers:
    if buffer.data is not None:
      buffer.data = None
      buffer.offset = 1
      buffer.size = 1
  dummy_bytearray = convert_object_to_bytearray(model)
  # calculate the correct buffer size and offset
  while len(dummy_bytearray) % 16:
    dummy_bytearray += b"\0"
  for buffer_idx, buffer in enumerate(model.buffers):
    buffer_data = constant_map[buffer_idx]
    if buffer_data is None:
      continue
    buffer.offset = len(dummy_bytearray)
    buffer.size = len(buffer_data)
    dummy_bytearray += buffer_data
    while len(dummy_bytearray) % 16:
      dummy_bytearray += b"\0"
  del dummy_bytearray

  # build new tflite file with correct buffer offset
  model_bytearray = convert_object_to_bytearray(model)
  while len(model_bytearray) % 16:
    model_bytearray += b"\0"
  for buffer_idx, _ in enumerate(model.buffers):
    buffer_data = constant_map[buffer_idx]
    if buffer_data is None:
      continue
    model_bytearray += buffer_data
    while len(model_bytearray) % 16:
      model_bytearray += b"\0"
  return model_bytearray


def main() -> None:
  args = parser.parse_args()

  from tensorflow.lite.python import schema_py_generated as schema_fb
  from tensorflow.lite.tools import flatbuffer_utils

  builtin_ops = {
      getattr(schema_fb.BuiltinOperator, k): k
      for k in dir(schema_fb.BuiltinOperator)
      if not k.startswith("_")
  }

  model = flatbuffer_utils.read_model(args.input)

  def get_opname(op):
    opcode = model.operatorCodes[op.opcodeIndex].builtinCode
    opname = builtin_ops.get(opcode, None)
    return opname

  composite_subgraph_indices = set()

  for subgraph in model.subgraphs:
    for op in subgraph.operators:
      if get_opname(op) == "STABLEHLO_COMPOSITE":
        subgraph_index = op.builtinOptions2.decompositionSubgraphIndex
        composite_subgraph_indices.add(subgraph_index)

  for i in composite_subgraph_indices:
    subgraph = model.subgraphs[i]
    delete_ops = True

    new_outputs = []
    for output in subgraph.outputs:
      for input in subgraph.inputs:
        ti = subgraph.tensors[input]
        to = subgraph.tensors[output]
        if ti.shape.tolist() == to.shape.tolist() and ti.type == to.type:
          new_outputs.append(input)
          break
      else:
        delete_ops = False
        break

    if delete_ops:
      print(f"Deleting operators in subgraph {i}")
      assert max(subgraph.inputs) <= len(subgraph.inputs)
      subgraph.operators = []
      subgraph.outputs = new_outputs
      subgraph.tensors = subgraph.tensors[: max(subgraph.inputs) + 1]

  print(f"Writing model to {args.output}...")
  if args.large:
    print(f"USING LARGE MODEL SERIALIZER")
    _, constant_map = process_constant_map(model)
    with open(args.output, "wb") as f:
      f.write(serialize_large_model(model, constant_map))
  else:
    flatbuffer_utils.write_model(model, args.output)


if __name__ == "__main__":
  main()