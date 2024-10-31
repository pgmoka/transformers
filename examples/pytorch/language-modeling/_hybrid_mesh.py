import dataclasses
import collections
import numpy as np
from typing import Sequence, Any
from jax.experimental import mesh_utils
import jax

def create_custom_64x4_device_mesh(
    mesh_shape: Sequence[int],
    dcn_mesh_shape: Sequence[int],
    devices: Sequence[Any],
    process_is_granule: bool = False,
    should_sort_granules_by_key: bool = True,
) -> np.ndarray:
  """Custom device mesh for 64x4 ici parallelism"""
  assert len(devices) % 256 == 0, f"This custom mesh is not valid for {len(devices)} devices"
  attr = "process_index" if process_is_granule else "slice_index"
  if not hasattr(devices[0], attr):
    raise ValueError(f"Device {devices[0]} does not have attribute {attr}. See" " `process_is_granule` option.")
  granule_dict = collections.defaultdict(list)
  for dev in devices:
    granule_dict[getattr(dev, attr)].append(dev)
  granules = (
      [granule_dict[key] for key in sorted(granule_dict.keys())] if should_sort_granules_by_key else granule_dict.values()
  )
  if np.prod(dcn_mesh_shape) != len(granules):
    raise ValueError(f"Number of slices {len(granules)} must equal the product of " f"dcn_mesh_shape {dcn_mesh_shape}")
  per_granule_meshes = [
      mesh_utils.create_device_mesh(
          [16, 16],
          granule,
          allow_split_physical_axes=False,
      )
      for granule in granules
  ]

  def reshape_mesh_to_rings(a):
    b = []
    for i in range(8):
      b.append([])
      for j in range(8):
        a_i = i * 2
        a_j = j * 2
        # forms a ring of size 4
        b[i].append([a[a_i, a_j], a[a_i, a_j + 1], a[a_i + 1, a_j + 1], a[a_i + 1, a_j]])
    b = np.array(b)
    b = np.reshape(b, (64, 4))
    return b

  per_granule_meshes = [np.reshape(reshape_mesh_to_rings(x), mesh_shape) for x in per_granule_meshes]
  # TODO(jekbradbury): handle non-uniform DCN topologies
  granule_mesh = np.arange(len(granules)).reshape(dcn_mesh_shape)
  blocks = np.vectorize(lambda i: per_granule_meshes[i], otypes=[object])(granule_mesh)
  device_mesh = np.block(blocks.tolist())
  return device_mesh


@dataclasses.dataclass
class Device:
  process_index: int
  slice_index: int
  uid: int
  device_kind: str = ''
  platform: str = 'cpu'


def get_hybrid_mesh(ici_mesh_shape: Sequence[int], dcn_mesh_shape: Sequence[int], num_devices: int, num_slices: int) -> np.ndarray:
  num_devices_per_granule = num_devices // num_slices
  devices = [Device(i // num_devices_per_granule, i // num_devices_per_granule, i) for i in range(num_devices)]
  devices = create_custom_64x4_device_mesh(ici_mesh_shape, dcn_mesh_shape, devices).reshape(-1).tolist()
  devices = np.array(jax.tree_map(lambda d: d.uid, devices))
  return devices
