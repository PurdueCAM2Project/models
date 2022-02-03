import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.mplot3d import art3d

from official.vision.beta.projects.mesh_rcnn.ops.cubify import cubify


def create_voxels(grid_dims, batch_size, occupancy_locs):
  ones = tf.ones(shape=[len(occupancy_locs)])
  voxels = tf.scatter_nd(
      indices=tf.convert_to_tensor(occupancy_locs, tf.int32),
      updates=ones,
      shape=[batch_size, grid_dims, grid_dims, grid_dims])
    
  return voxels

def visualize_mesh(verts, faces, verts_mask, faces_mask):
  for i in range(verts.shape[0]):
    v = verts[i].numpy()
    f = faces[i].numpy()
    vm = verts_mask[i].numpy() == 1
    fm = faces_mask[i].numpy() == 1

    new_f = f[fm]
    fig = plt.figure(i)
    ax = fig.add_subplot(projection="3d")

    new_v = v[new_f]
    pc = art3d.Poly3DCollection(new_v, facecolors=(1, 0.5, 1, 0.3), edgecolor="black")
    ax.add_collection(pc)
    
  plt.show()


if __name__ == '__main__':
  grid_dims = 2
  batch_size = 5
  occupancy_locs = [
      [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1],

      [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 1],

      [3, 0, 0, 0], [3, 0, 0, 1], [3, 0, 1, 0], [3, 0, 1, 1],
      [3, 1, 0, 0], [3, 1, 0, 1], [3, 1, 1, 0], [3, 1, 1, 1],
  ]
  voxels = create_voxels(grid_dims, batch_size, occupancy_locs)
  verts, faces, verts_mask, faces_mask = cubify(voxels, 0.5)

  batch_to_view = 0
  visualize_mesh(verts[batch_to_view, :], 
                 faces[batch_to_view, :],
                 verts_mask[batch_to_view, :],
                 faces_mask[batch_to_view, :])
