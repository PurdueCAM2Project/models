"""Mesh Visualization"""

import matplotlib.pyplot as plt
import tensorflow as tf
import open3d as o3d
from mpl_toolkits.mplot3d import art3d

from official.vision.beta.projects.mesh_rcnn.ops.cubify import cubify, initialize_mesh


def create_voxels(grid_dims, batch_size, occupancy_locs):
    ones = tf.ones(shape=[len(occupancy_locs)])
    voxels = tf.scatter_nd(
        indices=tf.convert_to_tensor(occupancy_locs, tf.int32),
        updates=ones,
        shape=[batch_size, grid_dims, grid_dims, grid_dims])

    return voxels


def visualize_mesh(verts, faces, verts_mask, faces_mask, smoothing = False):

    """Visualize meshes using Open3D
    Args:
        verts: A "Tensor" of shape of [V,3], where V is the number of vertices
        faces: A "Tensor" of shape of [F,3], where each face is composed of 3 integers,
        which map to the points in the vertex tensor
        verts_mask: A "Tensor" of shape of [V,1], which consists of 0 or 1 to determine
        which vertices are valid
        face_mask: A "Tensor" of shape of [F,1], which consists of 0 or 1 to determine
        which faces are valid
        smoothing: A 'bool', whether to make meshes smooth
    """


    triangle_mesh = o3d.geometry.TriangleMesh()
    verts_numpy = verts.numpy()
    faces_numpy = faces.numpy()

    # Convert numpy array to Open3D format
    triangle_mesh.vertices = o3d.utility.Vector3dVector(verts_numpy)
    triangle_mesh.triangles = o3d.utility.Vector3iVector(faces_numpy)

    # If smoothing is True, smooth the meshes
    if smoothing:
        mesh_out = triangle_mesh.filter_smooth_simple(number_of_iterations = 5)
    else:
      mesh_out = triangle_mesh
    # Compute triangle normals
    mesh_out.compute_vertex_normals()

    # Draw the meshes and visualize
    o3d.visualization.draw_geometries([mesh_out])


if __name__ == '__main__':
    _grid_dims = 2
    _batch_size = 5
    _occupancy_locs = [
        [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1],

        [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 1],

        [3, 0, 0, 0], [3, 0, 0, 1], [3, 0, 1, 0], [3, 0, 1, 1],
        [3, 1, 0, 0], [3, 1, 0, 1], [3, 1, 1, 0], [3, 1, 1, 1],
    ]
    voxels = create_voxels(_grid_dims, _batch_size, _occupancy_locs)

    verts, faces = initialize_mesh(3, 'topleft')
    verts_mask = tf.constant(value = 1, shape = (verts.shape[0], 1))
    faces_mask = tf.constant(value = 1, shape = (faces.shape[0], 1))

    #_verts = mesh['verts']
    #_faces = mesh['faces']
    #_verts_mask = mesh['verts_mask']
    #_faces_mask = mesh['faces_mask']

    batch_to_view = 0
    visualize_mesh(verts,faces,verts_mask,faces_mask)

    plt.show()