# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Convert raw Pix3D dataset to TFRecord format.
Example usage:
    python create_pix3d_tf_record.py --logtostderr \
      --pix3d_dir="${TRAIN_IMAGE_DIR}" \
      --pix3d_json_file="${TRAIN_ANNOTATIONS_FILE}" \
      --output_file_prefix="${OUTPUT_DIR/FILE_PREFIX}" \
      --num_shards=100 \
"""

import json
import logging
import os
from typing import List, Tuple

import numpy as np
import scipy.io
import tensorflow as tf
from absl import app, flags
from pycocotools import mask

from official.vision.data import tfrecord_lib
from official.vision.data.tfrecord_lib import (
    image_info_to_feature_dict, write_tf_record_dataset)

flags.DEFINE_multi_string('pix3d_dir', '', 'Directory containing Pix3d.')
flags.DEFINE_string('output_file_prefix', '/tmp/train', 'Path to output file')
flags.DEFINE_integer('num_shards', 32, 'Number of shards for output file.')
flags.DEFINE_multi_string("pix3d_json_file", "pix3d.json",
                          "Json file containing all pix3d info")

FLAGS = flags.FLAGS

logger = tf.get_logger()
logger.setLevel(logging.INFO)

def coco_segmentation_to_mask_png(segmentation, height, width, is_crowd):
  """Encode a COCO mask segmentation as PNG string."""
  run_len_encoding = mask.frPyObjects(segmentation, height, width)
  binary_mask = mask.decode(run_len_encoding)
  if not is_crowd:
    binary_mask = np.amax(binary_mask, axis=2)

  return tfrecord_lib.encode_mask_as_png(binary_mask)

def convert_to_feature(value, value_type=None):
  """Converts the given python object to a tf.train.Feature.

  Args:
    value: int, float, bytes or a list of them.
    value_type: optional, if specified, forces the feature to be of the given
        type. Otherwise, type is inferred automatically.
    Returns:
      feature: A tf.train.Feature object.
  """

  if value_type is None:

    element = value[0] if isinstance(value, list) else value

    if isinstance(element, bytes):
      value_type = 'bytes'

    elif isinstance(element, (int, np.integer)):
      value_type = 'int64'

    elif isinstance(element, (float, np.floating)):
      value_type = 'float'

    elif isinstance(element, str):
      value_type = "str"

    elif isinstance(element, list):
      value_type = "2d"

    elif isinstance(element, np.ndarray):
      value_type = 'ndarray'

    elif element is None:
      value_type = "none"

    else:
      raise ValueError(
          'Cannot convert type {} to feature'.format(type(element)))

    if isinstance(value, list):
      value_type = value_type + '_list'

  if value_type == 'int64':
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  elif value_type == 'int64_list':
    value = np.asarray(value).astype(np.int64).reshape(-1)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

  elif value_type == 'float':
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

  elif value_type == 'float_list':
    value = np.asarray(value).astype(np.float32).reshape(-1)
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

  elif value_type == 'bytes':
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  elif value_type == 'bytes_list':
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

  elif value_type == "str":
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value.encode("utf-8")]))

  elif value_type == "2d_list":
    data = tf.convert_to_tensor(value)
    serialized_data = tf.io.serialize_tensor(data)

    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[serialized_data.numpy()]))

  elif value_type == "ndarray":
    data = tf.convert_to_tensor(value)
    serialized_data = tf.io.serialize_tensor(data)

    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[serialized_data.numpy()]))

  elif value_type == "none":
    return tf.train.Feature()

  else:
    raise ValueError(
        'Unknown value_type parameter - {}'.format(value_type))

def create_tf_example(image: dict):
  """Converts image and annotations to a tf.Example proto.
  Args:
    image: dict with keys:
      ['filename, 'height', 'width', 'iscrowd', 'segmentation', 'model',
        'category_id', 'K', 'bbox', 'trans_mat', 'area', 'image_id',
        'rot_mat', 'voxel', 'pix3d_dir'].
  Returns:
    example: The converted tf.Example.
    num_annotations_skipped: Number of (invalid) annotations that were
      ignored.
  Raises:
    ValueError: if the image is not able to be found. This indicates the file
    structure of the Pix3D folder is incorrect.
  """

  img_height = image['img_size'][0]
  img_width = image['img_size'][1]
  img_filename = image['img']
  img_id = image['image_id']
  pix3d_dir = image['pix3d_dir']
  mask_filename = image['mask']
  model = image['model']
  voxel_file = image['voxel']

  # Create image annotation
  with tf.io.gfile.GFile(os.path.join(pix3d_dir, img_filename), 'rb') as fid:
    encoded_img = fid.read()

  if 'jpg' in img_filename or 'jpeg' in img_filename:
    img_format = 'jpg'
  else:
    img_format = 'png'

  feature_dict = image_info_to_feature_dict(
      img_height, img_width, img_filename, img_id, encoded_img,
      img_format)

  # Create mask annotation (encoded bytes will be decoded in Pix3dDecoder)
  with tf.io.gfile.GFile(os.path.join(pix3d_dir, mask_filename), 'rb') as fid:
    encoded_mask = fid.read()

  feature_dict.update({'image/object/mask': convert_to_feature(encoded_mask)})

  # Create voxel and mesh annotations
  model_vertices, model_faces = parse_obj_file(os.path.join(pix3d_dir, model))
  feature_dict.update(
      {'model/vertices': convert_to_feature(model_vertices),
       'model/faces': convert_to_feature(model_faces),
      })

  voxel_indices, voxel_shape = parse_voxel_file(
      os.path.join(pix3d_dir, voxel_file))
  feature_dict.update(
      {'model/voxel_indices': convert_to_feature(voxel_indices),
       'model/voxel_shape': convert_to_feature(voxel_shape)})

  # Create camera annotations
  rot_mat = image['rot_mat']
  trans_mat = image['trans_mat']
  intrinstic_mat = []
  intrinstic_mat.append(image['focal_length'] * (img_width/32))
  intrinstic_mat.append(img_width/2)
  intrinstic_mat.append(img_height/2)

  feature_dict.update(
      {'camera/rot_mat': convert_to_feature(rot_mat),
       'camera/trans_mat': convert_to_feature(trans_mat),
       'camera/intrinstic_mat': convert_to_feature(intrinstic_mat)})

  # Create bounding box annotations
  xmin, ymin, xmax, ymax = image['bbox']
  xmin = float(xmin) / img_width
  ymin = float(ymin) / img_height
  xmax = float(xmax) / img_width
  ymax = float(ymax) / img_height

  ##img_class_id = image['category_id']
  is_crowd = 0

  feature_dict.update({
      'image/object/bbox/xmin': convert_to_feature([float(xmin)]),
      'image/object/bbox/ymin': convert_to_feature([float(ymin)]),
      'image/object/bbox/xmax': convert_to_feature([float(xmax)]),
      'image/object/bbox/ymax': convert_to_feature([float(ymax)]),
      'image/object/class/label': image['category'],
      'image/object/is_crowd': convert_to_feature([is_crowd])})

  example = tf.train.Example(
      features=tf.train.Features(feature=feature_dict))

  return example, 0

def parse_obj_file(file: str) -> Tuple[List[List[float]], List[List[int]]]:
  """
  Parses the vertex and face data from a .obj file.

  Args:
    file: String filepath to .obj file.

  Return:
    vertices: An ndarray containing the vertices of the object.
    faces: An ndarray containing the faces of object.
  """
  vertices = []
  faces = []

  obj_file = open(file, 'r')
  lines = obj_file.readlines()

  for line in lines:
    line_id = line[0:2]

    # Vertex data only consists of x, y, z coordinates
    # Example: v -0.251095661374 -0.368396054024 -0.157995216677
    if line_id == "v ":
      vertex_line = line[2:].split(" ")
      vertex = []
      for v in vertex_line:
        vertex.append(float(v))
      vertices.append(vertex)

    # Face data might contain information about the vertices, vertex normals,
    # and vertex textures. These are grouped by "/" characters. The first
    # element in the group is the vertex index
    # Example: f 6/8 5/7 4/6
    if line_id == "f ":
      face_line = line[2:].split(" ")
      face = []
      for f in face_line:
        face.append(int(f.split("/")[0]))
      faces.append(face)

  return np.array(vertices), np.array(faces)

def parse_voxel_file(file: str):
  """
  Loads a voxel from a .mat file.

  Args:
    file: String filepath to .mat file.

  Return:
    voxel_indices: An ndarray containing the indices of the occupied locations
      in the original voxel.
    voxel_shape: An ndarray containing the shape of the original voxel.
  """
  voxel = scipy.io.loadmat(file)['voxel']
  voxel = np.rot90(voxel, k=3, axes=(1, 2))
  voxel_indices = np.argwhere(voxel > 0)

  voxel_shape = np.shape(voxel)
  return voxel_indices, list(voxel_shape)

def generate_annotations(annotation_dict: dict, pix3d_dir: str) -> List:
  """Generator for Pix3D annotations.

  Args:
    annotation_dict: Dictionary containing the raw annotations from the
      Pix3D annotations json file.
    pix3d_dir: pix3d_dir: String, path to Pix3D download directory.

  Return:
    annotations: List containing the annotations for each image to write to the
      TFRecord.
  """

  #raw_annotations = annotation_dict['annotations']
  #images = annotation_dict['images']

  #image_info = {}
  #for image in images:
    #image_info[image['id']] = {'filename': image['file_name'],
                               #'height': image['height'],
                               #'width': image['width']}

  annotations = []
  
  for id, info in enumerate(annotation_dict):
    annotations.append(
      {'img': info['img'], 'category': info['category'], 'img_size': info['img_size'],
       '2d_keypoints': info['2d_keypoints'], 'mask': info['mask'], 'img_source': info['img_source'],
       'model': info['model'], 'model_raw': info['model_raw'], 'model_source': info['model_source'],
       '3d_keypoints': info['3d_keypoints'], 'voxel': info['voxel'], 'rot_mat': info['rot_mat'],
       'trans_mat': info['trans_mat'], 'focal_length': info['focal_length'], 'cam_position': info['cam_position'],
       'inplane_rotation': info['inplane_rotation'], 'truncated': info['truncated'], 'occluded': info['occluded'],
       'slightly_occluded': info['slightly_occluded'], 'bbox': info['bbox'], 'image_id': id, 'pix3d_dir': pix3d_dir})
    #annotations.append(
        #{'filename': info['filename'], 'height': info['height'],
         #'width': info['width'], 'iscrowd': annotation['iscrowd'],
         #'segmentation': annotation['segmentation'],
         #'model': annotation['model'],
         #'category_id': annotation['category_id'],
         #'K': annotation['K'], 'bbox': annotation['bbox'],
         #'trans_mat': annotation['trans_mat'], 'area': annotation['area'],
         #'image_id': annotation['image_id'], 'rot_mat': annotation['rot_mat'],
         #'voxel': annotation['voxel'], 'pix3d_dir': pix3d_dir})
    
  return annotations


def _create_tf_record_from_pix3d_dir(pix3d_dir: str,
                                     output_path: str,
                                     num_shards: int,
                                     pix3d_json_file: str):
  """Loads Pix3D json files and converts to tf.Record format.

  Args:
    pix3d_dir: String, path to Pix3D download directory.
    output_path: Path to output tf.Record files.
    num_shards: Number of output files to create.
    pix3d_json_file: Name of the Pix3D annotation file inside pix3d_dir.
  """

  logging.info('writing to output path: %s', output_path)

  annotation_dict = json.load(open(os.path.join(pix3d_dir, pix3d_json_file)))

  pix3d_annotations_iter = generate_annotations(
      annotation_dict=annotation_dict, pix3d_dir=pix3d_dir)

  num_skipped = write_tf_record_dataset(
      output_path, pix3d_annotations_iter, create_tf_example,
      num_shards, unpack_arguments=False, multiple_processes=0)

  logging.info('Finished writing, skipped %d annotations.', num_skipped)

def main(_):
  assert FLAGS.pix3d_dir, '`pix3d_dir` missing.'

  directory = os.path.dirname(FLAGS.output_file_prefix)
  if not tf.io.gfile.isdir(directory):
    tf.io.gfile.makedirs(directory)

  _create_tf_record_from_pix3d_dir(
      FLAGS.pix3d_dir[0], FLAGS.output_file_prefix, FLAGS.num_shards,
      FLAGS.pix3d_json_file[0])

if __name__ == '__main__':
  app.run(main)
