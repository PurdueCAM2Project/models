# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Contains factory functions for Mesh R-CNN networks."""

from typing import Optional

import tensorflow as tf  # type: ignore

from official.vision.beta.projects.mesh_rcnn.configs.mesh_rcnn import VoxelHead
from official.vision.beta.projects.mesh_rcnn.modeling.heads import voxel_head


def build_voxel_head(head_config: VoxelHead,
                      kernel_regularizer:
                      Optional[tf.keras.regularizers.Regularizer],
                      bias_regularizer:
                      Optional[tf.keras.regularizers.Regularizer],
                      activity_regularizer:
                      Optional[tf.keras.regularizers.Regularizer]
                    ) -> voxel_head.VoxelHead:
  """Builds Voxel Branch Prediction Head.
  Args:
    head_config: Dataclass parameterization instance for voxel head.
    kernel_regularizer: Convolutional layer weight regularizer object.
    bias_regularizer: Convolutional layer bias regularizer object.
    activity_regularizer: Convolutional layer activation regularizer object.
  Returns:
    Voxel head layer instance.
  """
  return voxel_head.VoxelHead(
    voxel_depth=head_config.voxel_depth,
    conv_dim=head_config.conv_dim,
    num_conv=head_config.num_conv,
    use_group_norm=head_config.use_group_norm,
    predict_classes=head_config.predict_classes,
    bilinearly_upscale_input=head_config.bilinearly_upscale_input,
    class_based_voxel=head_config.class_based_voxel,
    num_classes=head_config.num_classes,
    kernel_regularizer=kernel_regularizer,
    bias_regularizer=bias_regularizer,
    activity_regularizer=activity_regularizer,
  )
