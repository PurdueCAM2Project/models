"""MeshRCNN task definition."""

from official.core import base_task
from official.core import task_factory
from official.vision.beta.projects.mesh_rcnn.configs import mesh_rcnn as exp_cfg
from official.vision.beta.projects.mesh_rcnn.modeling import factory
from official.vision.beta.projects.mesh_rcnn.dataloaders import meshrcnn_input
import tensorflow as tf
from typing import Any, Optional, List, Tuple, Mapping

from official.vision.dataloaders import input_reader_factory
from official.vision.dataloaders import tf_example_decoder
from official.vision.dataloaders import tf_example_label_map_decoder
from official.common import dataset_fn as dataset_fn_lib

@task_factory.register_task_cls(exp_cfg.MeshRCNNTask)
class MeshRCNNTask(base_task.Task):
    """A single-replica view of training procedure.
        MeshRCNN task provides artifacts for training/evalution procedures, including
    loading/iterating over Datasets, initializing the model, calculating the loss,
    post-processing, and customized metrics with reduction.
    """
    def __init__(self, params, logging_dir: Optional[str] = None):
        super().__init__(params, logging_dir)
        return

    def build_model(self):
        """Build Mesh R-CNN model."""

        input_specs = tf.keras.layers.InputSpec(
            shape=[None] + self.task_config.model.input_size)

        l2_weight_decay = self.task_config.losses.l2_weight_decay # either T or F
        
        # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
        # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
        # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
        l2_regularizer = (tf.keras.regularizers.l2(
            l2_weight_decay / 2.0) if l2_weight_decay else None)

        model, losses = factory.build_mesh_rcnn(
            input_specs, model_base_cfg, l2_regularizer)

        if self.task_config.freeze_backbone:
            model.backbone.trainable = False

        return model

    def build_inputs(
        self, 
        params: exp_cfg.DataConfig, 
        input_context: Optional[tf.distribute.InputContext] = None,
        dataset_fn: Optional[dataset_fn_lib.PossibleDatasetType] = None
    ) -> tf.data.Dataset:
        """Build input dataset."""
        """Decoder Builder"""
        decoder_cfg = params.decoder.get()
        if params.decoder.type == 'simple_decoder':
            decoder = tf_example_decoder.TfExampleDecoder(
                include_mask=self._task_config.model.include_mask,
                regenerate_source_id=decoder_cfg.regenerate_source_id,
                mask_binarize_threshold=decoder_cfg.mask_binarize_threshold)
        elif params.decoder.type == 'label_map_decoder':
            decoder = tf_example_label_map_decoder.TfExampleDecoderLabelMap(
                label_map=decoder_cfg.label_map,
                include_mask=self._task_config.model.include_mask,
                regenerate_source_id=decoder_cfg.regenerate_source_id,
                mask_binarize_threshold=decoder_cfg.mask_binarize_threshold)
        else:
            raise ValueError('Unknown decoder type: {}!'.format(params.decoder.type))

        #Parser
        parser = meshrcnn_input.Parser(
            output_size=self.task_config.model.input_size[:2],
            min_level=self.task_config.model.min_level,
            max_level=self.task_config.model.max_level,
            num_scales=self.task_config.model.anchor.num_scales,
            aspect_ratios=self.task_config.model.anchor.aspect_ratios,
            anchor_size=self.task_config.model.anchor.anchor_size,
            dtype=params.dtype,
            rpn_match_threshold=0.7, 
            rpn_unmatched_threshold=0.3, 
            rpn_batch_size_per_im=256, 
            rpn_fg_fraction=0.5, 
            aug_rand_hflip=False, 
            aug_scale_min=1, 
            aug_scale_max=1, 
            skip_crowd_during_training=True, 
            max_num_instances=100, 
            max_num_verts=108416, 
            max_num_faces=126748, 
            max_num_voxels=2097152, 
            include_mask=True, 
            mask_crop_size=112
        )

        if not dataset_fn:
            dataset_fn = dataset_fn_lib.pick_dataset_fn(params.file_type)

        reader = input_reader_factory.input_reader_generator(
            params,
            dataset_fn=dataset_fn,
            decoder_fn=decoder.decode,
            parser_fn=parser.parse_fn(params.is_training))
        dataset = reader.read(input_context=input_context)
        return dataset
    
    def build_metrics(self, training=True):
        """Build metrics."""
        return

    def build_losses(self, outputs, labels, aux_losses=None):
        """Build Mesh RCNN losses."""
        return

    def initialize(self, model: tf.keras.Model):
        """Loading pretrained checkpoint."""
        return

    def train_step(self, inputs, model, optimizer, metrics=None):
        """Train Step.
        Forward step and backwards propagate the model.
        Args:
        inputs: a dictionary of input tensors.
        model: the model, forward pass definition.
        optimizer: the optimizer for this training step.
        metrics: a nested structure of metrics objects.
        Returns:
        A dictionary of logs.
        """
        return

    def validation_step(self, inputs, model, metrics=None):
        """Validatation step.
        Args:
        inputs: a dictionary of input tensors.
        model: the keras.Model.
        metrics: a nested structure of metrics objects.
        Returns:
        A dictionary of logs.
        """
        return
    
    def aggregate_logs(self, state=None, step_outputs=None):
        """Get Metric Results."""
        return

    def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
        """Reduce logs and remove unneeded items. Update with results."""
        return

    def create_optimizer(self,
                       optimizer_config: OptimizationConfig,
                       runtime_config: Optional[RuntimeConfig] = None):
        """Creates an TF optimizer from configurations.
        Args:
        optimizer_config: the parameters of the Optimization settings.
        runtime_config: the parameters of the runtime.
        Returns:
        A tf.optimizers.Optimizer object.
        """
        
        return
