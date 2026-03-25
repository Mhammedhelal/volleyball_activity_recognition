"""
tests package
-------------
Unit and integration tests for volleyball_activity_recognition.

Test organization:
  - test_dataset.py: VolleyballDataset and data loading
  - test_person_embedder.py: PersonEmbedder model component
  - test_subgroup_pooler.py: SubGroupPooler and subgroup utilities
  - test_frame_descriptor.py: FrameDescriptor temporal aggregation
  - test_hierarchical.py: Full HierarchicalGroupActivityModel integration
  - conftest.py: Shared fixtures (dimensions, sample tensors, model instances)

Fixtures defined in conftest.py:
  - device: CUDA availability detection
  - dims: Standard dimension dict (N, T, C, H, W, etc.)
  - crops: Synthetic [N, T, C, H, W] video frames
  - person_embeddings: Synthetic PersonEmbedder outputs [N, T, embed_dim]
  - frame_descriptor_input: Synthetic frame descriptor input Z
  - person_labels, group_labels: Random annotation labels
  - subgroup_indices: Precomputed pooling indices
  - person_embedder, subgroup_pooler, frame_descriptor: Model instances
  - hierarchical_model: Full model instance
  - backbone_builder: Parametrized CNN backbones
"""

__all__ = []
