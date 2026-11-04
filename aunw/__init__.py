"""AuNW Analysis: length and width of gold nanowires in SEM images.

Modules (in the order of the workflow)
    config      all parameters as dataclasses
    io          loading TIFF/JPEG, normalisation, calibration from metadata
    synthetic   physics-based synthetic SEM images with exact labels
    dataset     dataset generation and the tf.data input pipeline
    model       residual U-Net, losses, metrics
    training    training / fine-tuning with live monitoring
    inference   tiled prediction, test-time augmentation, automatic scale
    measure     wire separation (skeleton graph) and length/width measurement
    stats       descriptive statistics, fits, edge correction, tests
    evaluation  accuracy on synthetic ground truth and real annotations
    plots       figures;  export: CSV/Excel/ZIP;  cli: command line
"""
__version__ = "1.0.0"
