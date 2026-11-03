"""Detector backends.

``classical`` runs today with numpy and no model file; ``onnx_yolo`` consumes
the YOLOv8n exported from module 3's fine-tuning run.  Both return the same
:class:`rover_perception.detector.Detection` objects.
"""
