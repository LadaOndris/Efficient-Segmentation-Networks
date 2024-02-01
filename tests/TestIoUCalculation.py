import unittest
import torch
from ..utils.metric.iou import calculate_iou


class TestIoUCalculation(unittest.TestCase):
    def test_iou_calculation(self):
        """
        The function should produce IoU per sample in the batch.
        """
        # Generate example predictions and targets (binary masks)
        num_classes = 2
        batch_size = 3
        height, width = 4, 4

        # Assuming binary masks for simplicity
        outputs = torch.randint(0, 2, (batch_size, height, width))
        targets = torch.randint(0, 2, (batch_size, height, width))

        # Calculate IoU
        mean_iou, iou_per_class = calculate_iou(outputs, targets, num_classes)

        # Values are returned per sample
        self.assertTupleEqual(mean_iou.shape, (batch_size,))
        self.assertTupleEqual(iou_per_class.shape, (batch_size, num_classes))

        # Ensure the mean IoU values are within the expected range (0 to 1)
        for iou_value in mean_iou:
            self.assertGreaterEqual(iou_value, 0.0)
            self.assertLessEqual(iou_value, 1.0)

        # Ensure the per-class IoU values are within the expected range (0 to 1)
        for iou_class in iou_per_class.flatten():
            self.assertGreaterEqual(iou_class, 0.0)
            self.assertLessEqual(iou_class, 1.0)
