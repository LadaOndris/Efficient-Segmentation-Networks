import torch


def calculate_iou(outputs, targets, num_classes):
    # Convert predictions and targets to one-hot encoding
    outputs_one_hot = torch.nn.functional.one_hot(outputs, num_classes=num_classes)
    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes)

    # Calculate intersection and union for each class
    intersection = torch.sum(outputs_one_hot & targets_one_hot, dim=(1, 2))
    union = torch.sum(outputs_one_hot | targets_one_hot, dim=(1, 2))

    # Calculate IoU for each class
    iou_per_class = intersection.float() / (union.float() + 1e-10)

    # Average IoU across all classes
    mean_iou = torch.mean(iou_per_class, dim=1)

    return mean_iou.cpu().numpy(), iou_per_class.cpu().numpy()
