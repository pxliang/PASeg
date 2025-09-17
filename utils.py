import matplotlib.pyplot as plt
import numpy as np
import cv2

from matplotlib.patches import Rectangle

from sklearn.metrics import confusion_matrix

def vis_img(image, pred_mask, true_mask, text_prompt, box, save_path):

    # (4) Plotting
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
    axs[0].imshow(image)
    axs[0].set_title(text_prompt)
    axs[0].axis("off")

    axs[1].imshow(pred_mask, cmap="gray")
    axs[1].set_title("Predicted Mask")
    axs[1].axis("off")

    if box is not None:
        x0, y0 = int(box[0]), int(box[1])
        w, h = int(box[2] - box[0]), int(box[3] - box[1])
        axs[0].add_patch(Rectangle((x0, y0), w, h, edgecolor="red", facecolor=(0, 0, 0, 0), lw=2))

    prob_map_8bit = (pred_mask * 255).astype(np.uint8)

    # Apply Otsu's threshold
    _, binary_mask = cv2.threshold(prob_map_8bit, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    axs[2].imshow(binary_mask, cmap="gray")
    axs[2].set_title("Binary Mask")
    axs[2].axis("off")

    unique_idx = np.unique(true_mask)
    unique_idx = unique_idx[unique_idx > 0]

    axs[3].imshow(true_mask, cmap='gray', vmin=0, vmax=1)
    axs[3].set_title("Ground Truth Mask "+', '.join(str(x) for x in unique_idx))
    axs[3].axis("off")


    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    assert len(unique_idx) == 1

    return binary_mask



def evaluate_segmentation(pred_mask, true_mask, eps=1e-7):
    """
    Evaluates binary segmentation mask performance using NumPy arrays.

    Args:
        pred_mask (np.ndarray): Predicted binary mask or probability map, shape [H, W] or [1, H, W]
        true_mask (np.ndarray): Ground truth binary mask, same shape as pred_mask
        eps (float): Small epsilon to avoid division by zero

    Returns:
        tuple: (dice, accuracy, precision, recall)
    """

    # Binarize masks (threshold at 0)
    pred_bin = (pred_mask > 0).astype(np.float32).reshape(-1)
    true_bin = (true_mask > 0).astype(np.float32).reshape(-1)

    intersection = np.sum(pred_bin * true_bin)
    union = np.sum(pred_bin) + np.sum(true_bin)

    dice = (2.0 * intersection + eps) / (union + eps)
    acc = np.mean(pred_bin == true_bin)

    tp = np.sum(pred_bin * true_bin)
    fp = np.sum(pred_bin * (1.0 - true_bin))
    fn = np.sum((1.0 - pred_bin) * true_bin)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    return dice, acc, precision, recall



def compute_multi_class_metrics(gt, pred):
   
    metrics = {}
    epsilon = 1e-7

    num_classes = max(int(np.amax(gt)), int(np.amax(pred))) + 1

    cm = confusion_matrix(gt.flatten(), pred.flatten(), labels=np.arange(num_classes))

    # Per-class metrics
    IoU = []
    Dice = []
    Precision = []
    Recall = []

    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        # Skip classes not present in the GT
        if (TP + FN) == 0:
            continue  # Class i not present in GT, skip it

        iou = TP / (TP + FP + FN + epsilon)
        dice = 2 * TP / (2 * TP + FP + FN + epsilon)
        prec = TP / (TP + FP + epsilon)
        rec = TP / (TP + FN + epsilon)

        IoU.append(iou)
        Dice.append(dice)
        Precision.append(prec)
        Recall.append(rec)

   

    metrics["IoU"] = np.mean(IoU)
    metrics["Dice"] = np.mean(Dice)
    metrics["Precision"] = np.mean(Precision)
    metrics["Recall"] = np.mean(Recall)

    return metrics