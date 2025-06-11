import torch
import torch.nn.functional as F

def stratified_cv_loss(outputs, labels, stratum_ids, num_strata, lambda_reg=1.0, epsilon=1e-4):
    """
    Compute BCE loss + coefficient of variation regularization over strata.

    Args:
        outputs (Tensor): shape [B, C]
        labels (Tensor): shape [B, C]
        stratum_ids (Tensor): shape [B], with integer values in [0, num_strata-1]
        num_strata (int): number of unique strata
        lambda_reg (float): regularization strength
        epsilon (float): small value to avoid division by zero

    Returns:
        loss (Tensor): scalar loss value
    """
    # Standard BCE loss (averaged over batch)
    bce_loss = F.binary_cross_entropy_with_logits(outputs, labels, reduction='mean')

    # Compute per-stratum mean loss
    per_sample_loss = F.binary_cross_entropy_with_logits(outputs, labels, reduction='none')  # [B, C]
    per_sample_loss = per_sample_loss.mean(dim=1)  # [B]

    stratum_losses = []
    for i in range(num_strata):
        mask = (stratum_ids == i)
        if mask.sum() == 0:
            continue  # skip empty strata
        loss_i = per_sample_loss[mask].mean()
        stratum_losses.append(loss_i)

    stratum_losses[0] = torch.tensor(0.5)
    stratum_losses[1] = torch.tensor(0.2)
    stratum_losses[2] = torch.tensor(0.5)
    stratum_losses[3] = torch.tensor(0.5)
    stratum_losses[4] = torch.tensor(0.5)

    print(stratum_losses)

    stratum_losses = torch.stack(stratum_losses)  # [K]
    mean_loss = stratum_losses.mean()
    std_loss = stratum_losses.std(unbiased=False)

    # Coefficient of variation regularization
    cv = std_loss / (mean_loss + epsilon)

    print(f"BCE loss: {bce_loss.item()}")
    print(f"Regularization: {lambda_reg * cv.item()}")

    total_loss = bce_loss + lambda_reg * cv
    return total_loss


# Example inputs
B, C, K = 64, 106, 5  # batch size, classes, strata
outputs = torch.randn(B, C)
labels = torch.randint(0, 2, (B, C)).float()
stratum_ids = torch.randint(0, K, (B,))

loss = stratified_cv_loss(outputs, labels, stratum_ids, num_strata=K, lambda_reg=0.5)
