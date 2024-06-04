import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128' # set max_split_size_mb to reduce reserved unused memory
import torch
from torch.utils.data import DataLoader, random_split
from torch import optim
from sklearn.metrics import f1_score, jaccard_score
import numpy as np
from sklearn.decomposition import PCA # For PCA analysis
# Class imports from seperate files #
from dataclass_corrosion import Dataclass
from unet_corrosion import UNet
import lovasz_losses as L

metrics_list = []
def log_training(epoch, train_seg_loss, train_domain_loss, train_f1, train_iou, val_seg_loss,
                 val_domain_loss, val_f1, val_iou, train_domain_acc,
                 val_domain_acc, log_file="Metrics.txt"):
    """
    Logs training and validation metrics for each epoch, including hyperparameters.

    Parameters:
    - epoch (int): Current epoch number.
    - train_seg_loss (float): Average segmentation loss on the training set.
    - train_f1 (float): Average F1 score for segmentation on the training set.
    - train_iou (float): Average Intersection over Union (IoU) for segmentation on the training set.
    - val_seg_loss (float): Average segmentation loss on the validation set.
    - val_f1 (float): Average F1 score for segmentation on the validation set.
    - val_iou (float): Average IoU for segmentation on the validation set.
    - log_file (str): Path to the log file.
    """
    metrics_str = f"Epoch: {epoch}, " \
                  f"Training Segmentation Loss: {train_seg_loss:.4f}, " \
                  f"Training Segmentation F1: {train_f1:.4f}, Training Segmentation IoU: {train_iou:.4f}, " \
                  f"Validation Segmentation Loss: {val_seg_loss:.4f} " \
                  f"Validation Segmentation F1: {val_f1:.4f}, Validation Segmentation IoU: {val_iou:.4f}\n"

    # Append metrics to the list
    metrics_list.append(metrics_str)

    # Write the accumulated metrics to the file at specified intervals and at the end of training
    if epoch % 10 == 0 or epoch == num_epochs:
        with open(log_file, "a") as file:
            for metric in metrics_list:
                file.write(metric)
            metrics_list.clear()

# Enable TF32 for improved performance on compatible GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set random seed for reproducibility
torch.manual_seed(42)

# Step 1: Setup Dataset
dataset = Dataclass(
    image_dir='/zhome/10/0/181468/corrosion/inspectrone/inspectrone_train',
    annotation_file='/zhome/10/0/181468/corrosion/inspectrone/annotations/inspectrone_train.json')

# Define the sizes for training, validation, and test sets
train_size = int(0.7 * len(dataset))  # 60% for training
val_size = int(0.2 * len(dataset))    # 20% for validation
test_size = len(dataset) - train_size - val_size  # Remaining 20% for test

# Split dataset into training, validation, and test sets
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders for each set
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Step 2: Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_class=7).to(device) # Total of 8 classes = {Red, Pink, Purple, Yellow, Blue, Cyan, Green, Black}

# Step 3: Loss Function & Optimizer
segmentation_criterion = L.lovasz_softmax # Try Lovasz
domain_criterion = torch.nn.BCEWithLogitsLoss() # Try different criterions
learning_rate = 0.00031055335014095966
weight_decay = 0.052909730367861826
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


# Initialize lists to store metrics
train_losses = []
val_losses = []
train_f1_scores = []
train_ious = []
val_f1_scores = []
val_ious = []

# Step 4: Training Loop
num_epochs = 10  # ADJUSTABLE
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    total_seg_loss_accum = 0.0
    total_domain_loss_accum = 0.0
    train_f1_accum = 0
    train_iou_accum = 0
    train_domain_acc_accum = 0  # Initialize accumulator for domain accuracy
    
    for batch_idx, (images, segmentation_labels) in enumerate(train_loader):
        images = images.to(device)
        segmentation_labels = segmentation_labels.to(device)
        optimizer.zero_grad()  # zero the parameter gradients
        # Forward pass through the model    
        segmentation_pred, _ = model(images)
        # Segmentation and domain classification loss
        seg_loss = segmentation_criterion(segmentation_pred, segmentation_labels)
        # Combine the losses
        total_loss = seg_loss

        # Backward pass and optimize
        total_loss.backward()

        # Monitor gradients
        #max_grad = 0.0
        #for param in model.parameters():
        #    if param.grad is not None:
        #        # Calculate the maximum gradient among all model parameters
       #         max_grad = max(max_grad, param.grad.abs().max().item())
        
        #print(f"Max gradient after backward pass in epoch {epoch}: {max_grad}")
        # Gradient clipping to avoid exploding gradient
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0, norm_type=2.0) # Experiment with max_norm, norm_type is the default L2.

        optimizer.step()

        # Accumulate losses for logging
        total_seg_loss_accum += seg_loss.item()

        # Convert model outputs to discrete predictions
        predictions = torch.argmax(segmentation_pred, dim=1).detach().cpu().numpy()
        segmentation_labels_numpy = segmentation_labels.squeeze(1).long().detach().cpu().numpy()  # Convert labels to numpy
        # Flatten the arrays to 1D for metric calculations
        labels_flat = segmentation_labels_numpy.flatten()
        predictions_flat = predictions.flatten()
        # Calculate and accumulate F1-score and IoU
        f1 = f1_score(labels_flat, predictions_flat, average='macro')
        iou = jaccard_score(labels_flat, predictions_flat, average='macro')
        train_f1_accum += f1
        train_iou_accum += iou
        
        # Batch losses for printing
        average_seg_loss_so_far = total_seg_loss_accum / (batch_idx + 1)

        if batch_idx % 10 == 0:  # print every 10 batches
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                f"Total Loss: {total_loss.item()}, "
                f"Avg Seg Loss So Far: {average_seg_loss_so_far}, ")
            
            
    # Averaging metrics over the epoch
    average_train_seg_loss = total_seg_loss_accum / len(train_loader)
    average_train_f1 = train_f1_accum / len(train_loader)
    average_train_iou = train_iou_accum / len(train_loader)
    # Logging epoch-level metrics
    print(f"Epoch {epoch+1}/{num_epochs}, Avg Seg Loss: {average_train_seg_loss}, Avg F1: {average_train_f1}, Avg IoU: {average_train_iou}")


    # VALIDATION PHASE
    model.eval()  # Set model to evaluation mode
    val_f1_accum = 0
    val_iou_accum = 0
    val_domain_acc_accum = 0
    val_seg_loss_accum = 0  # Accumulator for segmentation loss during validation
    val_domain_loss_accum = 0  # Accumulator for domain loss during validation
    valid_seg_batches = 0  # Counter for batches with valid segmentation labels

    with torch.no_grad():
        for images, segmentation_labels in val_loader:
            images = images.to(device)
            segmentation_labels = segmentation_labels.to(device)

            # Forward pass
            segmentation_pred, _ = model(images)

            # Compute validation losses
            val_seg_loss = segmentation_criterion(segmentation_pred, segmentation_labels)
            val_seg_loss_accum += val_seg_loss.item()
            
            # Convert model outputs to discrete predictions for metric calculations
            predictions = torch.argmax(segmentation_pred, dim=1).detach().cpu().numpy()
            true_labels = segmentation_labels.squeeze(1).detach().cpu().numpy() # Convert to numpy

            # Flatten arrays for F1-score and IoU calculations
            predictions_flat = predictions.flatten()
            true_labels_flat = true_labels.flatten()
            # Calculate and accumulate F1-score and IoU
            f1 = f1_score(true_labels_flat, predictions_flat, average='macro')
            iou = jaccard_score(true_labels_flat, predictions_flat, average='macro')
            val_f1_accum += f1
            val_iou_accum += iou

            valid_seg_batches += 1  # Increment counter
        
        # Calculate average validation loss, F1-score, mIoU and domain accuracy for the epoch
        average_val_seg_loss = val_seg_loss_accum / valid_seg_batches if valid_seg_batches > 0 else 0
        average_val_f1 = val_f1_accum / valid_seg_batches if valid_seg_batches > 0 else 0
        average_val_iou = val_iou_accum / valid_seg_batches if valid_seg_batches > 0 else 0
        # Logging validation metrics
        print(f'Validation\nEpoch {epoch+1}: Seg Loss: {average_val_seg_loss}, F1-score: {average_val_f1}, mIoU: {average_val_iou}') # ALL ARE AVERAGES

        #log_training(epoch + 1, total_loss, total_segmentation_loss, train_f1_scores[-1], train_ious[-1], val_f1_scores[-1], val_ious[-1]) # Log the metrics
        log_training(epoch + 1, average_train_seg_loss, average_train_f1, average_train_iou, average_val_seg_loss, average_val_f1, average_val_iou)
# Save the model after training
torch.save(model.state_dict(), 'new_unet_model.pth')
print("Finished Training and saved the model.")