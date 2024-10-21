import os
import numpy as np
import glob
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary
import torch.optim as optim
from time import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

def train(model, opt, loss_fn, epochs, train_loader, val_loader):
    # Grab a batch of validation data
    X_test, Y_test = next(iter(val_loader))

    for epoch in range(epochs):
        
        print('* Epoch %d/%d' % (epoch + 1, epochs))

        avg_loss = 0
        model.train()  # Set model to training mode

        for X_batch, Y_batch in train_loader:
            # Move data to device
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # Zero the gradients
            opt.zero_grad()

            # Forward pass
            Y_pred = model(X_batch)
            #print("Y_pred shape:", Y_pred.shape)
            #print("Y_batch shape:", Y_batch.shape)

            # Calculate loss
            loss = loss_fn(Y_batch,Y_pred)  # Correct order: model output first
            loss.backward()  # Backward pass
            opt.step()  # Update weights

            # Accumulate average loss
            avg_loss += loss.item() / len(train_loader)

       # toc = time()  # End timing
        print(' - loss: %f' % avg_loss)

        # Evaluate on the validation set
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation for evaluation
            Y_hat = F.sigmoid(model(X_test.to(device))).detach().cpu()

        # Clear previous output and plot the results
        clear_output(wait=True)  # Only if using Jupyter Notebook
        for k in range(4):  # Display first 4 images
            plt.subplot(2, 6, k + 1)
            plt.imshow(np.rollaxis(X_test[k].cpu().numpy(), 0, 3), cmap='gray')  # Move X_test to CPU
            plt.title('Real')
            plt.axis('off')

            plt.subplot(2, 6, k + 7)
            plt.imshow(Y_hat[k, 0].cpu().numpy(), cmap='gray')  # Move Y_hat to CPU
            plt.title('Output')
            plt.axis('off')

        plt.suptitle('%d / %d - loss: %f' % (epoch + 1, epochs, avg_loss))
        plt.show()



import sys
import os
sys.path.append('/zhome/45/0/155089/Deeplearning_in_computer_vision/Segmentation_project/Asignments_DeepLearningForCV/') 
from Encoder_Decoder_PH2 import device
from Performance_Metrics import dice_coefficient, intersection_over_union, accuracy, sensitivity, specificity
import time 
import time 
def test(model, test_loader, loss_fn):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():  # Disable gradient calculation
        for X_test_batch, Y_test_batch in test_loader:
            X_test_batch = X_test_batch.to(device)
            Y_test_batch = Y_test_batch.to(device)

            Y_test_pred = model(X_test_batch)
            loss = loss_fn(Y_test_batch,Y_test_pred )  # Compute test loss
            test_loss += loss.item()  # Accumulate test loss

            all_y_true.append(Y_test_batch.cpu())
            all_y_pred.append(Y_test_pred.cpu())

    avg_test_loss = test_loss / len(test_loader)
    print('Test Loss: %f' % avg_test_loss)

    # Concatenate all predictions and ground truths
    all_y_true = torch.cat(all_y_true)
    all_y_pred = torch.cat(all_y_pred)

    # Calculate metrics
    dice = dice_coefficient(all_y_true, all_y_pred)
    iou = intersection_over_union(all_y_true, all_y_pred)
    acc = accuracy(all_y_true, all_y_pred)
    sens = sensitivity(all_y_true, all_y_pred)
    spec = specificity(all_y_true, all_y_pred)

    # Print metrics
    print(f'Dice: {dice:.4f}, IoU: {iou:.4f}, Accuracy: {acc:.4f}, Sensitivity: {sens:.4f}, Specificity: {spec:.4f}')

    # Pause for a moment to allow user to read metrics
    #time.sleep(5)  # Adjust time as needed, here it waits for 5 seconds

    # Visualization of results
    clear_output(wait=True)  # Clear previous output
    clear_output(wait=True)
    clear_output(wait=True)
    X_test_batch, Y_test_batch = next(iter(test_loader))
    Y_test_pred = F.sigmoid(model(X_test_batch.to(device))).detach().cpu()

    #Plot the first 4 images and their predictions
    for k in range(4):  # For example, visualize the first 4 elements
        plt.subplot(2, 6, k + 1)
        plt.imshow(np.rollaxis(X_test_batch[k].cpu().numpy(), 0, 3), cmap='gray')
        #plt.title('Real')
        plt.axis('off')
        plt.tight_layout()

        plt.subplot(2, 6, k + 7)
        plt.imshow(Y_test_pred[k, 0], cmap='gray')
        #plt.title('Output')
        plt.axis('off')
        plt.tight_layout()

    plt.suptitle('Test - Loss: %f' % avg_test_loss)
    plt.show()  # This will block execution until you close the plot
    print(f'Dice: {dice:.4f}, IoU: {iou:.4f}, Accuracy: {acc:.4f}, Sensitivity: {sens:.4f}, Specificity: {spec:.4f}')
    
    
    
    

def visualize_test_predictions(model, test_loader, device, num_images=4):
    model.eval()  # Set the model to evaluation mode
    X_test_batch, Y_test_batch = next(iter(test_loader))  # Fetch a batch of test data
    X_test_batch = X_test_batch.to(device)
    Y_test_batch = Y_test_batch.to(device)

    # Get predictions
    with torch.no_grad():
        Y_test_pred = model(X_test_batch)
        Y_test_pred = torch.sigmoid(Y_test_pred).detach().cpu()  # Apply sigmoid and move to CPU

    # Plotting
    fig, axs = plt.subplots(3, num_images, figsize=(16, 10))  # Larger figure for clearer view

    # Set a nicer colormap
    cmap = 'inferno'  # Try using 'plasma', 'magma', 'inferno', etc.
    
    for k in range(num_images):  # Visualize the specified number of images
        # Original images
        axs[0, k].imshow(np.rollaxis(X_test_batch[k].cpu().numpy(), 0, 3), cmap='gray')
        axs[0, k].axis('off')

        # Ground truth images
        axs[1, k].imshow(Y_test_batch[k].cpu().numpy().squeeze(), cmap='gray')  # Squeeze to remove the channel dimension
        axs[1, k].axis('off')

        # Predicted images with a colorful colormap
        img = axs[2, k].imshow(Y_test_pred[k, 0], cmap='gray')  # Apply chosen colormap
        axs[2, k].axis('off')

    # Add rotated labels outside the subplots using fig.text
    fig.text(0.04, 0.82, 'Original', fontsize=30, rotation=90, va='center', color = 'white')
    fig.text(0.04, 0.5, 'Ground Truth', fontsize=30, rotation=90, va='center' , color = 'white')
    fig.text(0.04, 0.18, 'Prediction', fontsize=30, rotation=90, va='center', color = 'white')


    # Adjust layout to avoid overlap
    plt.subplots_adjust(wspace=0.1, hspace=0.2)  # Adjust spacing between plots
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar on the right
    plt.show()
