import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def bce_total_variation(y_real, y_pred):
    sigmoid_pred = torch.sigmoid(y_pred)
    diff_i  = abs(sigmoid_pred[1:, :] - sigmoid_pred[:-1, :])
    diff_j  = abs(sigmoid_pred[:, 1:] - sigmoid_pred[:, :-1])
    
    return stable_bce_loss(y_real, y_pred) + 0.01*(diff_i.sum() + diff_j.sum())

def stable_bce_loss(y_real, y_pred):
    # Preventing overflow by using max(y_pred, 0)
    max_val = torch.clamp(y_pred, min=0)  # clamp to ensure y_pred is at least 0
    loss = max_val - y_real * y_pred + torch.log1p(torch.exp(-torch.abs(y_pred)))
    return torch.mean(loss)

# Replace your loss function with the new dice_loss
def dice_loss(y_real, y_pred):
    y_pred = torch.sigmoid(y_pred)
    intersection = y_pred * y_real
    numerator = torch.mean(2 * intersection + 1)
    denominator = torch.mean(y_pred + y_real) + 1
    loss = 1 - numerator / denominator
    return loss

def focal_loss_chatten(y_real, y_pred, alpha=1, gamma=2):
    sigmoid_pred = torch.sigmoid(y_pred)
    
    # Calculate the binary cross-entropy for each class
    bce_loss = y_real * torch.log(sigmoid_pred + 1e-8) + (1 - y_real) * torch.log(1 - sigmoid_pred + 1e-8)
    
    # Calculate the modulating factor (1 - p_t) ^ gamma
    p_t = y_real * sigmoid_pred + (1 - y_real) * (1 - sigmoid_pred)
    modulating_factor = (1 - p_t) ** gamma
    
    # Apply the modulating factor and the alpha balancing factor
    focal_loss = -alpha * modulating_factor * bce_loss
    
    return torch.mean(focal_loss)

def bce_loss(y_real, y_pred):
    return torch.mean(y_pred - y_real*y_pred + torch.log(1 + torch.exp(-y_pred)))

def focal_loss(y_real, y_pred):
    sigmoid = lambda x: 1/(1+torch.exp(-x))
    return -torch.sum((1-sigmoid(y_pred))**2*y_real*torch.log(sigmoid(y_pred)) + (1-y_real) * torch.log(1-sigmoid(y_pred)))

def bce_loss_(y_real,y_pred):
    
    return loss

class UNet2(nn.Module):
    def __init__(self):
        super(UNet2, self).__init__()
        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool0 = nn.Conv2d(64, 64,kernel_size=3,stride=2, padding=1)  # 128 -> 64
        self.enc_conv1 = nn.Conv2d(64, 128,kernel_size= 3, padding=1,)
        self.pool1 = nn.Conv2d(128, 128,kernel_size=3, stride=2,padding=1)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool2 = nn.Conv2d(256, 256,kernel_size=3, stride=2,padding=1)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool3 = nn.Conv2d(512, 512,kernel_size=3, stride=2,padding=1)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(512, 1024, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.ConvTranspose2d(1024,1024,2,stride = 2, padding = 0)
        self.upsample1 = nn.ConvTranspose2d(512,512,2,stride = 2, padding = 0) # 16 -> 32
        self.upsample2 = nn.ConvTranspose2d(256,256,2,stride = 2, padding = 0)  # 32 -> 64
        self.upsample3 = nn.ConvTranspose2d(128,128,2,stride = 2, padding = 0)  # 64 -> 128

        self.dec_conv0 = nn.Conv2d(1024 + 512, 512, 3, padding=1)
        self.dec_conv1 = nn.Conv2d(512 + 256, 256, 3, padding=1)
        self.dec_conv2 = nn.Conv2d(256 + 128, 128, 3, padding=1)
        self.dec_conv3 = nn.Conv2d(128 + 64, 64, 3, padding=1)

        # final output layer
        self.final_conv = nn.Conv2d(64, 1, 1)  # 1x1 convolution for binary segmentation

    def forward(self, x):
        # encoder
        e0 = F.relu(self.enc_conv0(x))
        e1 = F.relu(self.enc_conv1(self.pool0(e0)))
        e2 = F.relu(self.enc_conv2(self.pool1(e1)))
        e3 = F.relu(self.enc_conv3(self.pool2(e2)))

        # bottleneck
        b = F.relu(self.bottleneck_conv(self.pool3(e3)))

        # decoder
        d0 = F.relu(self.dec_conv0(torch.cat([self.upsample0(b), e3], dim=1)))
        d1 = F.relu(self.dec_conv1(torch.cat([self.upsample1(d0), e2], dim=1)))
        d2 = F.relu(self.dec_conv2(torch.cat([self.upsample2(d1), e1], dim=1)))
        d3 = F.relu(self.dec_conv3(torch.cat([self.upsample3(d2), e0], dim=1)))


        # final output layer (logits)
        output = self.final_conv(d3)

        return output
    
def calc_dice_scores(val_loader,model):    
    dice_scores = [] 
    for image,label in val_loader:
        Y_True = label.to(device)
        model.eval()  # testing mode
        
        with torch.no_grad():  # Disable gradient calculation for evaluation
            Y_hat = F.sigmoid(model(image.to(device)))  # Get probabilities

        # Threshold the predictions to get class labels
        Y_hat_labels = (Y_hat > 0.5).to(torch.int64)

        # Now we can compute the generalized dice score using class indices
        gds = generalized_dice_score(Y_hat_labels, Y_True, num_classes=2)
        if gds[0].item() < 0.1:

            # Create a 1x2 subplot for two images: Y_True and Y_hat
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # Plot the ground truth (Y_True)
            axes[0].imshow(Y_True.detach().cpu().numpy().squeeze(), cmap='gray')
            axes[0].set_title('Ground Truth')
            axes[0].axis('off')

            # Plot the predicted output (Y_hat)
            axes[1].imshow(Y_hat.detach().cpu().numpy().squeeze(), cmap='gray')
            axes[1].set_title('Predicted Output')
            axes[1].axis('off')
            
        dice_scores.append(gds[0].item())
    return dice_scores


def train(model, opt, loss_fn, epochs, train_loader, test_loader):
    X_test, Y_test = next(iter(test_loader))

    for epoch in range(epochs):
        tic = time()
        print('* Epoch %d/%d' % (epoch+1, epochs))

        avg_loss = 0
        model.train()  # train mode
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # set parameter gradients to zero
            opt.zero_grad()

            # forward
            Y_pred = model(X_batch)
            loss = loss_fn(Y_batch, Y_pred)  # forward-pass
            loss.backward()  # backward-pass
            opt.step()  # update weights

            # calculate metrics to show the user
            avg_loss += loss / len(train_loader)
        toc = time()
        print(' - loss: %f' % avg_loss)

        # show intermediate results
        model.eval()  # testing mode
        Y_hat = F.sigmoid(model(X_test.to(device))).detach().cpu()
        clear_output(wait=True)
        for k in range(6):
            plt.subplot(2, 6, k+1)
            plt.imshow(np.rollaxis(X_test[k].numpy(), 0, 3), cmap='gray')
            plt.title('Real')
            plt.axis('off')

            plt.subplot(2, 6, k+7)
            plt.imshow(Y_hat[k, 0], cmap='gray')
            plt.title('Output')
            plt.axis('off')
        plt.suptitle('%d / %d - loss: %f' % (epoch+1, epochs, avg_loss))
        plt.show()
