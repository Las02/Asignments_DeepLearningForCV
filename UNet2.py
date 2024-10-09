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
