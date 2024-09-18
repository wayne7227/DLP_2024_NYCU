import torch
import torch.nn as nn
import torch.nn.functional as F


def construct_residual_layer(block, in_features, out_features, num_units, stride_length=1):
    downsample_layer = None
    if in_features != out_features * block.Exp or stride_length != 1:
        downsample_layer = nn.Sequential(
            nn.Conv2d(in_features, out_features * block.Exp, kernel_size=1, stride=stride_length, bias=False),
            nn.InstanceNorm2d(out_features * block.Exp),  # Changed to InstanceNorm2d
        )

    initial_block = block(in_features, out_features, stride_length, downsample_layer)
    in_features = out_features * block.Exp
    subsequent_blocks = [block(in_features, out_features) for _ in range(1, num_units)]
    layers = [initial_block] + subsequent_blocks

    return nn.Sequential(*layers)


class ResidualUnit(nn.Module):
    Exp = 1

    def __init__(self, input_dim, output_dim, stride_length=1, downsample_layer=None):
        super(ResidualUnit, self).__init__()

        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride_length, padding=1, bias=False)
        self.batch_norm1 = nn.InstanceNorm2d(output_dim)  # Changed to InstanceNorm2d
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)  # Changed to LeakyReLU

        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, bias=False)
        self.batch_norm2 = nn.InstanceNorm2d(output_dim)  # Changed to InstanceNorm2d

        self.downsample_layer = downsample_layer
        self.stride_length = stride_length

    def forward(self, x):
        Id = x

        output_1 = self.activation(self.batch_norm1(self.conv1(x)))
        output_2 = self.batch_norm2(self.conv2(output_1))

        if self.downsample_layer is not None:
            Id = self.downsample_layer(x)

        output_2 += Id
        output_2 = self.activation(output_2)

        return output_2


class UpConvBlock(nn.Module):
    def __init__(self, input_dim, bridge_dim, output_dim):
        super(UpConvBlock, self).__init__()
        self.up_conv = nn.ConvTranspose2d(input_dim, bridge_dim, kernel_size=2, stride=2)

        self.concat_conv1 = nn.Conv2d(bridge_dim + output_dim, output_dim, kernel_size=3, padding=1)
        self.batch_norm1 = nn.InstanceNorm2d(output_dim)  # Changed to InstanceNorm2d
        self.activation1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)  # Changed to LeakyReLU

        self.concat_conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)
        self.batch_norm2 = nn.InstanceNorm2d(output_dim)  # Changed to InstanceNorm2d
        self.activation2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)  # Changed to LeakyReLU

    def forward(self, x, skip_connection):
        x = self.up_conv(x)

        diff_y = skip_connection.size()[2] - x.size()[2]
        diff_x = skip_connection.size()[3] - x.size()[3]
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat((x, skip_connection), dim=1)

        x = self.concat_conv1(x)
        x = self.batch_norm1(x)
        x = self.activation1(x)

        x = self.concat_conv2(x)
        x = self.batch_norm2(x)
        x = self.activation2(x)

        return x


class ResNet34UNet(nn.Module):
    def __init__(self, out_channels=1):
        super(ResNet34UNet, self).__init__()

        self.initial_block_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.initial_block_bn1 = nn.InstanceNorm2d(64)  # Changed to InstanceNorm2d
        self.initial_block_relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)  # Changed to LeakyReLU
        self.initial_block_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.initial_block_bn2 = nn.InstanceNorm2d(64)  # Changed to InstanceNorm2d
        self.initial_block_relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)  # Changed to LeakyReLU

        self.encoder1 = construct_residual_layer(ResidualUnit, 64, 64, 3)
        self.encoder2 = construct_residual_layer(ResidualUnit, 64, 128, 4, stride_length=2)
        self.encoder3 = construct_residual_layer(ResidualUnit, 128, 256, 6, stride_length=2)
        self.encoder4 = construct_residual_layer(ResidualUnit, 256, 512, 3, stride_length=2)

        self.decoder1 = UpConvBlock(512, 256, 256)
        self.decoder2 = UpConvBlock(256, 128, 128)
        self.decoder3 = UpConvBlock(128, 64, 64)
        self.decoder4 = UpConvBlock(64, 64, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.final_activation = nn.Sigmoid()  # Changed to Sigmoid

    def forward(self, x):
        x = self.initial_block_conv1(x)
        x = self.initial_block_bn1(x)
        x = self.initial_block_relu1(x)
        x = self.initial_block_conv2(x)
        x = self.initial_block_bn2(x)
        enc1 = self.initial_block_relu2(x)

        enc2 = self.encoder1(enc1)
        enc3 = self.encoder2(enc2)
        enc4 = self.encoder3(enc3)
        bottleneck = self.encoder4(enc4)

        dec1 = self.decoder1(bottleneck, enc4)
        dec2 = self.decoder2(dec1, enc3)
        dec3 = self.decoder3(dec2, enc2)
        dec4 = self.decoder4(dec3, enc1)

        output = self.final_conv(dec4)
        output_3 = self.final_activation(output)
        return output_3

# Testing the model
if __name__ == "__main__":
    model = ResNet34UNet(out_channels=1)
    x = torch.randn((1, 3, 256, 256))  # Input (batch_size, channels, height, width)
    output = model(x)
    print(output.shape)
