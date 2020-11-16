import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class OSL_model(nn.Module):

    def __init__(self, channels=3):
        super(OSL_model, self).__init__()

        # IMG CONV
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1)
        self.pool_5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # CROP CONV
        self.c_conv1_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        self.c_conv1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.c_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c_conv2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.c_conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.c_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c_conv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)        
        self.c_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c_conv4_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)        
        self.c_pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c_conv5_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1)        
        self.c_pool_5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c_conv6_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, stride=1)       

        # CONV 1X1 (FILTERS ADJUSTMENT)         
        self.f_conv_16  = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1)
        self.f_conv_32  = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1)
        self.f_conv_64  = nn.Conv2d(in_channels=768,  out_channels=128, kernel_size=1, stride=1)       
        self.f_conv_128 = nn.Conv2d(in_channels=640,  out_channels=64,  kernel_size=1, stride=1)           

        # DECONV
        self.d_conv1_1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1)
        self.d_conv1_2 = nn.Conv2d(in_channels=512,  out_channels=512, kernel_size=3, stride=1)
        self.deconv1   = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=1)

        self.d_conv2_1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1)
        self.d_conv2_2 = nn.Conv2d(in_channels=512,  out_channels=512, kernel_size=3, stride=1)
        self.deconv2   = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=1)

        self.d_conv3_1 = nn.Conv2d(in_channels=768, out_channels=256, kernel_size=3, stride=1)
        self.d_conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.deconv3   = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=1)

        self.d_conv4_1 = nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, stride=1)
        self.d_conv4_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.deconv4   = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1)

        self.d_conv5_1 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, stride=1)
        self.d_conv5_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.deconv5   = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)


        # OUT
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1)       

        
    def forward(self, img, crop):
        # img conv
        img_x = img

        img_x = self.conv1_1(img_x)
        img_x = F.relu( img_x )
        img_x = self.conv1_2(img_x)
        img_x = F.relu( img_x )
        img_128 = self.pool_1(img_x) # 64 x 128 x 128

        img_x = self.conv2_1(img_128)
        img_x = F.relu( img_x )
        img_x = self.conv2_2(img_x)
        img_x = F.relu( img_x )
        img_64 = self.pool_2(img_x) # 128 x 64 x 64

        img_x = self.conv3_1(img_64)
        img_x = F.relu( img_x )
        img_x = self.conv3_2(img_x)
        img_x = F.relu( img_x )
        img_32 = self.pool_3(img_x) # 256 x 32 x 32

        img_x = self.conv4_1(img_32)
        img_x = F.relu( img_x )
        img_x = self.conv4_2(img_x)
        img_x = F.relu( img_x )
        img_16 = self.pool_4(img_x) # 512 x 16 x 16

        img_x = self.conv5_1(img_16)
        img_x = F.relu( img_x )
        img_x = self.conv5_2(img_x)
        img_x = F.relu( img_x )
        img_8 = self.pool_5(img_x) # 512 x 8 x 8

        # crop conv
        crop_x = crop

        crop_x = self.c_conv1_1(crop_x)
        crop_x = F.relu( crop_x )
        crop_x = self.c_conv1_2(crop_x)
        crop_x = F.relu( crop_x )
        crop_x = self.c_pool_1(crop_x)

        crop_x = self.c_conv2_1(crop_x)
        crop_x = F.relu( crop_x )
        crop_x = self.c_conv2_2(crop_x)
        crop_x = F.relu( crop_x )
        crop_x = self.c_pool_2(crop_x)

        crop_x = self.c_conv3_1(crop_x)        
        crop_x = F.relu( crop_x )
        crop_x = self.c_pool_3(crop_x)

        crop_x = self.c_conv4_1(crop_x)        
        crop_x = F.relu( crop_x )
        crop_x = self.c_pool_4(crop_x)

        crop_x = self.c_conv5_1(crop_x)        
        crop_x = F.relu( crop_x )
        crop_x = self.c_pool_5(crop_x)

        crop_x = self.c_conv6_1(crop_x)
        crop_x = F.relu( crop_x ) # 1 x 1 x 512

        # tiles
        tile_128 = crop_x.repeat( 128, 128, 1 )
        tile_64  = crop_x.repeat(  64,  64, 1 )
        tile_32  = crop_x.repeat(  32,  32, 1 )
        tile_16  = crop_x.repeat(  16,  16, 1 )
        tile_8   = crop_x.repeat(   8,   8, 1 )

        # depth-wise concatenation
        dw_128 = torch.cat( (img_128, tile_128), dim=0 ) # filters 64  + 512
        dw_64  = torch.cat( (img_64,  tile_64),  dim=0 ) # filters 128 + 512
        dw_32  = torch.cat( (img_32,  tile_32),  dim=0 ) # filters 256 + 512
        dw_16  = torch.cat( (img_16,  tile_16),  dim=0 ) # filters 512 + 512
        dw_8   = torch.cat( (img_8,   tile_8),   dim=0 ) # filters 512 + 512        



        # decoder (deconv) 
        
        # conv 1x1 filters adjustment
        fa_dw_8   = dw_8
        fa_dw_16  = self.f_conv_16(  dw_16  )
        fa_dw_32  = self.f_conv_32(  dw_32  )
        fa_dw_64  = self.f_conv_64(  dw_64  )
        fa_dw_128 = self.f_conv_128( dw_128 )        
        

        # forward deconv
        d_dw_8 = fa_dw_8
        x_dw_8 = self.d_conv1_1( d_dw_8 )
        x_dw_8 = F.relu( x_dw_8 )
        x_dw_8 = self.d_conv1_2( x_dw_8 )
        x_dw_8 = F.relu( x_dw_8 )
        x_dw_16 = self.deconv1( x_dw_8 ) # 16 x 16 x 512

        
        d_dw_16 = torch.cat( (x_dw_16, fa_dw_16), dim=0 )
        x_dw_16 = self.d_conv2_1( d_dw_16 )
        x_dw_16 = F.relu( x_dw_16 )
        x_dw_16 = self.d_conv2_2( x_dw_16 )
        x_dw_16 = F.relu( x_dw_16 )
        x_dw_32 = self.deconv2( x_dw_16 )


        d_dw_32 = torch.cat( (x_dw_32, fa_dw_32), dim=0 )
        x_dw_32 = self.d_conv3_1( d_dw_32 )
        x_dw_32 = F.relu( d_dw_32 )
        x_dw_32 = self.d_conv3_2( d_dw_32 )
        x_dw_32 = F.relu( d_dw_32 )
        x_dw_64 = self.deconv3( d_dw_32 )


        d_dw_64 = torch.cat( (x_dw_64, fa_dw_64), dim=0 )
        x_dw_64 = self.d_conv4_1( d_dw_64 )
        x_dw_64 = F.relu( d_dw_64 )
        x_dw_64 = self.d_conv4_2( d_dw_64 )
        x_dw_64 = F.relu( d_dw_64 )
        x_dw_128 = self.deconv4( d_dw_64 )


        d_dw_128 = torch.cat( (x_dw_128, fa_dw_128), dim=0 )
        x_dw_128 = self.d_conv4_1( d_dw_128 )
        x_dw_128 = F.relu( d_dw_128 )
        x_dw_128 = self.d_conv4_2( d_dw_128 )
        x_dw_128 = F.relu( d_dw_128 )
        x_dw_256 = self.deconv4( d_dw_128 )

        return x_dw_256

        
