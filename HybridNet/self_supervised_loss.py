import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class self_supervised_loss (nn.modules.Module):
    def __init__(self, n=1, SSIM_w=0.85, disp_gradient_w=1.0, lr_w=1.0):
        super(self_supervised_loss, self).__init__()
        self.SSIM_w = SSIM_w
        self.disp_gradient_w = disp_gradient_w
        self.lr_w = lr_w
        self.n = n

    def scale_pyramid(self, img, scale):

        scaled_imgs = [img]
        # pyramid
        for i in range(1, scale):
            scaled_imgs.append(img)
            #scaled_imgs.append(scaled_imgs[-1][:, :, ::2, ::2])

        return scaled_imgs

    def scale_disp(self, disp, scale):

        scaled_disp = [disp[-1]]

        for i in (2, scale):
            print(scale)
            print(i)
            down_disp = F.upsample(disp[-i], (int(disp[-i].shape[2]/2), int(disp[-i].shape[3]/2)), mode='bilinear')/2
            scaled_disp.append(down_disp)

        return scaled_disp


    def gradient_x(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx

    def gradient_y(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy


    def apply_disparity(slef, img, disp, cuda=True):
        '''
        img.shape = b, c, h, w
        disp.shape = b, h, w
        '''
        b, c, h, w = img.shape

        disp = disp.squeeze(1)

        if cuda == True:
            right_coor_x = (torch.arange(start=0, end=w, out=torch.cuda.FloatTensor())).repeat(b, h, 1)
            right_coor_y = (torch.arange(start=0, end=h, out=torch.cuda.FloatTensor())).repeat(b, w, 1).transpose(1, 2)
        else:
            right_coor_x = (torch.arange(start=0, end=w, out=torch.FloatTensor())).repeat(b, h, 1)
            right_coor_y = (torch.arange(start=0, end=h, out=torch.FloatTensor())).repeat(b, w, 1).transpose(1, 2)

        left_coor_x1 = right_coor_x + disp
        left_coor_norm1 = torch.stack((left_coor_x1 / (w - 1) * 2 - 1, right_coor_y / (h - 1) * 2 - 1), dim=1)
        ## backward warp
        warp_img = torch.nn.functional.grid_sample(img, left_coor_norm1.permute(0, 2, 3, 1))

        return warp_img

    def generate_image_left(self, img, disp):
        return self.apply_disparity(img, -disp)

    def generate_image_right(self, img, disp):
        return self.apply_disparity(img, disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)

    def SSIM_WEIGHT(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        return torch.clamp((SSIM) / 2, 0, 1)

    def disp_smoothness(self, disp, pyramid):

        disp_gradients_x = self.gradient_x(disp)
        disp_gradients_y = self.gradient_y(disp)

        image_gradients_x = self.gradient_x(pyramid)
        image_gradients_y = self.gradient_y(pyramid)

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1,
                     keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1,
                     keepdim=True))

        smoothness_x = disp_gradients_x * weights_x

        smoothness_y = disp_gradients_y * weights_y


        return torch.abs(smoothness_x) + torch.abs(smoothness_y)


    def reconstruction_image_first_order_gradient(self, left_est, left_pyramid):

        RI_x = [self.gradient_x(d) for d in left_est]
        RI_y = [self.gradient_y(d) for d in left_est]

        OI_x = [self.gradient_x(d) for d in left_pyramid]
        OI_y = [self.gradient_y(d) for d in left_pyramid]

        fisrt_order_loss = [torch.mean(torch.abs(RI_x[i] - OI_x[i])) + torch.mean(torch.abs(RI_y[i] - OI_y[i]))
                            for i in range(self.n)]

        return fisrt_order_loss

    def forward(self, disp, ImgL, ImgR):

        """
        Args:
            input [disp1, disp2, disp3, disp4, disp5 ]
            target [left, right]

        Return:
            (float): The loss
        """
        assert len(disp[-1].shape) == 4
        scale = len(disp)

        # disp_pyramid = self.scale_disp(disp, scale)
        disp_pyramid = disp
        left_pyramid = self.scale_pyramid(ImgL, scale)  # [1, 1/2, 1/4,...]
        right_pyramid = self.scale_pyramid(ImgR, scale)

        #pdb.set_trace()
        # Generate images
        loss = []
        for i in range (scale):

            left_est = self.generate_image_left(right_pyramid[i], disp_pyramid[i])

            # Disparities smoothness
            disp_left_smoothness = self.disp_smoothness(disp_pyramid[i], left_pyramid[i])
            l1_left = torch.mean(torch.abs(left_est - left_pyramid[i]))

            ssim_left = torch.mean(self.SSIM(left_est, left_pyramid[i]))
            image_loss_left = self.SSIM_w * ssim_left+ (1 - self.SSIM_w) * (l1_left)

            image_loss = image_loss_left

            # Disparities smoothness
            disp_gradient_loss = torch.mean(torch.abs(disp_left_smoothness))

            # total loss
            loss_scale = image_loss + self.disp_gradient_w * disp_gradient_loss
            loss.append(loss_scale)

        return loss
