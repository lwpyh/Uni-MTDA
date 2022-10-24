import torch
import numpy as np
from PIL import Image as PILImage
from imagecorruptions import corrupt, get_corruption_names

class ACVCGenerator:
    def __init__(self):
        return

    def get_severity(self):
        return np.random.randint(1, 6)

    def draw_cicle(self, shape, diamiter):
        """
        Input:
        shape    : tuple (height, width)
        diameter : scalar
        Output:
        np.array of shape  that says True within a circle with diamiter =  around center
        """
        assert len(shape) == 2
        TF = np.zeros(shape, dtype="bool")
        center = np.array(TF.shape) / 2.0

        for iy in range(shape[0]):
            for ix in range(shape[1]):
                TF[iy, ix] = (iy - center[0]) ** 2 + (ix - center[1]) ** 2 < diamiter ** 2
        return TF

    def filter_circle(self, TFcircle, fft_img_channel):
        temp = np.zeros(fft_img_channel.shape[:2], dtype=complex)
        temp[TFcircle] = fft_img_channel[TFcircle]
        return temp

    def inv_FFT_all_channel(self, fft_img):
        img_reco = []
        for ichannel in range(fft_img.shape[2]):
            img_reco.append(np.fft.ifft2(np.fft.ifftshift(fft_img[:, :, ichannel])))
        img_reco = np.array(img_reco)
        img_reco = np.transpose(img_reco, (1, 2, 0))
        return img_reco

    def high_pass_filter(self, x, severity):
        x = x.astype("float32") / 255.
        c = [.01, .02, .03, .04, .05][severity - 1]

        d = int(c * x.shape[0])
        TFcircle = self.draw_cicle(shape=x.shape[:2], diamiter=d)
        TFcircle = ~TFcircle

        fft_img = np.zeros_like(x, dtype=complex)
        for ichannel in range(fft_img.shape[2]):
            fft_img[:, :, ichannel] = np.fft.fftshift(np.fft.fft2(x[:, :, ichannel]))

        # For each channel, pass filter
        fft_img_filtered = []
        for ichannel in range(fft_img.shape[2]):
            fft_img_channel = fft_img[:, :, ichannel]
            temp = self.filter_circle(TFcircle, fft_img_channel)
            fft_img_filtered.append(temp)
        fft_img_filtered = np.array(fft_img_filtered)
        fft_img_filtered = np.transpose(fft_img_filtered, (1, 2, 0))
        x = np.clip(np.abs(self.inv_FFT_all_channel(fft_img_filtered)), a_min=0, a_max=1)

        x = PILImage.fromarray((x * 255.).astype("uint8"))
        return x

    def constant_amplitude(self, x, severity):
        """
        A visual corruption based on amplitude information of a Fourier-transformed image
        Adopted from: https://github.com/MediaBrain-SJTU/FACT
        """
        x = x.astype("float32") / 255.
        c = [.05, .1, .15, .2, .25][severity - 1]

        # FFT
        x_fft = np.fft.fft2(x, axes=(0, 1))
        x_abs, x_pha = np.fft.fftshift(np.abs(x_fft), axes=(0, 1)), np.angle(x_fft)

        # Amplitude replacement
        beta = 1.0 - c
        x_abs = np.ones_like(x_abs) * max(0, beta)

        # Inverse FFT
        x_abs = np.fft.ifftshift(x_abs, axes=(0, 1))
        x = x_abs * (np.e ** (1j * x_pha))
        x = np.real(np.fft.ifft2(x, axes=(0, 1)))

        x = PILImage.fromarray((x * 255.).astype("uint8"))
        return x

    def phase_scaling(self, x, severity):
        """
        A visual corruption based on phase information of a Fourier-transformed image
        Adopted from: https://github.com/MediaBrain-SJTU/FACT
        """
        x = x.astype("float32") / 255.
        c = [.1, .2, .3, .4, .5][severity - 1]

        # FFT
        x_fft = np.fft.fft2(x, axes=(0, 1))
        x_abs, x_pha = np.fft.fftshift(np.abs(x_fft), axes=(0, 1)), np.angle(x_fft)

        # Phase scaling
        alpha = 1.0 - c
        x_pha = x_pha * max(0, alpha)

        # Inverse FFT
        x_abs = np.fft.ifftshift(x_abs, axes=(0, 1))
        x = x_abs * (np.e ** (1j * x_pha))
        x = np.real(np.fft.ifft2(x, axes=(0, 1)))

        x = PILImage.fromarray((x * 255.).astype("uint8"))
        return x

    def apply_corruption(self, x, corruption_name):
        severity = self.get_severity()

        custom_corruptions = {"high_pass_filter": self.high_pass_filter,
                              "constant_amplitude": self.constant_amplitude,
                              "phase_scaling": self.phase_scaling}

        if corruption_name in get_corruption_names('all'):
            x = corrupt(x, corruption_name=corruption_name, severity=severity)
            x = PILImage.fromarray(x)

        elif corruption_name in custom_corruptions:
            x = custom_corruptions[corruption_name](x, severity=severity)

        else:
            assert True, "%s is not a supported corruption!" % corruption_name
        return x

    def __call__(self, x):
        i = np.random.randint(0, 22)
        corruption_func = {0: "fog",
                           1: "snow",
                           2: "frost",
                           3: "spatter",
                           4: "zoom_blur",
                           5: "defocus_blur",
                           6: "glass_blur",
                           7: "gaussian_blur",
                           8: "motion_blur",
                           9: "speckle_noise",
                           10: "shot_noise",
                           11: "impulse_noise",
                           12: "gaussian_noise",
                           13: "jpeg_compression",
                           14: "pixelate",
                           15: "elastic_transform",
                           16: "brightness",
                           17: "saturate",
                           18: "contrast",
                           19: "high_pass_filter",
                           20: "constant_amplitude",
                           21: "phase_scaling"}
        x = np.array(x)
        return self.apply_corruption(x, corruption_func[i])
