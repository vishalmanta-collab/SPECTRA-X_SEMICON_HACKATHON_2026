
import os
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):

    def __init__(
        self,
        channels,
        reduction=8
    ):
        super().__init__()

        hidden = max(
            channels // reduction,
            4
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden,
                1
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Conv2d(
                hidden,
                channels,
                1
            ),
            nn.Sigmoid()
        )

    def forward(self, x):

        attention = self.avg_pool(x)
        attention = self.fc(attention)

        return x * attention


class ResidualAttentionBlock(nn.Module):

    def __init__(
        self,
        channels
    ):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1
            ),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1
            )
        )

        self.attention = ChannelAttention(
            channels
        )

    def forward(self, x):

        residual = self.body(x)
        residual = self.attention(
            residual
        )

        return x + residual


class MultiScaleBlock(nn.Module):

    def __init__(
        self,
        channels
    ):
        super().__init__()

        self.conv3 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1
        )

        self.conv5 = nn.Conv2d(
            channels,
            channels,
            5,
            padding=2
        )

        self.fuse = nn.Conv2d(
            channels * 2,
            channels,
            1
        )

        self.act = nn.GELU()

    def forward(self, x):

        a = self.act(
            self.conv3(x)
        )

        b = self.act(
            self.conv5(x)
        )

        out = torch.cat(
            [a, b],
            dim=1
        )

        out = self.fuse(out)

        return x + out


class SPECTRAX(nn.Module):

    def __init__(
        self,
        scale=2.0,
        channels=48,
        blocks=8
    ):
        super().__init__()

        self.scale = scale

        self.head = nn.Conv2d(
            1,
            channels,
            3,
            padding=1
        )

        self.body = nn.Sequential(
            MultiScaleBlock(
                channels
            ),

            *[
                ResidualAttentionBlock(
                    channels
                )
                for _ in range(blocks)
            ],

            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1
            )
        )

        self.upsample = nn.Sequential(

            nn.Conv2d(
                channels,
                channels * 4,
                3,
                padding=1
            ),

            nn.PixelShuffle(2),

            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1
            )
        )

        self.tail = nn.Conv2d(
            channels,
            1,
            3,
            padding=1
        )

    def forward(self, x):

        base = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        features = self.head(x)

        residual = self.body(
            features
        )

        residual = residual + features

        residual = self.upsample(
            residual
        )

        residual = self.tail(
            residual
        )

        output = base + residual

        return torch.clamp(
            output,
            0.0,
            1.0
        )


def load_npy(path):

    arr = np.load(
        path,
        allow_pickle=False
    )

    arr = np.asarray(
        arr,
        dtype=np.float32
    ).copy()

    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2D grayscale image, "
            f"got {arr.shape}: {path}"
        )

    return arr


def preprocess(arr):

    arr = np.asarray(
        arr,
        dtype=np.float32
    ).copy()

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    if arr.max() > 1.0:

        scale = max(
            float(
                np.percentile(
                    arr,
                    99.5
                )
            ),
            1e-8
        )

        arr = arr / scale

    arr = np.clip(
        arr,
        0.0,
        1.0
    )

    return torch.from_numpy(
        arr
    ).float().unsqueeze(
        0
    ).unsqueeze(
        0
    )


def load_model(
    checkpoint_path,
    device
):

    model = SPECTRAX(
        scale=2.0,
        channels=48,
        blocks=8
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.to(device)
    model.eval()

    return model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        required=True
    )

    parser.add_argument(
        "--checkpoint",
        default="weights/SPECTRAX_KLA_best.pth"
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("SPECTRA-X KLA STANDALONE INFERENCE")
    print("=" * 70)

    print(
        "Device:",
        device
    )

    print(
        "Checkpoint:",
        args.checkpoint
    )

    print(
        "Input:",
        args.input_dir
    )

    print(
        "Output:",
        args.output_dir
    )

    model = load_model(
        args.checkpoint,
        device
    )

    files = sorted(
        [
            f
            for f in os.listdir(
                args.input_dir
            )
            if f.lower().endswith(
                ".npy"
            )
        ]
    )

    if len(files) == 0:

        raise RuntimeError(
            "No .npy files found in input directory."
        )

    print(
        "Images found:",
        len(files)
    )

    with torch.no_grad():

        for i, filename in enumerate(files):

            input_path = os.path.join(
                args.input_dir,
                filename
            )

            arr = load_npy(
                input_path
            )

            tensor = preprocess(
                arr
            ).to(device)

            output = model(
                tensor
            )

            output = output.squeeze().cpu().numpy()

            output = np.clip(
                output,
                0.0,
                1.0
            )

            output_uint8 = (
                output * 255.0
            ).round().clip(
                0,
                255
            ).astype(
                np.uint8
            )

            stem = os.path.splitext(
                filename
            )[0]

            output_path = os.path.join(
                args.output_dir,
                stem + ".png"
            )

            Image.fromarray(
                output_uint8
            ).save(
                output_path
            )

            if (
                (i + 1) % 50 == 0
                or
                i == len(files) - 1
            ):

                print(
                    f"Processed "
                    f"{i + 1}/{len(files)}"
                )

    print("=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)

    print(
        "Restored images:",
        len(files)
    )


if __name__ == "__main__":
    main()
