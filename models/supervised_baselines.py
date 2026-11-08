"""Exact model class definitions recovered from original final Stage 5G.3.

Provenance: HAR_source_review_02.zip / history/stage5g3_supervised_clean_s194_l22.py
No GPU locks, filesystem side effects, or training at import time.
"""
import torch
import torch.nn as nn

class BaselineCNN1D(
    nn.Module
):

    def __init__(
        self,
        num_classes=4
    ):

        super().__init__()


        self.features = nn.Sequential(

            nn.Conv1d(
                3,
                64,
                kernel_size=7,
                padding=3
            ),

            nn.BatchNorm1d(
                64
            ),

            nn.ReLU(),

            nn.MaxPool1d(
                2
            ),

            nn.Conv1d(
                64,
                128,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(
                128
            ),

            nn.ReLU(),

            nn.MaxPool1d(
                2
            ),

            nn.Conv1d(
                128,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(
                128
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(
                1
            ),
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                128,
                num_classes
            ),
        )


    def forward(
        self,
        x
    ):

        return self.classifier(
            self.features(
                x
            )
        )


class BaselineResidualBlock(
    nn.Module
):

    def __init__(
        self,
        in_ch,
        out_ch,
        stride=1
    ):

        super().__init__()


        self.conv1 = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=5,
            stride=stride,
            padding=2,
            bias=False
        )

        self.bn1 = nn.BatchNorm1d(
            out_ch
        )


        self.conv2 = nn.Conv1d(
            out_ch,
            out_ch,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm1d(
            out_ch
        )


        if (
            stride != 1
            or
            in_ch != out_ch
        ):

            self.skip = nn.Sequential(

                nn.Conv1d(
                    in_ch,
                    out_ch,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),

                nn.BatchNorm1d(
                    out_ch
                ),
            )

        else:

            self.skip = (
                nn.Identity()
            )


    def forward(
        self,
        x
    ):

        identity = (
            self.skip(
                x
            )
        )


        out = torch.relu(
            self.bn1(
                self.conv1(
                    x
                )
            )
        )


        out = self.bn2(
            self.conv2(
                out
            )
        )


        return torch.relu(
            out
            +
            identity
        )


class BaselineResNet1D(
    nn.Module
):

    def __init__(
        self,
        num_classes=4
    ):

        super().__init__()


        self.stem = nn.Sequential(

            nn.Conv1d(
                3,
                64,
                kernel_size=7,
                padding=3,
                bias=False
            ),

            nn.BatchNorm1d(
                64
            ),

            nn.ReLU(),
        )


        self.layers = nn.Sequential(

            BaselineResidualBlock(
                64,
                64
            ),

            BaselineResidualBlock(
                64,
                128,
                stride=2
            ),

            BaselineResidualBlock(
                128,
                128
            ),

            BaselineResidualBlock(
                128,
                256,
                stride=2
            ),

            BaselineResidualBlock(
                256,
                256
            ),
        )


        self.pool = (
            nn.AdaptiveAvgPool1d(
                1
            )
        )


        self.fc = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                256,
                num_classes
            ),
        )


    def forward(
        self,
        x
    ):

        x = self.stem(
            x
        )

        x = self.layers(
            x
        )

        x = self.pool(
            x
        )

        return self.fc(
            x
        )


class BaselineTCNBlock(
    nn.Module
):

    def __init__(
        self,
        in_ch,
        out_ch,
        dilation
    ):

        super().__init__()


        # Correct SAME-length padding:
        padding = dilation


        self.conv1 = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            bias=False
        )


        self.bn1 = nn.BatchNorm1d(
            out_ch
        )


        self.conv2 = nn.Conv1d(
            out_ch,
            out_ch,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            bias=False
        )


        self.bn2 = nn.BatchNorm1d(
            out_ch
        )


        self.skip = (
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=1,
                bias=False
            )
            if in_ch != out_ch
            else nn.Identity()
        )


    def forward(
        self,
        x
    ):

        identity = (
            self.skip(
                x
            )
        )


        out = torch.relu(
            self.bn1(
                self.conv1(
                    x
                )
            )
        )


        out = self.bn2(
            self.conv2(
                out
            )
        )


        if (
            out.shape
            !=
            identity.shape
        ):

            raise RuntimeError(
                "TCN residual mismatch: "
                f"{tuple(out.shape)} vs "
                f"{tuple(identity.shape)}"
            )


        return torch.relu(
            out
            +
            identity
        )


class BaselineTCN(
    nn.Module
):

    def __init__(
        self,
        num_classes=4
    ):

        super().__init__()


        self.network = nn.Sequential(

            BaselineTCNBlock(
                3,
                64,
                1
            ),

            BaselineTCNBlock(
                64,
                128,
                2
            ),

            BaselineTCNBlock(
                128,
                128,
                4
            ),

            BaselineTCNBlock(
                128,
                256,
                8
            ),
        )


        self.pool = (
            nn.AdaptiveAvgPool1d(
                1
            )
        )


        self.fc = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                256,
                num_classes
            ),
        )


    def forward(
        self,
        x
    ):

        x = self.network(
            x
        )

        x = self.pool(
            x
        )

        return self.fc(
            x
        )

MODELS = {
    "cnn1d": BaselineCNN1D,
    "resnet1d": BaselineResNet1D,
    "tcn": BaselineTCN,
}

def build_model(name: str, classes: int = 4) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"Unknown supervised model: {name}")
    return MODELS[name](num_classes=classes)
