import torch
import torch.nn as nn

class SimpleDNA_CNN(nn.Module):
    def __init__(self, num_classes=4, seq_length=300):
        super(SimpleDNA_CNN, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            )

        # Calculate the output size of the convolutional layer
        # Pass a dummy tensor through the conv_layer to determine the size
        dummy_input = torch.randn(1, 4, seq_length)
        with torch.no_grad():
            conv_output = self.conv_layer(dummy_input)
        conv_output_size = conv_output.view(1, -1).size(1)


        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self,x):
      x = self.conv_layer(x)
      return self.classifier(x)