import torch
from torch import nn
from transformers import RobertaModel

class Mind_CNN(nn.Module):
    def __init__(self, embed_dim=768, num_classes=23, pretrained_model_path='./models/pretrained_iot-continue_2/checkpoint-19000/', dropout_rate=0.3):
        super(Mind_CNN, self).__init__()

        # Load pretrained weights for embeddings
        pretrained_weights = RobertaModel.from_pretrained(pretrained_model_path).embeddings.word_embeddings.weight
        self.embed = nn.Embedding.from_pretrained(pretrained_weights, freeze=True, padding_idx=1)

        # Convolutional layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=500, kernel_size=k) for k in [3, 4, 5]
        ])

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layers
        self.fc1 = nn.Linear(1500, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.output = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0, 2, 1)

        # Apply convolutions and poolings
        x = [nn.functional.max_pool1d(nn.functional.relu(conv(x)), conv(x).shape[2]) for conv in self.convs]
        x = torch.cat(x, dim=1)

        x = x.flatten(1)
        x = self.dropout(x)

        # Fully connected layers with ReLU activations
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))

        # Output layer
        x = self.output(x)
        return x
