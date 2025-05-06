import torch
import torch.nn as nn
import torchvision.models as models

class InverseModel(nn.Module):
    def __init__(self, inv_hidden_dim, action_dim, inv_act='relu'):
        super(InverseModel, self).__init__()
        self.inv_hidden_dim = inv_hidden_dim
        self.action_dim = action_dim
        self.inv_act = inv_act
        
        self.create_inv_model()

    def create_inv_model(self):
        activation = nn.ReLU if self.inv_act == 'relu' else nn.Mish
        
        # Load the pretrained ResNet-18 model
        resnet = models.resnet18(pretrained=True)
        # Remove the fully connected layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # The output dimension of ResNet-18's feature extractor
        resnet_output_dim = resnet.fc.in_features

        self.inv_model = nn.Sequential(
            nn.Linear(resnet_output_dim * 2, self.inv_hidden_dim),
            activation(),
            nn.Linear(self.inv_hidden_dim, self.inv_hidden_dim),
            activation(),
            nn.Linear(self.inv_hidden_dim, self.action_dim),
        )

    def forward(self, x):
        batch_size, channels, frames, height, width = x.size()
        
        # Process each frame separately
        x = x.reshape(batch_size * frames, channels, height, width)
        with torch.no_grad():
            x = self.feature_extractor(x)
        
        x = x.reshape(batch_size, frames, -1)  # Reshape to (batch, frames, feature_dim)

        # Initialize a tensor to store the actions
        actions = torch.zeros(batch_size, frames - 1, self.action_dim, device=x.device)

        for i in range(frames - 1):
            combined_features = torch.cat((x[:, i, :], x[:, i + 1, :]), dim=1)
            actions[:, i, :] = self.inv_model(combined_features)

        return actions
    
if __name__ == '__main__':
    # Test the model
    batch_size = 1
    channels = 3
    frames = 16
    height = 96
    width = 96

    # Create a dummy input tensor with the specified shape
    input_tensor = torch.randn(batch_size, channels, frames, height, width)

    # Instantiate the model
    inv_model = InverseModel(inv_hidden_dim=512, action_dim=10, inv_act='relu')

    # Forward pass
    output = inv_model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)
