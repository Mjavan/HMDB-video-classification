from torch import nn
from torchvision import models
import torch


class Resnt18Rnn(nn.Module):
    def __init__(self, params_model):
        super(Resnt18Rnn, self).__init__()
        num_classes = params_model["num_classes"]
        dr_rate= params_model["dr_rate"]
        pretrained = params_model["pretrained"]
        rnn_hidden_size = params_model["rnn_hidden_size"]
        rnn_num_layers = params_model["rnn_num_layers"]
        
        baseModel = models.resnet18(pretrained=pretrained)
        num_features = baseModel.fc.in_features
        baseModel.fc = Identity()
        self.baseModel = baseModel
        self.dropout= nn.Dropout(dr_rate)
        self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers)
        self.fc1 = nn.Linear(rnn_hidden_size, num_classes)
    def forward(self, x):
        b_z, ts, c, h, w = x.shape
        ii = 0
        y = self.baseModel((x[:,ii]))
        output, (hn, cn) = self.rnn(y.unsqueeze(1))
        for ii in range(1, ts):
            y = self.baseModel((x[:,ii]))
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
        out = self.dropout(out[:,-1])
        out = self.fc1(out) 
        return out 
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x 
    

# This model was suggested by chatGPT
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Use a pretrained ResNet18
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # Remove final FC layer

    def forward(self, x):
        features = self.feature_extractor(x)  # Extract features
        return features.view(features.size(0), -1)  # Flatten output
    
    
class VideoRNN(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=100, num_layers=1, num_classes=5):
        super(VideoRNN, self).__init__()
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)  # Final classification layer

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # Process sequence with LSTM
        last_frame_out = lstm_out[:, -1, :]  # Take the last frame output
        return self.fc(last_frame_out)  # Classification output
    

class CNN_RNN_Model(nn.Module):
    def __init__(self,feature_dim=512,hidden_dim=100,num_layers=1, num_classes=10):
        super(CNN_RNN_Model, self).__init__()
        self.cnn = CNNFeatureExtractor()
        self.rnn = VideoRNN(feature_dim=feature_dim,hidden_dim=hidden_dim,num_layers=num_layers,num_classes=num_classes)

    def forward(self, x):
        batch_size, time_steps, C, H, W = x.shape
        cnn_features = [self.cnn(x[:, t, :, :, :]) for t in range(time_steps)]  # Process each frame
        cnn_features = torch.stack(cnn_features, dim=1)  # Convert list to tensor
        return self.rnn(cnn_features)  # Pass to RNN
    
    
    
    
    

