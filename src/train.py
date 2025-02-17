import os
import random
import torch
import numpy as np
import argparse
from torch import optim
from torchvision import models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import get_train_test, VideoDataset, get_transform, collate_fn_cnn_rnn, collate_fn_r3d_18
from model import CNN_RNN_Model
import matplotlib.pyplot as plt
from torchvision.models.video import r3d_18, R3D_18_Weights


class VideoClassifier:
    def __init__(self, args):
        self.args = args
        self.device = self.set_seed_and_device(args.seed)
        self.model = self.create_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-5)
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="sum").to(self.device)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=1)
        self.train_dl, self.test_dl = self.load_data()
        self.best_loss = float('inf')
        self.loss_hist = {'train':[], 'test':[]}
        self.acc_hist = {'train':[], 'test':[]}
        self.save_dir = './checkpoints'
        self.plot_dir = './plots'
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

    def set_seed_and_device(self, seed):
        """Sets the random seeds and device for training."""
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        np.random.seed(seed)
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def load_data(self):
        """Loads and prepares the train/test datasets and dataloaders."""
        train_ids, train_labels, test_ids, test_labels, dic = get_train_test()
        
        print(f'model_type:{self.args.model_type}')
        trans_train = get_transform(model_type=self.args.model_type, phase='train')
        trans_test = get_transform(model_type=self.args.model_type, phase='test')
        
        train_ds = VideoDataset(vids=train_ids, labels=train_labels, transform=trans_train, dic=dic)
        test_ds = VideoDataset(vids=test_ids, labels=test_labels, transform=trans_test, dic=dic)
        
        if self.args.model_type == "cnn_rnn":
            train_dl = DataLoader(train_ds, batch_size=self.args.bs, shuffle=True, collate_fn=collate_fn_cnn_rnn)
            test_dl = DataLoader(test_ds, batch_size=2 * self.args.bs, shuffle=False, collate_fn=collate_fn_cnn_rnn)
        else:
            train_dl = DataLoader(train_ds, batch_size=self.args.bs, shuffle=True, collate_fn=collate_fn_r3d_18)
            test_dl = DataLoader(test_ds, batch_size=2 * self.args.bs, shuffle=False, collate_fn=collate_fn_r3d_18)
        
        return train_dl, test_dl

    def create_model(self):
        """Creates the model and moves it to the appropriate device."""
        if self.args.model_type=='cnn_rnn':
            model = CNN_RNN_Model(feature_dim=512, hidden_dim=100, num_layers=1, num_classes=5)
        elif self.args.model_type=='3dcnn':
            model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
            
        return model.to(self.device)

    def save_checkpoint(self, epoch, val_loss, val_acc):
        """Saves the model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        checkpoint_path = os.path.join(self.save_dir,self.args.model_type,f'best_model_epoch_{epoch+1}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    def train_epoch(self, dataloader):
        """Trains the model for one epoch."""
        self.model.train()
        running_loss = 0
        correct_preds = 0
        total_preds = 0

        for img, label in dataloader:
            img = img.to(self.device)
            label = label.to(self.device)

            output = self.model(img)
            loss = self.loss_func(output, label)

            _, preds = torch.max(output, 1)
            correct = (preds == label).sum().item()

            correct_preds += correct
            total_preds += len(label)

            running_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        avg_loss = running_loss / len(dataloader.dataset)
        avg_accuracy = correct_preds / total_preds
        return avg_loss, avg_accuracy

    def evaluate_epoch(self, dataloader):
        """Evaluates the model on validation or test set for one epoch."""
        self.model.eval()
        running_loss = 0
        correct_preds = 0
        total_preds = 0

        with torch.no_grad():
            for img, label in dataloader:
                img = img.to(self.device)
                label = label.to(self.device)

                output = self.model(img)
                loss = self.loss_func(output, label)

                _, preds = torch.max(output, 1)
                correct = (preds == label).sum().item()

                correct_preds += correct
                total_preds += len(label)

                running_loss += loss.item()

        avg_loss = running_loss / len(dataloader.dataset)
        avg_accuracy = correct_preds / total_preds
        return avg_loss, avg_accuracy

    def train(self):
        """Trains and evaluates the model across all epochs."""
        for epoch in range(self.args.epochs):
            print(f"Epoch {epoch+1}/{self.args.epochs}")
            
            # Train phase
            train_loss, train_accuracy = self.train_epoch(self.train_dl)
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

            # Evaluate phase
            val_loss, val_accuracy = self.evaluate_epoch(self.test_dl)
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            
            self.loss_hist['train'].append(train_loss)
            self.loss_hist['test'].append(val_loss)
            
            self.acc_hist['train'].append(train_accuracy)
            self.acc_hist['test'].append(val_accuracy)
            
            # Learning rate scheduler step
            self.lr_scheduler.step(val_loss)

            # Save checkpoint if the model is the best one
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(epoch, val_loss, val_accuracy)

        # Save the last model checkpoint
        self.save_checkpoint(self.args.epochs - 1, self.best_loss, val_accuracy)
        
    def draw_loss(self):
        """Draw loss and accuracy plot across all epochs for monitoring overfitting"""
        # Plot Loss History
        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
        plt.plot(self.loss_hist['train'], label='Train Loss', marker='o')
        plt.plot(self.loss_hist['test'], label='Validation Loss', marker='s')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss History')
        plt.legend()
        plt.grid()
        
        # Plot Accuracy History
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
        plt.plot(self.acc_hist['train'], label='Train Accuracy', marker='o')
        plt.plot(self.acc_hist['test'], label='Validation Accuracy', marker='s')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy History')
        plt.legend()
        plt.grid()
        
        plt.tight_layout()  # Adjust layout to prevent overlap
        
        # Create the full file path for the plot
        plot_dir = os.path.join(self.plot_dir, f'{self.args.model_type}_training_plots.png')
        plt.savefig(plot_dir, dpi=300, bbox_inches='tight')

        

def parse_args():
    parser = argparse.ArgumentParser(description='Classifying videos using CNN-RNN and 3dCNN')
    parser.add_argument('--seed', type=int, default=42, help='the seed for experiments!')
    parser.add_argument('--exp', type=int, default=1, help='the experiment number!')
    parser.add_argument('--model_type', type=str, default='3dcnn', choices=('cnn_rnn', '3dcnn'),
                        help='the model used for training!')
    parser.add_argument('--bs', type=int, default=16, help='batch size for training model!')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train the model!')
    return parser.parse_args()


def main():
    args = parse_args()
    classifier = VideoClassifier(args)
    classifier.train()
    classifier.draw_loss()


if __name__ == '__main__':
    main()
