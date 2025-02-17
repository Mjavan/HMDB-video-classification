import torch
import random
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights
from dataset import get_train_test, VideoDataset, get_transform, collate_fn_cnn_rnn, collate_fn_r3d_18
from model import CNN_RNN_Model


class VideoClassifierInference:
    def __init__(self, seed=42, model_type='cnn_rnn', batch_size=16, root='./checkpoints'):
        self.seed = seed
        self.model_type = model_type
        self.batch_size = batch_size
        self.root = root
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._set_seed()
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="sum").to(self.device)
        self.model = self._load_model()
        self.test_dl = self._create_dataloader()

    def _set_seed(self):
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(self.seed)
        np.random.seed(self.seed)

    def _load_model(self):
        print(f'{self.model_type}')
        if self.model_type == 'cnn_rnn':
            model = CNN_RNN_Model(feature_dim=512, hidden_dim=100, num_layers=1, num_classes=5)
        elif self.model_type == '3dcnn':
            model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        else:
            raise ValueError("Invalid model type. Choose either 'cnn_rnn' or '3dcnn'")
        ckpt_path = os.path.join(f'{self.root}',f'{self.model_type}','best_model_epoch_20.pt')
        model = model.to(self.device)
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def _create_dataloader(self):
        test_vids, test_labels, dic = get_train_test(phase='test')
        trans_test = get_transform(model_type=self.model_type, phase='test')
        test_ds = VideoDataset(vids=test_vids, labels=test_labels, transform=trans_test, dic=dic)
        
        collate_fn = collate_fn_cnn_rnn if self.model_type == "cnn_rnn" else collate_fn_r3d_18
        return DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

    def run_inference(self):
        self.model.eval()
        running_loss = 0
        correct_preds = 0
        total_preds = 0
        
        print("Inference started...")
        with torch.no_grad():
            for batch, (img, label) in enumerate(self.test_dl):
                img, label = img.to(self.device), label.to(self.device)
                output = self.model(img)
                loss = self.loss_func(output, label)
                _, preds = torch.max(output, 1)
                
                correct_preds += (preds == label).sum().item()
                total_preds += len(label)
                running_loss += loss.item()
                
                if batch == 10:  # Break after 10 batches for efficiency
                    break
        
        avg_loss = running_loss / len(self.test_dl.dataset)
        avg_accuracy = correct_preds / total_preds
        print(f'Accuracy: {avg_accuracy:.4f}, Loss: {avg_loss:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classifying videos using CNN-RNN and 3D CNN')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--model_type', type=str, default='3dcnn', choices=('cnn_rnn', '3dcnn'), help='Model type')
    parser.add_argument('--bs', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--root', type=str, default='./checkpoints')
    args = parser.parse_args()

    inference_runner = VideoClassifierInference(seed=args.seed, model_type=args.model_type, batch_size=args.bs,root=args.root)
                                                
    inference_runner.run_inference()
