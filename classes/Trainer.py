import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from classes.Dataset import collate_fn


class Trainer:
    def __init__(self, model, dataset, batch_size, lr, num_epochs, validate_every, weight_decay=0, early_stopping=False, patience=5, lr_scheduler_patience=5, log_dir="runs/wavenet_train", models_dir="models/", max_batch=None, lr_range_test=False):
        
        # DEVICE
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
            
        # PARAMS
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.validate_every = validate_every
        self.num_epochs = num_epochs
        self.lr = lr
        self.lr_range_test = lr_range_test

        # DATASET DIVISION
        total_size = len(dataset)
        if max_batch is not None:
            total_size = min(max_batch * batch_size, total_size)
            dataset = Subset(dataset, range(total_size))

        train_size = int(0.90 * total_size)
        val_size = int(0.05 * total_size)
        test_size = total_size - train_size - val_size
        self.train_set, self.val_set, self.test_set = random_split(dataset, [train_size, val_size, test_size])
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
        self.val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
        
        # LOSS + OPTIMIZER
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        
        # LR SCHEDULE
        if self.lr_range_test:
            lr_min = 1e-6
            lr_max = 1
            max_batches = (len(self.train_loader) * self.num_epochs) -1

            lr_lambda = lambda step: (lr_min/self.lr) * (lr_max / lr_min) ** (step / max_batches)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=lr_scheduler_patience)
        
        # EARLY STOPPING
        self.early_stopping = early_stopping
        self.patience = patience
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0
        
        # TENSORBOARD LOGGIGN
        self.writer = SummaryWriter(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        
        # MODEL SAVING
        os.makedirs(models_dir, exist_ok=True)



    def train(self):        
        print(f"Starting training on {self.device}...")
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            
        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss, running_correct, total_samples = 0.0, 0, 0 # loss pour une section de validate_every batchs
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            
            for i, (waveforms, speaker_ids) in progress_bar:                
                waveforms, speaker_ids = waveforms.to(self.device), speaker_ids.to(self.device)
                
                self.optimizer.zero_grad()
                
                # PREDICTIONS
                outputs = self.model(waveforms, speaker_ids) * 10
                preds = outputs[:, :, :-1].contiguous().view(-1, 256)
                #print("Preds mean:", preds.mean().item(), "Preds std:", preds.std().item())
                #print("Preds min:", preds.min().item(), "Preds max:", preds.max().item())
                #print(outputs[:, :, :-1].contiguous().shape)  # torch.Size([B, 256, T])
                #print(preds.shape) # torch.Size([T, 256])

                # TARGETS
                #print(waveforms[:, :, 1:].contiguous().shape) # torch.Size([B, 1, T])
                targets = torch.clamp(waveforms[:, :, 1:].contiguous().view(-1), 0, 255) # # torch.Size([T]),  flatten + dans [0, 255)
                #print(targets.shape)
                
                # TODO: régler ce problème de [0, 256] c'est juste pas normal, autrement qu'avec clamp
                #assert targets.min() >= 0 and targets.max() < 256
                #if targets.min() <= 0 or targets.max() >= 255:
                    #print(f"Min target: {targets.min()}, Max target: {targets.max()}")
                
                # LOSS
                loss = self.criterion(preds, targets.long())
                loss.backward()
                self.optimizer.step()
                
                
                running_loss += loss.item()
                total_samples += targets.numel()
                predicted = preds.argmax(dim=1) # torch.Size([B*T])
                running_correct += (predicted == targets).sum().item()
                
                
                tmp = self.validate_every if self.validate_every else len(self.train_loader)
                progress_bar.set_description(f"Epoch [{epoch+1}/{self.num_epochs}]")
                progress_bar.set_postfix(loss=f"{(running_loss/tmp):3f}", acc= running_correct / total_samples)


                # valide + log toutes les validatate_every itérations
                if self.validate_every and (i % self.validate_every == 0) and (i != 0):
                    # DEBUG (affichage des gradients)
                    #for name, param in self.model.named_parameters():
                        #if param.grad is not None:
                            #print(f"{name}: {param.grad.norm().item()}")
                            #print(f"{name}: Grad Mean {param.grad.mean().item()}, Grad Std {param.grad.std().item()}")
                    
                    # TRAIN LOSS
                    train_loss = running_loss / self.validate_every
                    train_acc = running_correct / total_samples
                    self.writer.add_scalar('Loss/train', train_loss, epoch * len(self.train_loader) + i)
                    self.writer.add_scalar('Accuracy/train', train_acc, epoch * len(self.train_loader) + i)
                    running_loss, running_correct, total_samples = 0.0, 0, 0
                    
                    # VAL LOSS
                    val_loss, val_acc = self.validate()
                    print(f"Val loss: {val_loss}, Val acc: {val_acc}")
                    self.writer.add_scalar('Loss/val', val_loss, epoch * len(self.train_loader) + i)
                    self.writer.add_scalar('Accuracy/val', val_acc, epoch * len(self.train_loader) + i)
                    
                    # LOG LR
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar("Learning Rate", current_lr, epoch)

                    # EARLY STOPPING + CHECKPOINTS SAVING
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        torch.save(self.model.state_dict(), 'models/best_model.pth')
                        self.early_stop_counter = 0
                    else:
                        self.early_stop_counter += 1
                    if self.early_stopping:                    
                        if self.early_stop_counter >= self.patience:
                            print(f"Early stopping at batch {i} with current loss {val_loss}.")
                            return
                    
                    # LR SCHED (ReduceOnPlateau se fait à chaque validation)
                    if not self.lr_range_test:
                        self.scheduler.step(val_loss)

                # LR SCHED en range test
                if self.lr_range_test:
                    self.scheduler.step()
            
            # VAL LOSS de fin d'epoch
            val_loss, val_acc = self.validate()
            print(f"Val loss: {val_loss}, Val acc: {val_acc}")
            self.writer.add_scalar('Loss/val', val_loss, epoch * len(self.train_loader) + i)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch * len(self.train_loader) + i)
                
        print("Training complete.")
        self.writer.close()

    def validate(self):
        self.model.eval()
        running_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for waveforms, speakers_id in self.val_loader:
                waveforms, speakers_id = waveforms.to(self.device), speakers_id.to(self.device)
                outputs = self.model(waveforms, speakers_id)
                targets = torch.clamp(waveforms[:, :, 1:].contiguous().view(-1), 0, 255)
                preds = outputs[:, :, :-1].contiguous().view(-1, 256)
                
                loss = self.criterion(preds, targets.long())
                running_loss += loss.item()
                predicted = preds.argmax(dim=1)
                correct += (predicted == targets).sum().item()
                total += targets.numel()
        return running_loss / len(self.val_loader), correct / total
    
    def test(self):
        self.model.eval()
        total_loss, correct, total_samples = 0.0, 0, 0
        
        with torch.no_grad():
            for waveforms, speakers_id in self.test_loader:
                waveforms, speakers_id = waveforms.to(self.device), speakers_id.to(self.device)
                outputs = self.model(waveforms, speakers_id)
                targets = torch.clamp(waveforms[:, :, 1:].contiguous().view(-1), 0, 255)
                preds = outputs[:, :, :-1].contiguous().view(-1, 256)
                
                loss = self.criterion(preds, targets.long())
                
                total_loss += loss.item()
                predictions = preds.argmax(dim=1)
                correct += (predictions == targets).sum().item()
                total_samples += targets.numel()
        
        # TODO: rajouter du logging TB
        avg_loss = total_loss / len(self.test_loader)
        accuracy = correct / total_samples
        print(f"Test Loss: {avg_loss:.4f}, Test Acc: {accuracy:.4f}")
        return avg_loss, accuracy
    
    
    # Fonctions pas encore utilisées
    # TODO: utiliser ces fonctions pour save le modèle (peu importe quand on le fait) + vérifier si un modèle de la bonne architecture existe quand on veut charger des checkpoints pour reprendre un training
    def save_checkpoint(self, epoch, filepath="chkpt/model_checkpoint.pth"):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved at epoch {epoch} to {filepath}")
        
    def load_checkpoint(self, filepath="chkpt/model_checkpoint.pth"):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Checkpoint loaded from {filepath}")

