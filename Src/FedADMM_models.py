import os
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


class FedADMMAlgorithm:
    def __init__(self, model, n_gru_hidden_units, num_class, timestamp, train_loader, rho):
        self.rho = rho
        self.model = model
        self.n_gru_hidden_units = n_gru_hidden_units
        self.num_class = num_class
        self.timestamp = timestamp
        self.train_loader = train_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = self.model(self.n_gru_hidden_units, self.num_class, self.timestamp).to(self.device)
    
    def train(self, model, z_param, u_param, device, train_loader, optimizer, criterion):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            train_loss += loss.item()

            # update augmented lagrangian with its gradient
            ####### 
            model_weights_pre = copy.deepcopy(model.state_dict())  
            for name, param in model.named_parameters():
                if param.requires_grad == True:
                    param.grad = param.grad + (u_param[name] + self.rho * (model_weights_pre[name] - z_param[name]))
            #######
                    
            optimizer.step()
        
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = 100 * correct / total
        return train_loss, train_accuracy

    def test(self, model, device, test_loader, criterion):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                _, predicted = output.max(1)
                correct += predicted.eq(target).sum().item()

        test_loss /= len(test_loader)
        test_accuracy = 100. * correct / len(test_loader.dataset)
        return test_loss, test_accuracy
    
    def run(self, num_rounds, num_epochs):
        # auxulary variable
        z_param = copy.deepcopy(self.global_model.state_dict())
        
        result = []
        round_accuracy_all = []
        for round in range(num_rounds):
            print(f"---------- Round {round + 1}/{num_rounds} ----------")

            # List to store local model updates
            local_model_updates = []
            clients_local_sum = []

            client_results = {'loss':[], 'accuracy':[]}

            # Iterate over each client
            for client_id in range(1, len(self.train_loader)+1):
                print(f"\nTraining on Client {client_id}")

                local_sum = {}

                # Create a local copy
                local_model = self.model(self.n_gru_hidden_units, self.num_class, self.timestamp).to(self.device)
                local_model.load_state_dict(z_param)  # Initialize with z parameters

                u_param = {}
                weights = copy.deepcopy(local_model.state_dict())
                for key in weights.keys():
                    u_param[key] = torch.zeros_like(weights[key]).cuda()

                local_model_w_prev = copy.deepcopy(local_model.state_dict())
                u_param_prev = copy.deepcopy(u_param)

                # Define loss function and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(local_model.parameters(), lr=0.001)

                # Local training
                loss, accuracy = [], []
                for epoch in range(num_epochs):

                    train_loss, train_accuracy = self.train(local_model, z_param, u_param, self.device, self.train_loader[f'client{client_id}']['train'], optimizer , criterion)
                    val_loss, val_accuracy = self.test(local_model, self.device, self.train_loader[f'client{client_id}']['test'], criterion)

                    loss.append((train_loss, val_loss))
                    accuracy.append((train_accuracy, val_accuracy))
                    print(f'        Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
                    
                client_results['loss'].append(loss)
                client_results['accuracy'].append(accuracy)


                # FedADMM
                # update u parameter
                weights = local_model.state_dict()
                for key in u_param.keys():
                    u_param[key] = u_param[key] + self.rho * (weights[key] - z_param[key])
                # calculate local sum of model u and client
                for key in u_param.keys():
                    local_sum[key] = (weights[key] - local_model_w_prev[key])+ (1/self.rho) * (u_param[key] - u_param_prev[key])
                
            
                # Save the locally updated model parameters
                local_model_updates.append(local_model.state_dict())
                clients_local_sum.append(local_sum)

            # Aggregate local model updates using FedADMM
            averaged_state_dict = {}
            
            client_size = [len(v['test'].dataset) for k, v in self.train_loader.items()]
            for key in self.global_model.state_dict():
                # Weighted average of the model parameters
                averaged_state_dict[key] = sum(client_size[ind]*update[key] for ind, update in enumerate(clients_local_sum)) / np.sum(client_size)

            # update z parameters
            for key in z_param.keys():
                z_param[key] = z_param[key] + 0.01 * averaged_state_dict[key]

            # Update the global model
            self.global_model.load_state_dict(z_param)

            # calculate round test accuracy with weighted mean
            round_acc = []
            for cl, data_ in self.train_loader.items():
                _, val_accuracy = self.test(self.global_model, self.device, data_['test'], criterion)
                round_acc.append(val_accuracy)
            round_accuracy = np.average(round_acc, weights=client_size)
            print(f'\n The round accuracy is: {round_accuracy}')
            round_accuracy_all.append(round_accuracy)
            result.append(client_results)
            result.append(client_results)

        torch.save(self.global_model.state_dict(), f'./FedADMM_model.pth')
        return round_accuracy_all, result
    

