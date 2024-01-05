import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from sam import SAM

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")



class FedProxAlgorithm:
    def __init__(self, model, n_gru_hidden_units, num_class, timestamp, train_loader, mu=0.01):
        self.mu = mu
        self.model = model
        self.n_gru_hidden_units = n_gru_hidden_units
        self.num_class = num_class
        self.timestamp = timestamp
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.global_model = self.model(self.n_gru_hidden_units, self.num_class, self.timestamp).to(self.device)
        self.train_loader = train_loader
        

    def proximal_penalty(self, global_model, local_model, mu):
        # Add the proximal term to the loss
        
        # penalty = 0
        # for local_weights, global_weights in zip(local_model.parameters(), [val.detach().clone() for val in global_model.parameters()]):
        #     penalty += torch.square((local_weights - global_weights).norm(2))
        # return penalty
            
        global_state_dict = global_model.state_dict()
        local_state_dict = local_model.state_dict()

        weight_differences = {k: torch.abs(global_state_dict[k] - local_state_dict[k]) for k in global_state_dict}
        # Compute the L2 norm
        norm_diff = torch.sqrt(sum(torch.sum(w**2) for w in weight_differences.values()))
        return (mu / 2.0) * norm_diff.item()

    
    def train(self, model, device, train_loader, optimizer, criterion):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            penalty = self.proximal_penalty(self.global_model, model, self.mu)
            loss += penalty
            train_loss += loss.item()
            loss.backward()
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
        result = []
        round_accuracy_all = []
        for round in range(num_rounds):
            print(f"---------- Round {round + 1}/{num_rounds} ----------")

            local_model_updates = []
            client_results = {'loss': [], 'accuracy': []}

            for client_id in range(1, len(self.train_loader) + 1):
                print(f"\nTraining on Client {client_id}")

                # Create a local copy
                local_model = self.model(self.n_gru_hidden_units, self.num_class, self.timestamp).to(self.device)
                local_model.load_state_dict(self.global_model.state_dict())
                
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(local_model.parameters(), lr=0.001)

                # Local training
                loss, accuracy = [], []
                for epoch in range(num_epochs):

                    train_loss, train_accuracy = self.train(local_model, self.device, self.train_loader[f'client{client_id}']['train'], optimizer, criterion)
                    val_loss, val_accuracy = self.test(local_model, self.device, self.train_loader[f'client{client_id}']['test'], criterion)
                    
                    loss.append((train_loss, val_loss))
                    accuracy.append((train_accuracy, val_accuracy))
                    print(f'        Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, ', 
                          f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

                
                client_results['loss'].append(loss)
                client_results['accuracy'].append(accuracy)

                local_model_updates.append(local_model.state_dict())
            
            # FedProx
            # Weighted average updating
            averaged_state_dict = {}
            client_size = [len(v['train'].dataset) for k, v in self.train_loader.items()]
            
            for key in self.global_model.state_dict():
                averaged_state_dict[key] = sum(client_size[ind]*update[key] for ind, update in enumerate(local_model_updates)) / np.sum(client_size)
            
            # update global model
            self.global_model.load_state_dict(averaged_state_dict)

            # calculate round test accuracy with weighted mean
            round_acc = []
            client_size = [len(v['test'].dataset) for k, v in self.train_loader.items()]
            for cl, data_ in self.train_loader.items():
                _, val_accuracy = self.test(self.global_model, self.device, data_['test'], criterion)
                round_acc.append(val_accuracy)
            round_accuracy = np.average(round_acc, weights=client_size)
            print(f'\n The round accuracy is: {round_accuracy}')
            round_accuracy_all.append(round_accuracy)
            result.append(client_results)
        
        torch.save(self.global_model.state_dict(), f'./results/FedProx_model.pth')
        return round_accuracy_all, result
    


class FedProxAlgorithm_SAM:
    def __init__(self, model, n_gru_hidden_units, num_class, timestamp, train_loader, mu=0.01):
        self.mu = mu
        self.model = model
        self.n_gru_hidden_units = n_gru_hidden_units
        self.num_class = num_class
        self.timestamp = timestamp
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.global_model = self.model(self.n_gru_hidden_units, self.num_class, self.timestamp).to(self.device)
        self.train_loader = train_loader
        

    def proximal_penalty(self, global_model, local_model, mu):
        # Add the proximal term to the loss
        
        # penalty = 0
        # for local_weights, global_weights in zip(local_model.parameters(), [val.detach().clone() for val in global_model.parameters()]):
        #     penalty += torch.square((local_weights - global_weights).norm(2))
        # return penalty
            
        global_state_dict = global_model.state_dict()
        local_state_dict = local_model.state_dict()

        weight_differences = {k: torch.abs(global_state_dict[k] - local_state_dict[k]) for k in global_state_dict}
        # Compute the L2 norm
        norm_diff = torch.sqrt(sum(torch.sum(w**2) for w in weight_differences.values()))
        return (mu / 2.0) * norm_diff.item()

    
    def train(self, model, device, train_loader, optimizer, criterion):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            # first forward-backward pass
            loss = criterion(model(data), target)  # use this loss for any training statistics
            penalty = self.proximal_penalty(self.global_model, model, self.mu)
            loss += penalty
            loss.backward()
            train_loss += loss.item()
            optimizer.first_step(zero_grad=True)
            
            # second forward-backward pass
            criterion(model(data), target).backward()
            optimizer.second_step(zero_grad=True)

            output = model(data)
        
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
        result = []
        round_accuracy_all = []
        for round in range(num_rounds):
            print(f"---------- Round {round + 1}/{num_rounds} ----------")

            local_model_updates = []
            client_results = {'loss': [], 'accuracy': []}

            for client_id in range(1, len(self.train_loader) + 1):
                print(f"\nTraining on Client {client_id}")

                # Create a local copy
                local_model = self.model(self.n_gru_hidden_units, self.num_class, self.timestamp).to(self.device)
                local_model.load_state_dict(self.global_model.state_dict())
                
                criterion = nn.CrossEntropyLoss()
                base_optimizer = optim.Adam
                optimizer = SAM(local_model.parameters(), base_optimizer=base_optimizer, lr=0.01)

                # Local training
                loss, accuracy = [], []
                for epoch in range(num_epochs):

                    train_loss, train_accuracy = self.train(local_model, self.device, self.train_loader[f'client{client_id}']['train'], optimizer, criterion)
                    val_loss, val_accuracy = self.test(local_model, self.device, self.train_loader[f'client{client_id}']['test'], criterion)
                    
                    loss.append((train_loss, val_loss))
                    accuracy.append((train_accuracy, val_accuracy))
                    print(f'        Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, ', 
                          f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

                
                client_results['loss'].append(loss)
                client_results['accuracy'].append(accuracy)

                local_model_updates.append(local_model.state_dict())
            
            # FedProx
            # Weighted average updating
            averaged_state_dict = {}
            client_size = [len(v['train'].dataset) for k, v in self.train_loader.items()]
            
            for key in self.global_model.state_dict():
                averaged_state_dict[key] = sum(client_size[ind]*update[key] for ind, update in enumerate(local_model_updates)) / np.sum(client_size)
            
            # update global model
            self.global_model.load_state_dict(averaged_state_dict)

            # calculate round test accuracy with weighted mean
            round_acc = []
            client_size = [len(v['test'].dataset) for k, v in self.train_loader.items()]
            for cl, data_ in self.train_loader.items():
                _, val_accuracy = self.test(self.global_model, self.device, data_['test'], criterion)
                round_acc.append(val_accuracy)
            round_accuracy = np.average(round_acc, weights=client_size)
            print(f'\n The round accuracy is: {round_accuracy}')
            round_accuracy_all.append(round_accuracy)
            result.append(client_results)
        
        torch.save(self.global_model.state_dict(), f'./results/FedProx_SAM_model.pth')
        return round_accuracy_all, result
    


class FedProxAlgorithm_FGSM:
    def __init__(self, model, n_gru_hidden_units, num_class, timestamp, train_loader, mu=0.01):
        self.mu = mu
        self.model = model
        self.n_gru_hidden_units = n_gru_hidden_units
        self.num_class = num_class
        self.timestamp = timestamp
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.global_model = self.model(self.n_gru_hidden_units, self.num_class, self.timestamp).to(self.device)
        self.train_loader = train_loader
        

    def proximal_penalty(self, global_model, local_model, mu):
        
        global_state_dict = global_model.state_dict()
        local_state_dict = local_model.state_dict()

        weight_differences = {k: torch.abs(global_state_dict[k] - local_state_dict[k]) for k in global_state_dict}
        # Compute the L2 norm
        norm_diff = torch.sqrt(sum(torch.sum(w**2) for w in weight_differences.values()))
        return (mu / 2.0) * norm_diff.item()
    
    def FGSM_attack(self, data, epsilon, data_grad):
        sign_data_grad = data_grad.sign()
        perturbed_data = data + epsilon * sign_data_grad
        perturbed_data = torch.clamp(perturbed_data, 0, 1)  # Clamp perturbed data to valid range [0, 1]
        return perturbed_data

    
    def train(self, model, device, train_loader, optimizer, criterion):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            data.requires_grad = True 

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            penalty = self.proximal_penalty(self.global_model, model, self.mu)
            loss += penalty
            loss.backward()
            optimizer.step()
            # train_loss += loss.item()

            # FGSM attack
            data_grad = data.grad.data
            perturbed_data = self.FGSM_attack(data, epsilon=0.05, data_grad=data_grad)

            # Update the model using the perturbed data
            optimizer.zero_grad()
            perturbed_output = model(perturbed_data)
            perturbed_loss = criterion(perturbed_output, target)
            penalty = self.proximal_penalty(self.global_model, model, self.mu)
            perturbed_loss += penalty
            train_loss += perturbed_loss.item()
            perturbed_loss.backward()
            optimizer.step()

            _, predicted = torch.max(perturbed_output.data, 1)
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
        result = []
        round_accuracy_all = []
        for round in range(num_rounds):
            print(f"---------- Round {round + 1}/{num_rounds} ----------")

            local_model_updates = []
            client_results = {'loss': [], 'accuracy': []}

            for client_id in range(1, len(self.train_loader) + 1):
                print(f"\nTraining on Client {client_id}")

                # Create a local copy
                local_model = self.model(self.n_gru_hidden_units, self.num_class, self.timestamp).to(self.device)
                local_model.load_state_dict(self.global_model.state_dict())
                
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(local_model.parameters(), lr=0.001)

                # Local training
                loss, accuracy = [], []
                for epoch in range(num_epochs):

                    train_loss, train_accuracy = self.train(local_model, self.device, self.train_loader[f'client{client_id}']['train'], optimizer, criterion)
                    val_loss, val_accuracy = self.test(local_model, self.device, self.train_loader[f'client{client_id}']['test'], criterion)
                    
                    loss.append((train_loss, val_loss))
                    accuracy.append((train_accuracy, val_accuracy))
                    print(f'        Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, ', 
                          f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

                
                client_results['loss'].append(loss)
                client_results['accuracy'].append(accuracy)

                local_model_updates.append(local_model.state_dict())
            
            # FedProx
            # Weighted average updating
            averaged_state_dict = {}
            client_size = [len(v['train'].dataset) for k, v in self.train_loader.items()]
            
            for key in self.global_model.state_dict():
                averaged_state_dict[key] = sum(client_size[ind]*update[key] for ind, update in enumerate(local_model_updates)) / np.sum(client_size)
            
            # update global model
            self.global_model.load_state_dict(averaged_state_dict)

            # calculate round test accuracy with weighted mean
            round_acc = []
            client_size = [len(v['test'].dataset) for k, v in self.train_loader.items()]
            for cl, data_ in self.train_loader.items():
                _, val_accuracy = self.test(self.global_model, self.device, data_['test'], criterion)
                round_acc.append(val_accuracy)
            round_accuracy = np.average(round_acc, weights=client_size)
            print(f'\n The round accuracy is: {round_accuracy}')
            round_accuracy_all.append(round_accuracy)
            result.append(client_results)
        
        torch.save(self.global_model.state_dict(), f'./results/FedProx_FGSM_model.pth')
        return round_accuracy_all, result