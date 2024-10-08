# ISAC-based-gesture-recognition-with-federated-learning

Welcome to the repository for the Decentralized Sensing Framework designed for ISAC-based gesture recognition systems using the real Wi-Fi dataset, WIDAR 3.0.

This project showcases the implementation of several federated learning algorithms, including FedAvg, FedProx, and FedADMM, utilizing the PyTorch framework. These algorithms are key to distributed machine learning, allowing models to be trained across multiple devices while keeping data decentralized.

Additionally, to enhance the robustness of the models, we have incorporated various adversarial defense techniques. These methods include SAM (Sharpness-Aware Minimization), FGSM (Fast Gradient Sign Method), Poisoning Attacks, and Evasion Attacks, among others. These techniques are crucial for improving model resilience against potential adversarial threats.

## Features
- Federated Learning Algorithms: Implementation of **FedAvg, FedProx, and FedADMM.**
- Adversarial Defense Methods: Application of **SAM, FGSM, Poisoning Attacks, Evasion Attacks**, etc.
- Real Wi-Fi Dataset: Utilization of WIDAR 3.0 for gesture recognition.

## The model and Data used in this algorithms:

### Data:
Took into account Sweep, Clap, Slide, and Draw-N datasets, which were distributed in a Non-IID manner across 5 clients, as illustrated in the image below:

![image](https://github.com/user-attachments/assets/6a8ee3a6-8b3c-43ba-aea8-854189870abb)


### Model: 
This model integrates both GRU and CNN layers to handle time series data efficiently. By combining these two powerful architectures, it aims to capture complex temporal patterns and spatial features within the data.


![image](https://github.com/user-attachments/assets/494a499c-dd79-49e4-b872-61d4ab1cb1cf)


Y. Zhu, R. Zhang, Y. Cui, S. Wu, C. Jiang and X. Jing, "Communication-Efficient Personalized Federated Edge Learning for Decentralized Sensing in ISAC," 2023 IEEE International Conference on Communications Workshops (ICC Workshops), Rome, Italy, 2023, pp. 207-212, doi: 10.1109/ICCWorkshops57953.2023.10283763.

You can use your own models and datasets with these algorithms.
