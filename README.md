# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: MOHAMMAD SHAHIL

### Register Number: 212223240044

```python
import torch
import torch.nn as nn  
import matplotlib.pyplot as plt
```

```python
torch.manual_seed(71) 
X=torch.linspace(1,50,50).reshape(-1,1)
e=torch.randint(-8,9,(50,1),dtype=torch.float)
y=2*X+1+e
```

```python
plt.scatter(X, y, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Data for Linear Regression')
plt.show()
```

```python
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear=nn.Linear(in_features,out_features)

    def forward(self, x):
        return self.linear(x)
```

```python
torch.manual_seed(59)
model = Model(1, 1)
```

```python
initial_weight = model.linear.weight.item()
initial_bias = model.linear.bias.item()
print("\nName: Abbu Rehan")
print("Register No: 212223240165")
print(f'Initial Weight: {initial_weight:.8f}, Initial Bias: {initial_bias:.8f}\n')
```

```python
loss_function=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.001)
```

```python
epochs = 100
losses = []

for epoch in range(1, epochs + 1):  
    optimizer.zero_grad()
    y_pred=model(X)
    loss=loss_function(y_pred,y)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

    print(f'epoch: {epoch:2}  loss: {loss.item():10.8f}  '
          f'weight: {model.linear.weight.item():10.8f}  '
          f'bias: {model.linear.bias.item():10.8f}')
```

```python
plt.plot(range(epochs), losses, color='blue')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Loss Curve')
plt.show()
```

```python
final_weight = model.linear.weight.item()
final_bias = model.linear.bias.item()
print("\nName: Abbu Rehan")
print("Register No: 212223240165")
print(f'\nFinal Weight: {final_weight:.8f}, Final Bias: {final_bias:.8f}')
```

```python
x1 = torch.tensor([X.min().item(), X.max().item()]) 
y1 = x1 * final_weight + final_bias 
```

```python
plt.scatter(X, y, label="Original Data")
plt.plot(x1, y1, 'r', label="Best-Fit Line")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trained Model: Best-Fit Line')
plt.legend()
plt.show()
```

### Dataset Information
<img width="1486" height="572" alt="image" src="https://github.com/user-attachments/assets/11bce75f-bd02-40e2-b596-57b7d3dfbdb5" />

### OUTPUT
Training Loss Vs Iteration Plot
<img width="1486" height="564" alt="image" src="https://github.com/user-attachments/assets/cb40f707-aa80-486c-a3c0-eda31129ad90" />

Best Fit line plot
<img width="1485" height="567" alt="image" src="https://github.com/user-attachments/assets/1460ba56-9b2e-4a12-96b8-81ed6bc09e43" />

### New Sample Data Prediction
<img width="1485" height="236" alt="image" src="https://github.com/user-attachments/assets/70439aa9-c953-423f-a23a-eb9d3954c578" />

<img width="1485" height="249" alt="image" src="https://github.com/user-attachments/assets/60f602cf-07d6-45c8-941f-ec15d33e4d52" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
