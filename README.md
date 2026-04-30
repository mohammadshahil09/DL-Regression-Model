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

```
class Model(nn.Module):
    def __init__(self, in_features, out_features):  
        super().__init__()      
        self.linear = nn.Linear(in_features, out_features)   
        
    def forward(self, x):    
        y_pred = self.linear(x)
        return y_pred
torch.manual_seed(59)
model = Model(1, 1)
model.linear.weight.item()
model.linear.bias.item()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
epochs = 50 
losses = []

for i in range(epochs):

    i = i +1   
    y_pred = model.forward(X) 
    loss = criterion(y_pred, y)
    losses.append(loss.item())
    print(f'epoch: {i}  loss: {loss.item()}  weight: {model.linear.weight.item()} bias: {model.linear.bias.item()}') 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
plt.plot(range(epochs), losses)

plt.ylabel('MSE Loss')
plt.xlabel('Epoch')
plt.show()
x = np.linspace(0.0,50.0,50)
current_weight = model.linear.weight.item()
current_bias = model.linear.bias.item()

predicted_y = current_weight * x + current_bias
plt.scatter(X, y)

plt.plot(x,predicted_y, 'r')




```

### Dataset Information
<img width="678" height="872" alt="image" src="https://github.com/user-attachments/assets/b53f3bf6-7250-4dd9-bcf2-1bab1c76f867" />


### OUTPUT
Training Loss Vs Iteration Plot
<img width="789" height="562" alt="image" src="https://github.com/user-attachments/assets/dbff2797-79a9-42e2-a19e-49521338a8e5" />




Best Fit line plot

<img width="832" height="532" alt="image" src="https://github.com/user-attachments/assets/7e53fec5-e9dd-4996-86db-47c139be9c12" />





## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
