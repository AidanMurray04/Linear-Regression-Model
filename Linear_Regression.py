import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

np.random.seed(69)
x_vals = np.linspace(-50,50, 100)
#y_vals = np.array([2,4,6,8,10,12,14,16,18,20])
y_vals = x_vals * 2 + np.random.normal(0,5,100)

x = torch.Tensor(x_vals).unsqueeze(1)
y = torch.Tensor(y_vals).unsqueeze(1)

model = nn.Linear(1, 1)
error = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

plt.figure(figsize=(16,9))
num = 5
while num < 250:
    for epoch in range(num):
        optimizer.zero_grad()
        predictions = model(x)
        loss = error(predictions, y)
        loss.backward()
        optimizer.step()
        #print(loss.item(), end = ' ')
    plt.plot(x_vals, model.weight.item() * x_vals + model.bias.item(), label='predicted line of best fit')
    num += 25

print(f'predicted slope: {model.weight}') #2
print(f'predicted y-int: {model.bias}') #0

plt.plot(x_vals, model.weight.item() * x_vals + model.bias.item(), label = 'predicted line of best fit')
plt.scatter(x=x_vals, y=y_vals, color = 'red', label = 'real data')
plt.show()