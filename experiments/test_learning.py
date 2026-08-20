import torch
import torch.nn as nn

# Training data
x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = torch.tensor([[3.0], [5.0], [7.0], [9.0], [11.0]])

# Tiny neural network
model = nn.Linear(1, 1)

# Loss function
loss_fn = nn.MSELoss()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training
for epoch in range(1000):
    prediction = model(x)
    loss = loss_fn(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.6f}")

# Test
test_x = torch.tensor([[6.0]])
prediction = model(test_x)

print("\nPrediction for x=6:")
print(prediction.item())

print("\nLearned weight:")
print(model.weight.item())

print("Learned bias:")
print(model.bias.item())