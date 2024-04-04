import torch
import torch.nn as nn

import pandas as pd
import numpy as np
import plotext as plt
import time
import os

from contants import EPOCHS, LEARNING_RATE
from model import TabularModel


def cal_age(dt):
    today = pd.Timestamp.today()
    age = today.year - dt.year
    if dt.month < today.month:
        age -= 1

    return int(age)


# Load the data
df = pd.read_csv("./archive/fraud test.csv")

# Feature engineer age column
df["dob"] = pd.to_datetime(df["dob"])
df["age"] = df["dob"].apply(cal_age)

# Split columns into categorical and continuous
cat_cols = ["category", "gender", "state"]
cont_cols = ["amt", "zip", "lat", "long", "city_pop", "age", "merch_lat", "merch_long"]
y_col = "is_fraud"

# Convert categorical columns to category data type
for col in cat_cols:
    df[col] = df[col].astype("category")

# Stack the data
cats = np.stack([df[col].cat.codes.values for col in cat_cols], 1)
conts = np.stack([df[col].values for col in cont_cols], 1)

# Convert to tensor
cats = torch.tensor(cats, dtype=torch.int64)
conts = torch.tensor(conts, dtype=torch.float)
y = torch.tensor(df[y_col].values).flatten()

# Split into training and test sets
batch_sz = len(df)
test_sz = int(batch_sz * 0.2)
cat_train = cats[: batch_sz - test_sz]
cat_test = cats[batch_sz - test_sz : batch_sz]
con_train = conts[: batch_sz - test_sz]
con_test = conts[batch_sz - test_sz : batch_sz]
y_train = y[: batch_sz - test_sz]
y_test = y[batch_sz - test_sz : batch_sz]


# Create embedding sizes
cat_sz = [len(df[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size + 1) // 2)) for size in cat_sz]

# Define the model
model = TabularModel(emb_szs, conts.shape[1], 2, [200, 100], p=0.4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train the model
epochs = EPOCHS
losses = []
start_time = time.time()
for i in range(epochs):
    i += 1
    y_pred = model(cat_train, con_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss.item())

    if i % 10 == 0:
        print(f"Epoch: {i}\tLoss: {loss.item():.4f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plot the loss
plt.plot(list(range(epochs)), losses, label="Loss", xside="Epochs", yside="Loss")
plt.show()

print(f"Training took {(time.time() - start_time)/60:.2f} minutes")

# Test the model
print("Running the model against test data")
with torch.no_grad():
    y_val = model(cat_test, con_test)
    loss = criterion(y_val, y_test)

print(f"Loss: {loss:.4f}")

correct = 0
rows = len(y_test)
for i in range(rows):
    if y_test[i] == torch.argmax(y_val[i]):
        correct += 1

print(f"The model got {correct}/{rows} correct. Accuracy: {100*correct/rows:.4f}%")


decision = input("Save model?(y/n)")
if decision.lower() == "y":
    model_name = input("Provide a model name: ")
    # Check if the directory exists
    if not os.path.exists("SavedModels"):
        os.makedirs("SavedModels")

    torch.save(model.state_dict(), f"./SavedModels/{model_name}.pt")
    print("Model saved")

print("Exiting...")
