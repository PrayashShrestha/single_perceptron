from utils.model import Perceptron
form utils.all_utils import prepare_data
import pandas as pd
import numpy as np


AND = {
    'x1': [0,0,1,1],
    'x2': [0,1,0,1],
    'y' : [0,0,0,1]
}

# creating the dataframe
df = pd.DataFrame(OR)
print(df)


X, y = prepare_data(df)

ETA = 0.3 # 0 and 1
EPOCHS = 10

model = Perceptron(eta = ETA, epochs = EPOCHS)
model.fit(X, y)

_ = model.total_loss()