from fileinput import filename
from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import numpy as np
import logging
import os

logging_str = "[%(asctime)s %(levelname)s:%(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log") ,level=logging.INFO, format=logging_str, filemode="a")

def main(data, eta, epochs, filename, plotFileName):

    # creating the dataframe
    df = pd.DataFrame(data)
    logging.info(f"THis is actual dataframe{df}")


    X, y = prepare_data(df)


    model = Perceptron(eta = eta, epochs = epochs)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model, filename=filename)
    save_plot(df, plotFileName, model)

if __name__ == '__main__':
    AND = {
        'x1': [0,0,1,1],
        'x2': [0,1,0,1],
        'y' : [0,0,0,1]
    }

    ETA = 0.3 # 0 and 1
    EPOCHS = 100

    try:
        logging.info("\n>>>>>>>>  STARTING TRAINING >>>>>>>")
        main(data = AND, eta=ETA, epochs=EPOCHS, filename="and.model", plotFileName="and.png")
        logging.info("\n<<<<<<<<  TRAINING DONE SUCCESSFULLY >>>>>>>\n")

    except Exception as e:
        logging.exception(e)
        raise e