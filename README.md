# Weather Prediction for Zurich

This repository contains a script for predicting the next 7 days of maximum temperatures for Zurich using a trained transformer-based neural network model. The script fetches historical weather data, processes it for input into the model, and outputs the predicted temperatures.

### Overview

The main script performs the following tasks:

1. **Fetch Historical Weather Data**: Uses the `meteostat` library to get the last 30 days of maximum and minimum temperature data for Zurich.
2. **Process Data**: Cleans the data, handles NaN values, and converts it to a format suitable for model input.
3. **Model Prediction**: Utilizes a trained transformer model to predict the maximum temperatures for the next 7 days.
4. **Output**: Prints the actual temperatures for the last 7 days and the predicted temperatures for the next 7 days.

### Requirements

To run the script, you will need the following libraries:

- `torch`: For building and running the neural network.
- `pandas`: For data manipulation and handling.
- `numpy`: For numerical operations.
- `matplotlib`: For plotting (if needed).
- `meteostat`: For fetching historical weather data.

You can install the required libraries using pip:

```python
pip install torch pandas numpy matplotlib meteostat
```

### Usage
1. Set up the Environment: Ensure you have all the required libraries installed.
2. Download the Script: Clone this repository or download the script directly.
3. Run the Script: Execute the script in your Python environment.
   
### Code Explanation
`Import Libraries`
```python
import torch
from datetime import datetime, timedelta
import pandas as pd
from meteostat import Point, Daily
```

- Imports necessary libraries for tensor operations, date manipulation, data handling, and fetching weather data.

#### Set Location and Time Period

```python
zurich = Point(47.200, 8.3300)
end = datetime.today()
start = end - timedelta(days=30)
```

- Defines the geographical location for Zurich and sets the time period for fetching data.

#### Fetch and Process Data
```python
data = Daily(zurich, start, end)
data = data.fetch()
data = data.dropna(subset=['tmax', 'tmin'])
last_30_days_tmax = data['tmax'].values[-30:]
last_30_days_tmin = data['tmin'].values[-30:]
last_30_days = np.stack((last_30_days_tmax, last_30_days_tmin), axis=1)
input_sequence = torch.tensor(last_30_days, dtype=torch.float32).cuda()
```

- Fetches daily weather data, cleans it by dropping NaN values, and converts the last 30 days of temperature data to a PyTorch tensor.

#### Define Prediction Function
```python
def predict_next_7_days(model, input_sequence, days=7):
    model.eval()
    predictions = []

    for _ in range(days):
        with torch.no_grad():
            output = model(input_sequence.unsqueeze(0).unsqueeze(-1))
            next_day_temp = output.item()
            predictions.append(next_day_temp)
            input_sequence = torch.cat((input_sequence[1:], torch.tensor([next_day_temp], dtype=torch.float32).cuda()))

    return predictions
```

- Defines a function to predict the next 7 days of temperatures using the trained model. It updates the input sequence with each new prediction.

#### Predict and Print Results

```python
predicted_temperatures = predict_next_7_days(model, input_sequence, days=7)
print("Last 7 days max temperatures:", [round(temp * 2) / 2 for temp in last_30_days[-7:]])
print("Predicted max temperatures for the next 7 days:", [round(temp * 2) / 2 for temp in predicted_temperatures])
```

- Calls the prediction function and prints both the actual last 7 days of temperatures and the predicted next 7 days of temperatures.

### Notes
- Ensure that you have a trained model saved as model before running the prediction function.
- The script assumes that the model is compatible with the input shape provided by the input_sequence.
  
### Conclusion
This script provides a comprehensive tool for predicting future temperatures using historical data and a trained transformer model. By following the setup instructions and running the script, you can obtain predictions for the next 7 days of maximum temperatures in Zurich.
