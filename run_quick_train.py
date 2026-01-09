from app.utils.data_processor import DataProcessor
from app.models.lstm_autoencoder import LSTMAutoencoder
import pandas as pd, numpy as np

# create numeric df
n=200
dates = pd.date_range('2020-01-01', periods=n, freq='H')
df = pd.DataFrame({
    'timestamp': dates,
    'f1': np.random.randn(n),
    'f2': np.random.randn(n),
    'f3': np.random.randn(n)
})
df.set_index('timestamp', inplace=True)

proc = DataProcessor()
processed = proc.preprocess_data(df, {'sequence_length':10, 'validation_split':0.2})
train = processed['train_data']
print('train shape, dtype', train.shape, train.dtype)

model = LSTMAutoencoder(input_shape=processed['input_shape'])
# run a single epoch quick demo
hist = model.train(train, epochs=1, batch_size=8, validation_split=0.1, verbose=1)
print('trained')
