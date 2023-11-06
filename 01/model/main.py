from fastai.text.all import *
from pathlib import Path

CSV_PATH = Path("training_data.csv")

dls = TextDataLoaders.from_csv(path=CSV_PATH, text_col=2, label_col=1, valid_pct=0.2)
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
learn.fine_tune(8, 1e-2)
