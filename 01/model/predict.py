from fastai.text.all import *
import warnings

warnings.filterwarnings('ignore', category=UserWarning, message=".*'has_mps' is deprecated.*") # Filter warnings

path_to_pkl = Path('./model.pkl')
learn = load_learner(path_to_pkl)

def predict_text(text):
    pred, pred_idx, probs = learn.predict(text)
    print(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')

print("Enter 'exit' to quit the prediction loop.")
while True:
    text_to_classify = input("Enter the text you want to classify: ")
    if text_to_classify.lower() == 'exit':
        print("Exiting the prediction loop.")
        break

    predict_text(text_to_classify)
