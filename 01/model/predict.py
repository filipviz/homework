from fastai.text.all import *
import warnings

warnings.filterwarnings('ignore', category=UserWarning, message=".*'has_mps' is deprecated.*") # Filter warnings

path_to_pkl = Path('./model.pkl')
learn = load_learner(path_to_pkl)

# print(learn.dls.vocab)

def predict_text(text, n_preds=5):
    pred, pred_idx, probs = learn.predict(text)
    print(f'Prediction: {pred}; Probability: {(probs[pred_idx]*100.0):.02f}%.')

    runner_ups = probs.argsort(descending=True)[:n_preds]
    for idx in runner_ups:
        if idx != pred_idx and probs[idx] > 0.1:
            print(f"or {learn.dls.vocab[1][idx]} ({(probs[idx]*100.0):.02f}%)")
    print()

print("Enter 'exit' to quit the prediction loop.")
while True:
    text_to_classify = input("Enter the text you want to classify: ")
    if text_to_classify.lower() == 'exit':
        print("Exiting the prediction loop.")
        break

    predict_text(text_to_classify)
