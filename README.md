# AI-Based Network Intrusion Detection System Using Machine Learning

## How to Run

1. Install Python and VS Code.
2. Open this folder in VS Code.
3. Install required libraries:

```bash
pip install -r requirements.txt
```

4. Download NSL-KDD dataset file `KDDTrain+.txt`.
5. Put it inside the `dataset` folder.
6. Train the model:

```bash
python train_model.py
```

7. Run the web app:

```bash
streamlit run app.py
```

## Project Output
- Intrusion detection prediction
- Accuracy graph
- Confusion matrix
- Streamlit dashboard
