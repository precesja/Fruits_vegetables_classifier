# Fruit and Vegetable Image Classifier

This project implements a convolutional neural network (CNN) model for classifying images of fruits and vegetables. It uses TensorFlow/Keras to train the model on a custom dataset and Streamlit for an optional web interface that allows real-time prediction from uploaded images.

---

## Features

- Image classification using a **CNN** trained from scratch.
- Custom dataset with labeled images of fruits and vegetables.
- Training and validation performance visualization (accuracy & loss).
- Exported model in `.keras` format.
- Optional **Streamlit interface** for drag-and-drop image prediction.
- Supports **top-2 prediction ranking** with confidence scores.

---

## Requirements

### Python Dependencies:
Make sure you have Python 3.8+ installed. Required packages include:

- `tensorflow`
- `numpy`
- `matplotlib`
- `Pillow`
- `streamlit`

Install dependencies using:

```bash
pip install tensorflow numpy matplotlib Pillow streamlit
```

## Usage

Model training:

```bash
python train_model.py
```

Interface:

```bash
streamlit run streamlit_app.py
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For any questions or feedback, feel free to open an issue or contact me at **robert.niedziela.96@gmail.com**.
