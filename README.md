# Crop Disease Detection using Deep Learning (UN SDG 2: Zero Hunger)

End-Semester Deep Learning Project (`PCCCS582` - Neural Network & Deep Learning).

## Alignment with UN SDG
- **SDG 2 – Zero Hunger:** End hunger, improve food security and nutrition, and promote sustainable agriculture.
- Early automated detection of leaf diseases prevents large-scale crop loss and supports sustainable agricultural yield.

## Project Structure
- `training.ipynb`: Complete PyTorch pipeline (Data downloading via `kagglehub`, preprocessing, custom 3-Block CNN architecture, training/validation loops, accuracy/loss graphs, Precision/Recall/F1-score evaluation, and Confusion Matrix).
- `app.py`: Standalone offline Flask web prototype that accepts a crop leaf image and displays the predicted disease class and confidence score.
- `crop_disease_cnn.pth`: Trained PyTorch CNN model weights.
- `class_names.json`: Target class mappings for the 15 PlantVillage crop disease categories.

## Dataset Attribution
- **Dataset:** [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)

## How to Run the Web Prototype Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt