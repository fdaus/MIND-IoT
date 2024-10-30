# MIND-IoT

## Overview

This repository contains several Python scripts designed for MIND-IoT, extracting features from network traffic data, training a tokenizer, pretraining a RoBERTa model, and building a convolutional neural network (CNN) for classification tasks.

### Scripts
1. **feature_nfstream.py**
   - Extracts features from `.pcap` files using the `nfstream` library.
   - Combines data from multiple devices and saves it as CSV files.
  
2. **tokenizing.py**
   - Trains a Byte Pair Encoding (BPE) tokenizer on JSON datasets.
   - Saves the trained tokenizer model for later use.

3. **pretrained.py**
   - Loads a dataset and preprocesses it for training with a RoBERTa model.
   - Sets up the training environment and trains the model.

4. **Mind_CNN_single_task.py**
   - Implements a CNN model using pre-trained RoBERTa embeddings for classification tasks.
   - Contains the architecture and forward pass of the CNN.

## Requirements
- Python 3.9
- Required libraries:
    - torch
    - transformers
    - datasets
    - tokenizers
    - pandas
    - glob
    - os

You can install the required libraries using pip:

```bash
pip install torch transformers datasets tokenizers pandas
```
## Implementation Steps
#### 1. Feature Extraction (`feature_nfstream.py`)
1. **Directory Setup**:
* Create a directory containing subdirectories for each device, with .pcap files inside.

2. **Run the Script**:
* Adjust the inpath and outpath variables in the script to point to your input and output directories respectively.
* Execute the script:
```bash
python feature_nfstream.py
```
3. **Output**:

* The script will generate CSV files for each device, containing extracted features.

#### 2. Tokenization (`tokenizing.py`)
1. **Prepare JSON Data**:
* Place your JSON datasets in the data/json directory.

2. **Run the Script**:
* Execute the script to train the tokenizer:
```bash
python tokenizing.py
```
3. **Output**:
* The trained tokenizer model will be saved in the `models/tokenizer_iot` directory.

#### 3. Pretraining a RoBERTa Model (`pretrained.py`)
1. **Data Setup**:
* Ensure your dataset is in JSON format and placed in `data/json/dataset_balanced.json`.

2. **Run the Script**:
* Execute the pretraining script:
```bash
python pretrained.py
```
3. **Output**:
* The pretrained model will be saved in the `models/pretrained_iot` directory.

#### 4. Model Training with CNN (`Mind_CNN_single_task.py`)
1. **Configure Model**:
* Ensure that the path to the pretrained RoBERTa model in the script is correct.

2. **Run the Script**:
* Execute the CNN training script:
```bash
python Mind_CNN_single_task.py
```
3. **Output**:
* The model will train on the provided data and can be further fine-tuned or evaluated as needed.

## Additional Notes
* Adjust the parameters and paths within the scripts according to your specific dataset and requirements.
* Ensure you have sufficient memory and processing power, especially when dealing with large datasets and training deep learning models.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
