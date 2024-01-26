<p align="center">
    <img src="static/asset/logo_whosapp.png" width="200" height="200" alt="Whosapp Logo">
</p>

# Whosapp
![UNISA - FIA](https://img.shields.io/badge/unisa-FIA-blue)
![Whatsapp Stilometry](https://img.shields.io/badge/whatsapp_stilometry-lightgreen)

Welcome to Whosapp! 🚀 This project was developed as part of the "Fundamentals of Artificial Intelligence" course at the _University of Salerno_. Our objective is to create a machine learning model capable of identifying the authors of WhatsApp chats. Read on to learn how to use the model! 📱

## Project Structure 🏗️

The project is organized into the following directories:

- `configs/`: Contains configuration files, including:
  - `configs/alias.json`: Alias configuration
  - `configs/config.json`: Feature configuration

- `data/`: Holds project data with subdirectories:
  - `data/raw/`: Raw data storage
  - `data/dataset/`: Dataset used for training
  - `data/wordlist/`: Wordlist used for processing

- `frontend/`: Contains the project's frontend components

- `logs/`: Stores project logs, categorized into sections

- `models/`: Houses the machine learning models of the project

- `src/`: Hosts the source code of the project

## Getting Started 🚀

### Prerequisites

1. Create the following folders:
    - `data/raw/`
    - `configs/`

2. In the `configs/` folder, create the following files:
    - `configs/alias.json` where you will put the alias configuration. The file must be in the following format:
    ```json
    {
        "Username": ["Alias1", "Alias2", "Alias3"],
        "Username2": ["Alias1", "Alias2", "Alias3"]
    }
    ```

### Demo Instructions 🛠️

1. Clone the repository and install the requirements:
   ```bash
   git clone https://github.com/danlig/WhosApp.git
   cd WhosApp
   pip install -r requirements.txt
```

2. Upload the chat you want to analyze in the `data/raw/` folder
3. Run the `py src/pipeline.py` script. For more information, run `py src/pipeline.py -h`.
4. Run the `py src/main.py` script to load and use the model
5. Finally, run `node frontend/index.js` to start the frontend of the project 
   
### Other Useful Commands 🛠️

- `py src/new_dataset.py`: Create a new dataset from the raw data in `data/raw/` and save it in `data/dataset/` in _.parquet_ format
- `py src/new_model.py`: Create a new model from the dataset created with `py src/new_dataset.py` and save it in `models/` in _.joblib_ format.
- `py src/pipeline.py`:  Create a new dataset and model from the raw data in `data/raw/` and save them in `data/dataset/` and `models/` respectively.
- `py src/test_features.py`: test the features that you have configured in the `configs/config.json` file
For more information on how to use these scripts, run `py <script_name>.py -h`

---                                                                          
### Authors
- [Daniele Liguori](@danlig)
- [José Sgariglia](@jose-sgariglia)
- [Luigi Turco](@KronosPNG)


Feel free to adjust the formatting or content as needed!










