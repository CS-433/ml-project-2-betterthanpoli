\[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/fEFF99tU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=12899338&assignment_repo_type=AssignmentRepo)

# ML PROJECT 2

# Team
- **Team name**: BetterThanPoli
- **Team member**:
    - Elsa Farinella, SCIPER: 377583
    - Robin Faro, SCIPER: 370950
    - Marco Scialanga, SCIPER: 369469

# Codebase
The project's codebase is structured across the following files and folder:

**web_scraping.py**: This file contains the code to scrape all the comments from the Washington Post's article. After identified all the comments, we applied some manipulations on them in order to separate them and keep only the comments with a lenght greater than 20 character. 

**topic_extraction.py**: This script performs topic modeling on text data related to heat pumps using the BERTopic library. The code is organized in the following way: 

  - *Data Preprocessing*: The code begins by loading and preprocessing data from a CSV file ("comments_long.csv"). It employs spaCy for lemmatization and stopwords removal to prepare the text data for analysis.
  
  - *Topic Modeling*: An instance of the BERTopic model is created and fitted to the preprocessed text data. The "all-MiniLM-L6-v2" embedding model is utilized for indentify and extract topics from the data.


**DatasetChatbot.py**: This file defines a custom dataset class to be used for the training of the chatbot model. It reads data from a CSV file with question and answer columns, tokenizes the text using the specified tokenizer (that will be specified in the "train.py" file), and prepares the data for model input. It also ensures that the input sequences are appropriately formatted for the chatbot model.

**train.py**: The train.py script contains the code for training the GPT-2 model for chatbot purposes. Here's an explanation of the key components:

  - *Data Preparation*: The script loads and preprocesses the chatbot dataset, splitting it into training, validation, and test sets.
  
  - *Model Configuration*: It loads the GPT-2 Large model and tokenizer, adding special tokens as needed.

  - *Training Loop*: The code includes a training loop that fine-tunes the model. It tracks training and validation losses over epochs.
  
  - *Model Saving*: The trained model is saved to the specified path if it produces the lowest validation loss.

**data**: The data folder contains the two dataset we used for our exmperiments. Specifically it contains the following files:
  - *comments_long.csv*: This .csv file contains all the comments extracted from the Washington Post's article regarding heat pumps. These comments have been specifically screened and included based on the condition that their length exceeds 20 characters.

  - *qa_heatpumps_2.csv*: This .csv file contains the synthetic dataset we use for fine-tuning our model. Each row represents a pair of question and answer, following the format: "questions" ; "answer". 

**flask_chatbot**: The flask_chatbot folder contains all the necessary code to launch a local instance of the chatbot. The internal folders are organized according to Flask rules.  This structure includes key internal folders:

 - *Statistic Folder*: this is where all the .js and .css files are stored. These files are essential for the aesthetic and functional aspects of the website's user interface.

 - *Templates Folder*: this folder contains the HTML files. These files define the layout and elements of the web pages.
 - *App.py*: this script has a dual function. Firstly, its defines the routes for the front-end version of the bot. Secondly, it handles the internal computation of the chatbot. These computations are based on the 'infer' function.


# How to reproduce our experiments
In order to reproduce our experiments it is necessary to excute the python script we provided in the following way:
- **web_scraping.py** This program does not require any parameter. It will produce a 'comments_long.csv' file containing the preprocessed comments.
- **topic_extraction.py** Similarly to the previous program, this one does not require the user to insert any parameter. Anyway it relies on the 'comments_long.csv' file we generated before, so it's important to be sure that this file is placed in the 'data' folder as it is in the repository.
- **train.py** This script will reproduce the fine-tuning of our model. Considering it will produce the weights of the model, we ask the user to execute it passing the path of the folder in which he/she wants to save this weights. Please note that the '.pt' file's size will be around 2.88GB, so ensure to have enough available space. Here is an usage example " python train.py 'path/to/save/weights' "

# How to start an interactive session with the bot

To successfully use this GPT-2 Chatbot Web Application, please follow these steps:

- **Model State Download**:

  To obtain the required fine-tuned GPT-2 model state you can download the weights from the following link. Alternatively, it is possible to reproduce them by executing the 'train.py' script previously discussed. Once you have the 'model_state_2_large_v2.pt' file, please ensure it is placed inside the 'flask_chatbot' folder. 
  

- **Application Execution**:

  Launch the Flask application by running the file *app.py* in your terminal or command prompt. Once the application starts, it will be accessible through a web browser.

- **Chatbot Interaction**:

  The application will execute on localhost on the port 5002, hence you can access it just browsing the '127.0.0.1:5002' in your browser. Enter your question, and the chatbot will respond with generated answer based on your input.
  
  
## Requirements
- torch = 2.0.1
- transformers = 4.35.2
- flask = 2.2.2
- bertopic = 0.15.0
- pandas = 1.5.3
- spacy = 3.7.2
- selenium = 4.16.0
