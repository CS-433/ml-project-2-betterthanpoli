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

**comments_long.csv**: This .csv file contains all the comments extracted from the Washington Post's article regarding heat pumps. These comments have been specifically screened and included based on the condition that their length exceeds 20 characters.

**qa_heatpumps_2.csv**: This .csv file contains the synthetic dataset we use for fine-tuning our model. Each row represents a pair of question and answer, following the format: "questions" ; "answer". 

**DatasetChatbot.py**: This file defines a custom dataset class to be used for the training of the chatbot model. It reads data from a CSV file with question and answer columns, tokenizes the text using the specified tokenizer (that will be specified in the "train.py" file), and prepares the data for model input. It also ensures that the input sequences are appropriately formatted for the chatbot model.

**train.py**: The train.py script contains the code for training the GPT-2 model for chatbot purposes. Here's an explanation of the key components:

  - *Data Preparation*: The script loads and preprocesses the chatbot dataset, splitting it into training, validation, and test sets.
  
  - *Model Configuration*: It loads the GPT-2 Large model and tokenizer, adding special tokens as needed.

  - *Training Loop*: The code includes a training loop that fine-tunes the model. It tracks training and validation losses over epochs.
  
  - *Model Saving*: The trained model is saved to the specified path if it produces the lowest validation loss.

**flask_chatbot**: The flask_chatbot folder contains all the necessary code to launch a local instance of the chatbot. The internal folders are organized according to Flask rules.  This structure includes key internal folders:

 - *Statistic Folder*: this is where all the .js and .css files are stored. These files are essential for the aesthetic and functional aspects of the website's user interface.

 - *Templates Folder*: this folder contains the HTML files. These files define the layout and elements of the web pages.
 - *App.py*: this script has a dual function. Firstly, its defines the routes for the front-end version of the bot. Secondly, it handles the internal computation of the chatbot. These computations are based on the 'infer' function.


# Usage Instructions

To successfully use this GPT-2 Chatbot Web Application, please follow these steps:

- **Model State Download**:

  To obtain the required fine-tuned GPT-2 model state, 
  

- **Application Execution**:

  Launch the Flask application by running the file *app.py* in your terminal or command prompt. Once the application starts, it will be accessible through a web browser.

- **Chatbot Interaction**:

  You can interact with the GPT-2 chatbot by navigating to the web interface provided by the application. Enter your text input, and the chatbot will respond with generated text based on your input.
  
  
## Requirements
- torch = 2.0.1
- transformers = 4.35.2
- flask = 2.2.2
- bertopic = 0.15.0
- pandas = 1.5.3
- spacy = 3.7.2
- selenium = 4.16.0
