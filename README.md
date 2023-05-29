# Fake News Detection Telegram Bot

This is a Telegram bot that detects fake news using machine learning algorithms. It analyzes news articles and predicts whether they are fake or real based on a pre-trained model.

## Prerequisites

Before running the bot, make sure you have the following dependencies installed:

- Python 
- Required Python packages (specified in `requirements.txt`)

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/shubhamhgnis91/fakechecker.git

   ```

2. Install the required Python packages:

    ```shell
    pip install -r requirements.txt

    ```
  
3. Set up your Telegram API token by setting the ```TOKEN``` variable in ```app.py```. This token is required to interact with the Telegram Bot API. You can obtain the token by creating a Telegram bot using the BotFather.

4. Similarly, set up your Telegram Bot Username by setting ```BOT_USERNAME``` variable in ```app.py```. This is the username that you set while creating the bot with BotFather.


## Usage

 1. Run the bot:
    ```shell
    python app.py
    ```

 2. Start a conversation with the bot in your Telegram app. Use the /start command to begin and /help command to see instructions.

 3. Send news messages to the bot, and it will analyze and predict whether they are fake or real.


## Models and Data

The bot uses machine learning algorithms to analyze news articles. It trains on a dataset of news articles labeled as fake or real. The dataset is loaded from a CSV file (data.csv) using pandas.

The following classifiers are trained and evaluated:

1. Passive Aggressive Classifier
2. Support Vector Machine (SVM)
3. Naive Bayes
4. Logistic Regression

The accuracy of each classifier is calculated and printed during execution.

## Contributing

Contributions are welcome! If you want to improve this bot or add new features, feel free to submit a pull request.

  
