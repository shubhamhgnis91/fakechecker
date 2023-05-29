from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

TOKEN: Final = 'TELEGRAM_API_TOKEN'
BOT_USERNAME: Final = '@TELEGRAM_BOT_USERNAME'


# Read in the dataset of news articles using pandas
df = pd.read_csv('data.csv')

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=7)

# Vectorize the text data using TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
tfidf_test = tfidf_vectorizer.transform(x_test)

# Train and test different classifiers
pac = PassiveAggressiveClassifier(max_iter=1000, C=0.01)
pac.fit(tfidf_train, y_train)
y_pred_pac = pac.predict(tfidf_test)

svm = SVC(kernel='linear')
svm.fit(tfidf_train, y_train)
y_pred_svm = svm.predict(tfidf_test)

nb = MultinomialNB()
nb.fit(tfidf_train, y_train)
y_pred_nb = nb.predict(tfidf_test)

lr = LogisticRegression()
lr.fit(tfidf_train, y_train)
y_pred_lr = lr.predict(tfidf_test)

# Evaluate accuracy of different classifiers
pac_accuracy = accuracy_score(y_test, y_pred_pac)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
nb_accuracy = accuracy_score(y_test, y_pred_nb)
lr_accuracy = accuracy_score(y_test, y_pred_lr)

print(pac_accuracy)
print(svm_accuracy)
print(nb_accuracy)
print(lr_accuracy)


print('Starting up bot...')

#/start command
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hello there! I\'m a Fake news checking bot.')


#/help command
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Type or paste the news message you want to predict whether it is Fake or not!')




def handle_response(text: str) -> str:

    #Transform incoming messaged into a tfidf vector
    input_tfidf = tfidf_vectorizer.transform([text])

    result_message=""

    #Append to result message depending upon the classifications done
    
    if pac.predict(input_tfidf)[0] == 'FAKE':
        result_message+="Passive Aggressive Classifier: FAKE."+"\n"
    else:
        result_message+="Passive Aggressive Classifier: REAL"+"\n"

    if svm.predict(input_tfidf)[0] == 'FAKE':
        result_message+="SVM: FAKE"+"\n"
    else:
        result_message+="SVM: REAL"+"\n"

    if nb.predict(input_tfidf)[0] == 'FAKE':
        result_message+="Naive Bayes: FAKE"+"\n"
    else:
        result_message+="Naive Bayes: REAL"+"\n"

    if lr.predict(input_tfidf)[0] == 'FAKE':
        result_message+="Logistic Regression: FAKE"+"\n"
    else:
        result_message+="Logistic Regression: REAL"+"\n"

    return result_message

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Get basic info of the incoming message
    message_type: str = update.message.chat.type
    text: str = update.message.text

    # Print a log for debugging
    print(f'User ({update.message.chat.id}) in {message_type}: "{text}"')

    # React to group messages only if users mention the bot directly
    if message_type == 'group':
        # Replace with your bot username
        if BOT_USERNAME in text:
            new_text: str = text.replace(BOT_USERNAME, '').strip()
            response: str = handle_response(new_text)
        else:
            return  # We don't want the bot respond if it's not mentioned in the group
    else:
        response: str = handle_response(text)

    # Reply normal if the message is in private
    print('Bot:', response)
    await update.message.reply_text(response)


# Log errors
async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')


# Run the program
app = Application.builder().token(TOKEN).build()

# Commands
app.add_handler(CommandHandler('start', start_command))
app.add_handler(CommandHandler('help', help_command))

# Messages
app.add_handler(MessageHandler(filters.TEXT, handle_message))

# Log all errors
app.add_error_handler(error)

print('Polling...')
# Run the bot
app.run_polling(poll_interval=5)
