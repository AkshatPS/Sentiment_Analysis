# Importing necessary libraries
from transformers import Trainer, TrainingArguments, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import googleapiclient.discovery
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import re




# Data Extraction Beginning

# YouTube API setup
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyDJDtDkwtUGW95s5wIJjHnXQLuQ9rE0hIQ"

# Function to extract video id from video link
def extract_video_id(video_link):
    match = re.search(r"watch\?v=([a-zA-Z0-9_-]+)", video_link)
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid YouTube video link")

# Function to clean the video title for proper naming of csv file
def clean_string(input_string):
    cleaned_string = re.sub(r'[^a-zA-Z0-9\s]', '', input_string)
    cleaned_string = re.sub(r'\s', '_', cleaned_string)
    return cleaned_string

# Input video url
video_link = ''

def on_submit():
    global video_link
    user_input = entry.get()
    video_link = user_input
    root_input.destroy()

# Create the main window
root_input = tk.Tk()
root_input.title('Youtube Video Link')

# Set the size of the window
root_input.geometry('500x150')

# Create and place widgets
label = tk.Label(root_input, text='Enter video link:')
label.pack(pady=10)

entry = tk.Entry(root_input,width=80)
entry.pack(pady=10)

submit_button = tk.Button(root_input, text='Submit', command=on_submit)
submit_button.pack(pady=10)

result_label = tk.Label(root_input, text='')
result_label.pack(pady=10)

# Start the Tkinter event loop
root_input.mainloop()

# Extract video id from the link
video_id = extract_video_id(video_link)

# YouTube API request to get video information
youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)
video_request = youtube.videos().list(part="snippet", id=video_id)
video_response = video_request.execute()
video_title = video_response['items'][0]['snippet']['title']
video_title = clean_string(video_title)

# YouTube API request to get video comments
comments_request = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=100)
comments_response = comments_request.execute()

# Convert comments to a DataFrame
comments = []
for item in comments_response['items']:
    comment = item['snippet']['topLevelComment']['snippet']
    comments.append([comment['textDisplay']])
df = pd.DataFrame(comments, columns=['comment_text'])

# Save comments to a CSV file
csv_name_extracted = "D:\VIT\SEMESTER-3\CSE-FDA\Youtube Sentiment analysis project\Scraped Comments"+"\\"+video_title+"_Comments.csv"
df.to_csv(csv_name_extracted, index=False)
# Data Extraction Ends







# Data Cleaning

# Read the extracted data
df_cleaning = pd.read_csv(csv_name_extracted)

# Convert all text to lowercase
df_cleaning['comment_text'] = df_cleaning['comment_text'].str.lower()

# Remove non-alphabetic characters, numbers, and punctuation marks
df_cleaning['comment_text'] = df_cleaning['comment_text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x)))

# Remove multiple spaces
df_cleaning['comment_text'] = df_cleaning['comment_text'].apply(lambda x: re.sub(' +', ' ', x))

# Remove multiple lines in a comment
df_cleaning['comment_text'] = df_cleaning['comment_text'].replace('\n', ' ', regex=True)

#Remove common stopwords
stop_words = set(stopwords.words('english'))
df['comment_text'] = df['comment_text'].apply(lambda x: ' '.join([word for word in word_tokenize(str(x)) if word.lower() not in stop_words]))

# Handle Duplicate elements
df_cleaning.drop_duplicates(subset=['comment_text'], inplace=True)

# Handle missing data
df_cleaning.dropna(subset=['comment_text'], inplace=True)

# Save cleaned data to a CSV file
csv_name_cleaned = "D:\VIT\SEMESTER-3\CSE-FDA\Youtube Sentiment analysis project\Cleaned Data"+"\\"+video_title+"_Comments_Cleaned.csv"

df_cleaning.to_csv(csv_name_cleaned, index=False)
# Data Cleaning Ends




# Predicting from machine learning model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, sentences=None, labels=None):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

        if bool(sentences):
            self.encodings = self.tokenizer(self.sentences,
                                            truncation=True,
                                            padding=True)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        if self.labels == None:
            item['labels'] = None
        else:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.sentences)

    def encode(self, x):
        return self.tokenizer(x, return_tensors='pt').to(DEVICE)

class SentimentModel():

    def __init__(self, model_path):

        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
        args = TrainingArguments(output_dir='D:\\VIT\\SEMESTER-3\\CSE-FDA\\Youtube Sentiment analysis project\\Results', per_device_eval_batch_size=64)
        self.batch_model = Trainer(model=self.model, args=args)
        self.single_dataloader = DataLoader()

    def batch_predict_proba(self, x):

        predictions = self.batch_model.predict(DataLoader(x))
        logits = torch.from_numpy(predictions.predictions)

        if DEVICE == 'cpu':
            proba = torch.nn.functional.softmax(logits, dim=1).detach().numpy()
        else:
            proba = torch.nn.functional.softmax(logits, dim=1).to('cpu').detach().numpy()
        return proba

    def predict_proba(self, x):

        x = self.single_dataloader.encode(x).to(DEVICE)
        predictions = self.model(**x)
        logits = predictions.logits

        if DEVICE == 'cpu':
            proba = torch.nn.functional.softmax(logits, dim=1).detach().numpy()
        else:
            proba = torch.nn.functional.softmax(logits, dim=1).to('cpu').detach().numpy()
        return proba


# Load cleaned comments
df_cleaned_comments = pd.read_csv(csv_name_cleaned)
df_cleaned_comments =df_cleaned_comments.dropna()

# Create sentiment model instance
sentiment_model = SentimentModel('D:\\VIT\\SEMESTER-3\\CSE-FDA\\Youtube Sentiment analysis project\\sentiment_model')

#Extracting cleaned comments for sentiment analysis
batch_sentences = df_cleaned_comments['comment_text'].to_list()

# Predict sentiment for cleaned comments
comment_sentiment = sentiment_model.batch_predict_proba(batch_sentences)

id2label = sentiment_model.model.config.id2label
predicted_class_labels = [id2label[i] for i in np.argmax(comment_sentiment, axis=-1)]

df_cleaned_comments['sentiment'] = predicted_class_labels;

# Save predicted data to a CSV file
csv_name_predicted = "D:\VIT\SEMESTER-3\CSE-FDA\Youtube Sentiment analysis project\Predicted Data"+"\\"+video_title+"_Comment_sentiment_predicted.csv"
df_cleaned_comments.to_csv(csv_name_predicted, index=False)



# Data Visualization with Tkinter
df_visualise = pd.read_csv(csv_name_predicted)

# Set up Tkinter window
root = tk.Tk()
root.title('Youtube Comment Sentiment Analysis Results')

# Generate a pie chart for the 'sentiment' column with a hole in the center
sentiment_counts_pie = df_visualise['sentiment'].value_counts()

# Count occurrences of each unique value in the 'sentiment' column for the bar graph
sentiment_counts_bar = df_visualise['sentiment'].value_counts()

# Set up subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# Plot the pie chart
labels = sentiment_counts_pie.index
sizes = sentiment_counts_pie.values
ax1.pie(sentiment_counts_pie, autopct='%2.2f%%', startangle=140, textprops={'fontsize': 7})
ax1.set_title('Pie Chart of percentage of sentiments')
ax1.axis('equal')
ax1.legend(labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

# Set colors for the bar graph
colors = sns.color_palette("husl", n_colors=len(sentiment_counts_bar))
# Plot the bar graph
sentiment_counts_bar.plot(kind='bar', color=colors, ax=ax2)
ax2.set_title('Bar Graph of number of sentiments')
ax2.set_xlabel('Sentiments')
ax2.set_ylabel('Count')

# Draw a line to separate the two graphs
line_start = (0.553, 1)
line_end = (0.553, 0)
line = plt.Line2D((line_start[0], line_end[0]), (line_start[1], line_end[1]), color="black", linestyle="--", linewidth=2)
fig.add_artist(line)

# Set up Tkinter canvas
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()

# Adjust layout to prevent overlap
plt.tight_layout()

# Add a result label
result_label = tk.Label(root, text="", font=('Helvetica', 15))
max_category = sentiment_counts_bar.idxmax()
result_label.config(text=f"Majority Comments are of the sentiment: {max_category}")
result_label.pack()

# Function to stop code execution on window close
def on_close():
    root.destroy()
    raise SystemExit  # This stops the code execution

# Bind the on_close function to the window close event
root.protocol("WM_DELETE_WINDOW", on_close)

# Run the Tkinter event loop
root.mainloop()

#Data Visualisation ends
