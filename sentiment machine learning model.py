# Importing necessary libraries
import numpy as np
import pandas as pd
from fast_ml.model_development import train_valid_test_split  # Assuming a custom library for train-test split
from transformers import Trainer, TrainingArguments, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch import nn
from torch.nn.functional import softmax
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import datasets  # Assuming the 'datasets' library for metrics loading

# Enables GPU if present, else works on CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device Available: {DEVICE}')

# Read CSV file into a Pandas DataFrame
df = pd.read_csv("D:\VIT\SEMESTER-3\CSE-FDA\Youtube Sentiment analysis project\Trial data\Td new - Copy.csv")
print(df.head())

# Extract relevant columns ('comment_text', 'sentiment') and handle missing values
df_comments = df.loc[:, ['comment_text', 'sentiment']].dropna()

# Label encode the 'sentiment' column and print the unique classes
le = LabelEncoder()
df_comments['sentiment_index'] = le.fit_transform(df_comments['sentiment'])
print(df_comments.head())
print(le.classes_)

# Split the dataset into training, validation, and test sets
(train_texts, train_labels,
 val_texts, val_labels,
 test_texts, test_labels) = train_valid_test_split(df_comments, target='sentiment_index', train_size=0.8, valid_size=0.1,
                                                   test_size=0.1)

# Extract lists of texts and labels for training, validation, and testing
train_texts = train_texts['comment_text'].to_list()
train_labels = train_labels.to_list()
val_texts = val_texts['comment_text'].to_list()
val_labels = val_labels.to_list()
test_texts = test_texts['comment_text'].to_list()
test_labels = test_labels.to_list()

# DataLoader Class for tokenization and encoding
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

# Create DataLoader instances for training, validation, and testing
train_dataset = DataLoader(train_texts, train_labels)
val_dataset = DataLoader(val_texts, val_labels)
test_dataset = DataLoader(test_texts, test_labels)

# Print an example item from the DataLoader
print(train_dataset.__getitem__(0))

# Load metric functions for evaluation
f1 = datasets.load_metric('f1')
accuracy = datasets.load_metric('accuracy')
precision = datasets.load_metric('precision')
recall = datasets.load_metric('recall')

# Function to compute various metrics during evaluation
def compute_metrics(eval_pred):
    metrics_dict = {}
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    metrics_dict.update(f1.compute(predictions=predictions, references=labels, average='macro'))
    metrics_dict.update(accuracy.compute(predictions=predictions, references=labels))
    metrics_dict.update(precision.compute(predictions=predictions, references=labels, average='macro'))
    metrics_dict.update(recall.compute(predictions=predictions, references=labels, average='macro'))
    return metrics_dict

# Mapping labels to indices and vice versa
id2label = {idx: label for idx, label in enumerate(le.classes_)}
label2id = {label: idx for idx, label in enumerate(le.classes_)}

# Configure model using AutoConfig
config = AutoConfig.from_pretrained('distilbert-base-uncased', num_labels=6, id2label=id2label, label2id=label2id)
# Create model using AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_config(config)

# Print configuration and model
print(config)
print(model)

# Training arguments for the Trainer
training_args = TrainingArguments(
    output_dir='D:\\VIT\\SEMESTER-3\\CSE-FDA\\Youtube Sentiment analysis project\\Results',
    num_train_epochs=100,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_steps=100,
    weight_decay=0.05,
    report_to='none',
    evaluation_strategy='steps',
    logging_dir='D:\\VIT\\SEMESTER-3\\CSE-FDA\\Youtube Sentiment analysis project\\Logs',
    logging_steps=50)

# Initialize Trainer with the configured model and training arguments
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics)

# Train the model
trainer.train()

# Perform predictions on the test dataset
eval_results = trainer.predict(test_dataset)

# Print predictions, true labels, and computed metrics
print(eval_results.predictions)
print(eval_results.label_ids)
print(eval_results.metrics)

# Map model config labels to indices and perform softmax on predictions
label2id_mapper = model.config.id2label
proba = softmax(torch.from_numpy(eval_results.predictions))
# Map predicted indices back to labels
pred = [label2id_mapper[i] for i in torch.argmax(proba, dim=-1).numpy()]
# Map true indices back to labels
actual = [label2id_mapper[i] for i in eval_results.label_ids]

# Generate classification report
class_report = classification_report(actual, pred, output_dict=True)
# Convert to DataFrame for better display
pd.DataFrame(class_report)

# Save the trained model
trainer.save_model('D:\\VIT\\SEMESTER-3\\CSE-FDA\\Youtube Sentiment analysis project\\sentiment_model')
