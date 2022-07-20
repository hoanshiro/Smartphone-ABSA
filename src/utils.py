import re
import torch
import numpy as np
import pandas as pd
from transformers import AdamW
from vncorenlp import VnCoreNLP
from emoji import replace_emoji
from src.conf import AbsaConfig
from tqdm import tqdm
import seaborn as sns
import tqdm.notebook as tq
import torch.nn as nn
import matplotlib.pyplot as plt

conf = AbsaConfig()


def display_all_dataframe():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', -1)


def GetNewLabels():
    aspects = ['SCREEN', 'CAMERA', 'FEATURES', 'BATTERY', 'PERFORMANCE', 'STORAGE', 'DESIGN', 'PRICE', 'GENERAL',
               'SER&ACC']
    polarities = ['Positive', 'Neutral', 'Negative']
    new_labels = [f"{aspect}#{polarity}" for aspect in aspects for polarity in polarities]
    return new_labels


def GetStopWords():
    df_stopwords = pd.read_csv(
        'https://raw.githubusercontent.com/stopwords/vietnamese-stopwords/master/vietnamese-stopwords-dash.txt',
        sep='\n', header=None, names=['stopwords'])
    stop_words = set(df_stopwords.stopwords.values)
    return stop_words


def labels2onehot(row, raw_label='label'):
    list_labels = row[raw_label].split(';')[:-1]
    list_processed_labels = [re.sub('[{}]', '', label) for label in list_labels]
    for label in list_processed_labels:
        row[label] = 1
    return row


def DisplayTokenLen(df, tokenizer):
    token_lens = []
    for txt in tqdm(df.tokenize):
        tokens = tokenizer.encode(txt, max_length=128)
        token_lens.append(len(tokens))

    sns.distplot(token_lens)


def PlotTrainingHistory(history):
    plt.rcParams["figure.figsize"] = (10, 7)
    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')
    plt.title('Training history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
    plt.grid()


class TextProcessing:
    def __init__(self):
        self.rdrsegmenter = VnCoreNLP(conf.rdrsegmenter_path, annotators="wseg", max_heap_size='-Xmx500m')

    def ViTokenize(self, text, remove_stopwords=True):
        list_tokens = self.rdrsegmenter.tokenize(text)
        if remove_stopwords:
            stop_words = GetStopWords()
            list_tokens = [text for text in list_tokens[0] if text not in stop_words]
        else:
            list_tokens = list_tokens[0]
        tokenized_text = ' '.join(list_tokens)
        return tokenized_text

    def clean_text(self, text, remove_stopwords=True):
        # Lower
        text = text.lower()
        # Remove all emoji
        text = replace_emoji(text, replace='')
        # Remove all special char
        special_chars = r"[\"#$%&'()*+,.\-\/\\:;<=>@[\]^_`{|}~\n\r\t]"
        text = re.sub(special_chars, " ", text)
        # Vietnamese tokenize
        text = self.ViTokenize(text, remove_stopwords)
        return text


class SupportModel:
    def __init__(self):
        self.device = conf.device

    def loss_fn(self, outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)

    def train_model(self, training_loader, model, optimizer):
        losses = []
        correct_predictions = 0
        num_samples = 0
        # set model to training mode (activate droput, batch norm)
        model.train()
        # initialize the progress bar
        loop = tq.tqdm(enumerate(training_loader), total=len(training_loader),
                       leave=True, colour='steelblue')
        for batch_idx, data in loop:
            ids = data['input_ids'].to(self.device, dtype=torch.long)
            mask = data['attention_mask'].to(self.device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(self.device, dtype=torch.long)
            targets = data['targets'].to(self.device, dtype=torch.float)

            # forward
            outputs = model(ids, mask, token_type_ids)  # (batch,predict)=(32,8)
            loss = self.loss_fn(outputs, targets)
            losses.append(loss.item())
            # training accuracy, apply sigmoid, round (apply thresh 0.5)
            outputs = torch.sigmoid(outputs).cpu().detach().numpy().round()
            targets = targets.cpu().detach().numpy()
            correct_predictions += np.sum(outputs == targets)
            num_samples += targets.size  # total number of elements in the 2D array

            # backward
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # grad descent step
            optimizer.step()

            # Update progress bar
            # loop.set_description(f"")
            # loop.set_postfix(batch_loss=loss)

        # returning: trained model, model accuracy, mean loss
        return model, float(correct_predictions) / num_samples, np.mean(losses)

    def eval_model(self, validation_loader, model):
        losses = []
        correct_predictions = 0
        num_samples = 0
        # set model to eval mode (turn off dropout, fix batch norm)
        model.eval()

        with torch.no_grad():
            for batch_idx, data in enumerate(validation_loader, 0):
                ids = data['input_ids'].to(self.device, dtype=torch.long)
                mask = data['attention_mask'].to(self.device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype=torch.long)
                targets = data['targets'].to(self.device, dtype=torch.float)
                outputs = model(ids, mask, token_type_ids)

                loss = self.loss_fn(outputs, targets)
                losses.append(loss.item())

                # validation accuracy
                # add sigmoid, for the training sigmoid is in BCEWithLogitsLoss
                outputs = torch.sigmoid(outputs).cpu().detach().numpy().round()
                targets = targets.cpu().detach().numpy()
                correct_predictions += np.sum(outputs == targets)
                num_samples += targets.size  # total number of elements in the 2D array

        return float(correct_predictions) / num_samples, np.mean(losses)

    def get_predictions(self, model, data_loader):
        """
        Outputs:
          predictions -
        """
        model = model.eval()

        comments = []
        predictions = []
        prediction_probs = []
        target_values = []

        with torch.no_grad():
            for data in data_loader:
                comment = data["comment"]
                ids = data["input_ids"].to(self.device, dtype=torch.long)
                mask = data["attention_mask"].to(self.device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype=torch.long)
                targets = data["targets"].to(self.device, dtype=torch.float)

                outputs = model(ids, mask, token_type_ids)
                # add sigmoid, for the training sigmoid is in BCEWithLogitsLoss
                outputs = torch.sigmoid(outputs).detach().cpu()
                # thresholding at 0.5
                preds = outputs.round()
                targets = targets.detach().cpu()

                comments.extend(comment)
                predictions.extend(preds)
                prediction_probs.extend(outputs)
                target_values.extend(targets)

        predictions = torch.stack(predictions)
        prediction_probs = torch.stack(prediction_probs)
        target_values = torch.stack(target_values)

        return comments, predictions, prediction_probs, target_values

    def predict_raw_text(self, model, tokenizer, raw_text):
        TexProcesser = TextProcessing()
        cleaned_text = TexProcesser.clean_text(raw_text, remove_stopwords=False)
        #
        encoded_text = tokenizer.encode_plus(
            cleaned_text, max_length=conf.MAX_LEN, add_special_tokens=True,
            return_token_type_ids=True, pad_to_max_length=True,
            return_attention_mask=True, return_tensors='pt',
        )
        #
        input_ids = encoded_text['input_ids'].to(self.device)
        attention_mask = encoded_text['attention_mask'].to(self.device)
        token_type_ids = encoded_text['token_type_ids'].to(self.device)
        output = model(input_ids, attention_mask, token_type_ids)
        # add sigmoid, for the training sigmoid is in BCEWithLogitsLoss
        output = torch.sigmoid(output).detach().cpu()
        # thresholding at 0.5
        output = output.flatten().round().numpy()

        # Correctly identified the topic of the paper: High energy physics
        print(f"Comment: {raw_text}")
        target_list = GetNewLabels()
        for idx, p in enumerate(output):
            if p == 1:
                print(f"Label: {target_list[idx]}")

