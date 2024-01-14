!pip install scispacy
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_bc5cdr_md-0.5.3.tar.gz
!pip install pandas --upgrade

import os
import csv
import torch
from collections import Counter
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import spacy
import scispacy
import re
from typing import List


# Task 1: Extracting text from CSV files
def extract_text_from_csvs(folder_path):
    text_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            filepath = os.path.join(folder_path, filename)
            df = pd.read_csv(filepath, engine="python", error_bad_lines=False)
            text_column = df.columns[df.columns.str.contains('TEXT')].tolist()[0]
            raw_text = ' '.join(df[text_column].tolist())

            # Remove non-alphabetic characters and words with less than 4 letters
            cleaned_text = ' '.join(re.findall(r'\b[a-zA-Z]{4,}\b', raw_text))

            text_list.append(cleaned_text)

    with open('combined_text.txt', 'w') as f:
        f.write('\n'.join(text_list))

print('Task 1 complete')



def read_text_in_chunks(file_path: str, window_size: int = 512, overlap: int = 100) -> List[str]:
    with open(file_path, 'r') as f:
        long_text = f.read()
        chunks = [long_text[i:i + window_size] for i in range(0, len(long_text), window_size - overlap)]
        return chunks




print ('chunked')


# Task 3.1: Counting word occurrences
def count_words(text_file):
    word_counts = Counter()
    for chunk in read_text_in_chunks(text_file):
        words = chunk.split()
        word_counts.update(words)

    top_30_words = word_counts.most_common(30)

    with open('top_30_words.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Word', 'Count'])
        writer.writerows(top_30_words)
print ('3.1 done')


# Task 3.2: Counting unique tokens using Transformers
def count_unique_tokens(text_file, model_name="dmis-lab/biobert-v1.1"):
    # Using BioBERT Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize a Counter to keep track of unique tokens
    unique_tokens = Counter()

    # Iterate over text chunks using the read_text_in_chunks function
    for chunk in read_text_in_chunks(text_file):
        # Tokenize the chunk using the Auto Tokenizer
        tokens = tokenizer.tokenize(chunk)

        # Update the Counter with the token occurrences
        unique_tokens.update(tokens)

    # Get the top 30 most common tokens
    top_30_tokens = unique_tokens.most_common(30)

    # Write the results to a CSV file
    with open('top_30_tokens.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['Token', 'Count'])
        # Write the top 30 tokens and their counts
        writer.writerows(top_30_tokens)

print('3.2 done')

# Now call the function with the actual folder path:
folder_path = '/content'
extract_text_from_csvs(folder_path)
print('extract text done')
count_words('combined_text.txt')
print ('top 30 words counted')
count_unique_tokens('combined_text.txt')
print ('tokens counted')

#Task 4 part 1
# Function for Named Entity Recognition using spaCy

# Load spaCy model
nlp = spacy.load("/usr/local/lib/python3.10/dist-packages/en_core_sci_sm/en_core_sci_sm-0.5.3")

# Read the content from combined_text.txt in chunks
text_chunks = read_text_in_chunks('/content/combined_text.txt', window_size=100000, overlap=20000)

# Process each chunk using spaCy
all_entities_spacy = []
for chunk in text_chunks:
    doc_spacy = nlp(chunk)
    entities_spacy = [(ent.text, ent.label_) for ent in doc_spacy.ents]
    all_entities_spacy.extend(entities_spacy)

# Export entities identified by spaCy to a CSV file
with open('spacy_entities.csv', 'w') as spacy_file:
    spacy_file.write('Entity,Label\n')
    for entity in all_entities_spacy:
        spacy_file.write(f'{entity[0]},{entity[1]}\n')

print ('done sci')
#Task 4 part 2
# Function for Named Entity Recognition using spaCy

# Load spaCy model
nlp = spacy.load("/usr/local/lib/python3.10/dist-packages/en_ner_bc5cdr_md/en_ner_bc5cdr_md-0.5.3")

# Read the content from combined_text.txt in chunks
text_chunks = read_text_in_chunks('/content/combined_text.txt', window_size=100000, overlap=20000)

# Process each chunk using spaCy
all_entities_spacy = []
for chunk in text_chunks:
    doc_spacy = nlp(chunk)
    entities_spacy = [(ent.text, ent.label_) for ent in doc_spacy.ents]
    all_entities_spacy.extend(entities_spacy)

# Export entities identified by spaCy to a CSV file
with open('bc5cdr_entities.csv', 'w') as spacy_file:
    spacy_file.write('Entity,Label\n')
    for entity in all_entities_spacy:
        spacy_file.write(f'{entity[0]},{entity[1]}\n')

print ('done bc5')

# Load model directly
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")



# Named-Entity Recognition (NER) using BioBERT
def ner_with_biobert(text_file):
    entities = []

    # Read the content from text_file in chunks
    for chunk in read_text_in_chunks(text_file):
        # Tokenize the chunk using the Auto Tokenizer
        tokens = tokenizer.tokenize(chunk)

        # Convert tokens to IDs
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Convert input_ids to a tensor
        input_ids = torch.tensor([input_ids])

        # Perform inference using the model
        outputs = model(input_ids)

        # Extract entity text and label from the result (this is a placeholder, modify as needed)
        last_hidden_states = outputs.last_hidden_state
        predictions = last_hidden_states.argmax(dim=-1)
        entities.extend([(token, 'DISEASE') for token, label_id in zip(tokens, predictions[0])])

    return entities

# Example: Assuming 'combined_text.txt' is the file containing the text
file_path = 'combined_text.txt'
biobert_entities = ner_with_biobert(file_path)

# Export entities identified by BioBERT to a CSV file
with open('biobert_entities.csv', 'w') as biobert_file:
    biobert_file.write('Entity,Label\n')
    for entity in biobert_entities:
        biobert_file.write(f'{entity[0]},{entity[1]}\n')

print ('done biobert')


#Task 4 part 3
# Function to get most common words
def get_most_common_words(text, top_n=100):
    words = text.split()
    word_counts = Counter(words)
    return dict(word_counts.most_common(top_n))





# Function to get the total entities for each model
def get_total_entities(entity_file):
    entities_df = pd.read_csv(entity_file)
    total_entities = len(entities_df)
    return total_entities

# Now call the function with the actual file paths for each model
total_entities_sci = get_total_entities('spacy_entities.csv')
total_entities_bc5cdr = get_total_entities('bc5cdr_entities.csv')
total_entities_biobert = get_total_entities('biobert_entities.csv')

# Print or use the total_entities variables as needed
print(f'Total Entities (spaCy Sci): {total_entities_sci}')
print(f'Total Entities (spaCy BC5CDR): {total_entities_bc5cdr}')
print(f'Total Entities (BioBERT): {total_entities_biobert}')
# Function to compare most common words
# Function to compare most common words
def compare_most_common_words(model_words, other_model_words, model_name, other_model_name):
    common_words = set(model_words) & set(other_model_words)
    unique_model_words = set(model_words) - set(other_model_words)
    unique_other_model_words = set(other_model_words) - set(model_words)

    # Ensure that all lists have the same length
    max_length = max(len(common_words), len(unique_model_words), len(unique_other_model_words))

    # Pad the lists with empty strings to the same length
    common_words = list(common_words)[:max_length] + [''] * (max_length - len(common_words))
    unique_model_words = list(unique_model_words)[:max_length] + [''] * (max_length - len(unique_model_words))
    unique_other_model_words = list(unique_other_model_words)[:max_length] + [''] * (max_length - len(unique_other_model_words))

    # Create a list for the 'Total Entities' column with the same value for all rows
    total_entities_model = [len(model_words)] * max_length
    total_entities_other_model = [len(other_model_words)] * max_length

    comparison_result = {
        f'{model_name} Unique Words': unique_model_words,
        f'{other_model_name} Unique Words': unique_other_model_words,
        'Common Words': common_words,
        f'Total {model_name} Entities': total_entities_model,
        f'Total {other_model_name} Entities': total_entities_other_model,
    }

    return pd.DataFrame(comparison_result, columns=[f'{model_name} Unique Words', f'{other_model_name} Unique Words', 'Common Words'])

# Load entities detected by spaCy models
spacy_entities_sci = pd.read_csv('spacy_entities.csv')
spacy_entities_bc5cdr = pd.read_csv('bc5cdr_entities.csv')

# Load entities detected by BioBERT
biobert_entities = pd.read_csv('biobert_entities.csv')

# Get most common words for each model
most_common_words_sci = get_most_common_words(' '.join(spacy_entities_sci['Entity']))
most_common_words_bc5cdr = get_most_common_words(' '.join(spacy_entities_bc5cdr['Entity']))
most_common_words_biobert = get_most_common_words(' '.join(biobert_entities['Entity']))

# Compare most common words
comparison_sci_bc5cdr = compare_most_common_words(most_common_words_sci, most_common_words_bc5cdr, 'spaCy (Sci)', 'spaCy (BC5CDR)')
comparison_sci_biobert = compare_most_common_words(most_common_words_sci, most_common_words_biobert, 'spaCy (Sci)', 'BioBERT')
comparison_bc5cdr_biobert = compare_most_common_words(most_common_words_bc5cdr, most_common_words_biobert, 'spaCy (BC5CDR)', 'BioBERT')

# Export comparisons to CSV
comparison_sci_bc5cdr.to_csv('comparison_sci_bc5cdr.csv', index=False)
comparison_sci_biobert.to_csv('comparison_sci_biobert.csv', index=False)
comparison_bc5cdr_biobert.to_csv('comparison_bc5cdr_biobert.csv', index=False)

print('done comparisons')





