import numpy as np
import pandas as pd
import torch
import os
from Bio import SeqIO

def one_hot_encode(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = np.zeros((4, len(seq)), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in mapping:
            encoded[mapping[base], i] = 1.0
    return encoded

def compute_prediction(model, input_tensor, label_encoder):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
        probabilities = torch.softmax(output, dim=1)
        confidence = probabilities[0][predicted_class].item()*100

    predicted_label = label_encoder.inverse_transform(predicted_class.cpu().numpy())

    return predicted_label[0], confidence

def get_embeddings(extractor, input_tensor):
    extractor.eval()
    with torch.no_grad():
        embedding = extractor(input_tensor)

    return embedding.cpu().numpy().squeeze()

def extract_fasta(input_file):
    ids = []
    descriptions = []
    sequences = []
    lengths = []

    print("Parsing records from input file...")

    # Use SeqIO.parse() to iterate over each sequence record
    # The input_file can be a file handle from a local file or a web upload
    for record in SeqIO.parse(input_file, "fasta"):
        # record is a SeqRecord object
        
        # Append data to the respective lists
        ids.append(record.id)
        descriptions.append(record.description)
        # It's good practice to convert the Seq object to a string
        sequences.append(str(record.seq))
        lengths.append(len(record.seq))

    print("Finished parsing.")

    # Create a dictionary from the lists
    data = {
        'ID': ids,
        'Description': descriptions,
        'Sequence': sequences,
        'Length (bp)': lengths
    }

    # Create the Pandas DataFrame
    df = pd.DataFrame(data)
    
    # Handle cases where no data was parsed
    if df.empty:
        return df

    # Use a regular expression to extract the organism name (e.g., "Giardia lamblia")
    # This pattern captures the text between the first word (the ID) and the phrase "18S".
    # We use .fillna('') to handle cases where the pattern might not match.
    df['label'] = df['Description'].str.extract(r'^\S+\s(.*?)\s18S.*$')[0].fillna('')

    return df    