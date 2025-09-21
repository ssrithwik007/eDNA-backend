import base64
import io
from typing import List
import numpy as np
import pandas as pd
import torch
import os
import umap
from Bio import SeqIO
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns

SEQ_LENGTH = 300

def one_hot_encode(seq: str):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    encoded = np.zeros((4, len(seq)), dtype=np.float32)
    for i, base in enumerate(seq.upper()):
        if base in mapping and mapping[base] != 4:
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

def process_sequences(sequences: list[str], feature_extractor, device: str = "cpu"):
    feature_extractor.to(device)
    feature_extractor.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 4, SEQ_LENGTH)
        emb = feature_extractor(dummy)
        print("Embedding shape:", emb.shape)
    padded_sequences = [seq[:SEQ_LENGTH].ljust(SEQ_LENGTH, 'N') for seq in sequences]
    encoded_seqs = [one_hot_encode(seq) for seq in padded_sequences]
    seq_tensor = torch.tensor(np.array(encoded_seqs), dtype=torch.float32).to(device)
    with torch.no_grad():
        raw_embeddings = feature_extractor(seq_tensor).cpu().numpy()
    n_samples = raw_embeddings.shape[0]
    embeddings = raw_embeddings.reshape(n_samples, -1)
    dbscan = DBSCAN(eps=3.5, min_samples=2)
    cluster_labels = dbscan.fit_predict(embeddings)
    results_df = pd.DataFrame({'sequence': sequences, 'cluster_label': cluster_labels})
    return results_df, embeddings

def safe_umap_transform(reducer, emb_sanitized):
        if emb_sanitized.shape[0] == 1:
            # Single sample: compute pairwise distances manually to avoid Numba crash
            distances = pairwise_distances(emb_sanitized, reducer._raw_data, metric=reducer.metric)
            new_coords = reducer._initial_transform(emb_sanitized, distances)
            return new_coords
        else:
            return reducer.transform(emb_sanitized)

def create_umap(
    embeddings: np.ndarray,
    cluster_labels: List[int],
    predictions: List[str],
):
    """
    Generates a single UMAP plot colored by model prediction with outliers highlighted,
    and returns it as a Base64 encoded string.
    """
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)

    # Create the primary visualization DataFrame
    vis_df = pd.DataFrame(data=embeddings_2d, columns=['UMAP 1', 'UMAP 2'])

    # --- Create the new 'clusters' column for coloring ---
    # 1. Start with the model's predictions
    vis_df['cluster_label'] = cluster_labels
    vis_df['clusters'] = predictions

    # 2. Identify outliers from DBSCAN (where label is -1) and overwrite
    #    the prediction label with 'Outlier' for distinct coloring.
    is_outlier = [label == -1 for label in cluster_labels]
    vis_df.loc[is_outlier, 'clusters'] = 'Outlier'

    # 2. Overlay the new data points
    if not vis_df.empty:
        # Separate outliers from clustered new points
        outliers = vis_df[vis_df['cluster_label'] == -1]
        clustered_new = vis_df[vis_df['cluster_label'] != -1]

        # Plot the new points that were clustered
        if not clustered_new.empty:
            sns.scatterplot(
                data=clustered_new, x='UMAP 1', y='UMAP 2',
                hue='clusters',    
                palette='bright',
                marker='o',
                s=100, # Larger size to stand out
            )
        
        # Plot the outliers with a distinct marker
        if not outliers.empty:
            sns.scatterplot(
                data=outliers, x='UMAP 1', y='UMAP 2',
                color='red',
                marker='X', # Use an 'X' for outliers
                s=120, # Make them stand out even more
                label='New Sequence (Outlier)'
            )

    plt.title('UMAP of New Sequences vs. Reference Database')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(title='Predicted Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # --- Convert to Base64 ---
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    base64_img = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{base64_img}"

