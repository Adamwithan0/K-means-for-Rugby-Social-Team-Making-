import umap as umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import csv
from scipy import spatial
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from collections import defaultdict

def load_players_map(file_path):
    players_map = {}
    with open(file_path, 'r') as csvfile:
        players = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(players)
        for row in players:
            name, paragraph, position = row
            players_map[paragraph] = name + " " + position
    return players_map

def load_data(file_path):
    forward_data = []
    back_data = []
    with open(file_path, 'r') as csvfile:
        players = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(players)
        for row in players:
            name, paragraph, position = row
            if position == 'forward':
                forward_data.append([name + " " + position, paragraph])
            elif position == 'back':
                back_data.append([name + " " + position, paragraph])
    return forward_data, back_data

def compute_embeddings(model, paragraphs):
    return model.encode(paragraphs)

def perform_clustering(embeddings):
    reducer = umap.UMAP()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(embeddings)
    reduced_data = reducer.fit_transform(scaled_data)
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_data)
    return reduced_data, cluster_labels

def calculate_top_matches(players_map, embeddings):
    player_embeddings = {players_map[paragraph]: embedding for paragraph, embedding in embeddings}
    top_matches = {}
    all_personal_pairs = defaultdict(list)
    for player in players_map.values():
        for player1 in players_map.values():
            all_personal_pairs[player].append([spatial.distance.cosine(player_embeddings[player1], player_embeddings[player]), player1])

    for player in players_map.values():
        top_matches[player] = sorted(all_personal_pairs[player], key=lambda x: x[0])

    return top_matches

def plot_clusters(reduced_data, cluster_labels, names):
    x = [row[0] for row in reduced_data]
    y = [row[1] for row in reduced_data]

    plt.figure()
    plt.scatter(x, y, c=cluster_labels, cmap='rainbow', s=60)

    for i, name in enumerate(names):
        plt.annotate(name, (x[i], y[i]), fontsize="6")

    plt.axis('off')
    plt.savefig('equal_teams.png', dpi=800)

def main():
    players_map = load_players_map('test.csv')
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    forward_data, back_data = load_data('test.csv')
    forward_names, forward_paragraphs = zip(*forward_data)
    back_names, back_paragraphs = zip(*back_data)

    forward_embeddings = compute_embeddings(model, forward_paragraphs)
    back_embeddings = compute_embeddings(model, back_paragraphs)

    reduced_forward_data, cluster_labels_forward = perform_clustering(forward_embeddings)
    reduced_back_data, cluster_labels_back = perform_clustering(back_embeddings)

    all_data = np.concatenate((reduced_forward_data, reduced_back_data))
    all_labels = np.concatenate((cluster_labels_forward, cluster_labels_back))

    all_names = list(forward_names) + list(back_names)

    top_matches = calculate_top_matches(players_map, zip(forward_paragraphs + back_paragraphs, forward_embeddings.tolist() + back_embeddings.tolist()))

    for player in players_map.values():
        print(player, "'s top match was ", top_matches[player][-1][-1], "with a match score of ", top_matches[player][-1][0])

    plot_clusters(all_data, all_labels, all_names)

if __name__ == "__main__":
    main()