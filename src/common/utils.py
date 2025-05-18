import os
import json
import numpy as np

def save_embedding(name, embedding, embedding_path):
    with open(embedding_path, "r") as f:
        data = json.load(f)
    # Ghi đè nếu tên đã tồn tại
    data = [item for item in data if item["name"] != name]
    data.append({"name": name, "embedding": embedding})
    with open(embedding_path, "w") as f:
        json.dump(data, f, indent=2)

def load_embeddings(embedding_path):
    with open(embedding_path, "r") as f:
        return json.load(f)

def cosine_similarity(a, b):
    # Chuyển đổi a nếu là chuỗi hoặc dict
    if isinstance(a, dict):
        a = convert_embedding_str_to_array(a.get('face_embedding', '[]'))
    elif isinstance(a, str):
        a = convert_embedding_str_to_array(a)
    elif isinstance(a, list):
        a = np.array(a, dtype=float)
    elif not isinstance(a, np.ndarray):
        a = np.array([], dtype=float)
    
    # Chuyển đổi b nếu là chuỗi hoặc dict
    if isinstance(b, dict):
        b = convert_embedding_str_to_array(b.get('face_embedding', '[]'))
    elif isinstance(b, str):
        b = convert_embedding_str_to_array(b)
    elif isinstance(b, list):
        b = np.array(b, dtype=float)
    elif not isinstance(b, np.ndarray):
        b = np.array([], dtype=float)
    
    # Kiểm tra mảng rỗng
    if a.size == 0 or b.size == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def convert_embedding_str_to_array(embedding_str):
    embedding_str = embedding_str.replace('[', '').replace(']', '')
    embedding_array = np.fromstring(embedding_str, sep=' ')
    return embedding_array