from sentence_transformers import SentenceTransformer
import numpy as np
import torch

def encode_using_sbert(texts: list) -> np.array:
    try:
        # Input validation - check for None, empty, or non-string values
        for i, text in enumerate(texts):
            if text is None:
                raise ValueError(f"Text at index {i} is None")
            if not isinstance(text, str):
                raise ValueError(f"Text at index {i} is not a string: {type(text)}")
            if not text.strip():
                raise ValueError(f"Text at index {i} is empty or whitespace only")
        
        # Use GPU if available, otherwise CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {device}")
    
        # Load sentence transformer model
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)

        # Generate embeddings in batches
        embeddings = model.encode(texts, 
                                batch_size=32,         
                                show_progress_bar=True,  
                                convert_to_numpy=True)
        
        # Ensure output length matches input length
        if len(embeddings) != len(texts):
            raise ValueError(f"Length mismatch: input={len(texts)}, output={len(embeddings)}")
        
        if np.isnan(embeddings).any():
            raise ValueError("Embeddings contain NaN values")
        
        empty_vectors = np.sum(embeddings, axis=1) == 0
        if np.sum(empty_vectors) > 0:
            raise ValueError("Embeddings contain empty vectors")
        
        return embeddings
                
    except Exception as e:
        raise e