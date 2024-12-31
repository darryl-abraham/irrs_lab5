
def compute_similarity(vector1: list, vector2: list):
    # # @TODO: Implement the actual computation of similarity between vector1 and vector2.
    # The current implementation returns a placeholder value of 1. Update this function 
    # to perform the appropriate similarity calculation and return the result.

    # Cosine similarity (transformed to range [0, 1])
    dot_product = sum([a*b for a, b in zip(vector1, vector2)])
    magnitude_a = sum([a**2 for a in vector1])**0.5
    magnitude_b = sum([b**2 for b in vector2])**0.5
    similarity = dot_product / (magnitude_a * magnitude_b)

    return (similarity + 1) / 2
    

if __name__ == "__main__":
    
    vector_a, vector_b = [1, 2, 3, 4], [4, 3, 2, 1]
    compute_similarity(vector_a, vector_b)
    