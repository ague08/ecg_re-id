# src/evaluation.py
from scipy.spatial.distance import euclidean, cosine

def de_anonymize(test_delta_primes, train_fingerprints, metric='euclidean'):
    """
    De-anonymize users by comparing test delta primes with training fingerprints.
    
    Args:
        test_delta_primes (dict): Deltas for the test set (one per person).
        train_fingerprints (dict): Fingerprints for the training set (one per person).
        metric (str): Distance metric to use ('euclidean' or 'cosine').
    
    Returns:
        dict: A dictionary mapping test user IDs to predicted training user IDs.
    """
    match_results = {}
    
    for test_user_id, delta_prime in test_delta_primes.items():
        best_match = None
        min_distance = float('inf')
        
        for train_user_id, fingerprint in train_fingerprints.items():
            if metric == 'euclidean':
                distance = euclidean(delta_prime, fingerprint)
            elif metric == 'cosine':
                distance = cosine(delta_prime, fingerprint)
            else:
                raise ValueError("Unsupported metric. Choose 'euclidean' or 'cosine'.")
            
            if distance < min_distance:
                min_distance = distance
                best_match = train_user_id
        
        match_results[test_user_id] = best_match
    
    return match_results

def calculate_accuracy(match_results):
    """
    Calculate the accuracy of the de-anonymization process.
    
    Args:
        match_results (dict): A dictionary mapping test user IDs to predicted training user IDs.
    
    Returns:
        float: The accuracy percentage.
    """
    correct_matches = sum(1 for test_user, predicted_user in match_results.items() if test_user == predicted_user)
    total_matches = len(match_results)
    accuracy = (correct_matches / total_matches) * 100 if total_matches > 0 else 0
    return accuracy
