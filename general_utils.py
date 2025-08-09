import pandas as pd
from sklearn.metrics import cohen_kappa_score
import numpy as np

SCORE_RANGES = {
        1: {'sentence_fluency': (1, 6), 'word_choice': (1, 6), 'conventions': (1, 6),'organization': (1, 6), 
            'content': (1, 6), 'holistic': (2, 12)},
        2: {'sentence_fluency': (1, 6), 'word_choice': (1, 6), 'conventions': (1, 6),'organization': (1, 6), 
            'content': (1, 6), 'holistic': (1, 6)},
        3: {'narrativity': (0, 3), 'language': (0, 3), 'prompt_adherence': (0, 3), 'content': (0, 3), 'holistic': (0, 3)},
        4: {'narrativity': (0, 3), 'language': (0, 3), 'prompt_adherence': (0, 3), 'content': (0, 3), 'holistic': (0, 3)},
        5: {'narrativity': (0, 4), 'language': (0, 4), 'prompt_adherence': (0, 4), 'content': (0, 4), 'holistic': (0, 4)},
        6: {'narrativity': (0, 4), 'language': (0, 4), 'prompt_adherence': (0, 4), 'content': (0, 4), 'holistic': (0, 4)},
        7: {'conventions': (0, 6), 'organization': (0, 6), 'content': (0, 6),'holistic': (0, 30)},
        8: {'sentence_fluency': (2, 12), 'word_choice': (2, 12), 'conventions': (2, 12),'organization': (2, 12), 
            'content': (2, 12), 'holistic': (0, 60)}}

def read_data(path):
    """
    Reads the CSV file and returns a dictionary that has parallel lists of values.

    Parameters:
    - path (str): Path to the CSV file containing the essay data.

    Returns: data_dict (dict): A dictionary that has parallel lists, with the following keys:
        - 'essay_ids': Unique identifiers for each essay
        - 'prompt_ids': Identifiers for the prompt id
        - 'essay_text': Text contents of the essays
        - 'features': The 86 extracted features extracted from the essays
        - 'holistic': Holistic scores
        - 'content': Content scores
        - 'organization': Organization scores
        - 'word_choice': Word choice scores
        - 'sentence_fluency': Sentence fluency scores
        - 'conventions': Conventions scores
        - 'prompt_adherence': Prompt adherence scores
        - 'language': Language scores
        - 'narrativity': Narrativity scores
    """
     
    data = pd.read_csv(path)

    data_dict = {
        'essay_ids': data['essay_id'].values,
        'prompt_ids': data['prompt_id'].values,
        'essay_text': data['essay_text'].values,
        'features': data.iloc[:, 12:].values,
        'holistic':data['holistic'].values,
        'content':data['content'].values,
        'organization':data['organization'].values,
        'word_choice':data['word_choice'].values, 
        'sentence_fluency':data['sentence_fluency'].values, 
        'conventions':data['conventions'].values,
        'prompt_adherence':data['prompt_adherence'].values,
        'language':data['language'].values,
        'narrativity':data['narrativity'].values
    }

    return data_dict

def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculates the Quadratic Weighted Kappa (QWK) score between true labels and predictions using sklearn.

    Parameters:
    - y_true (array-like): The true labels
    - y_pred (array-like): The predicted labels

    Returns:
    - float: The QWK score between y_true and y_pred.
    """
    return cohen_kappa_score(y_true, np.round(y_pred), weights='quadratic')