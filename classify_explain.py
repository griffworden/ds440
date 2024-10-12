import pandas as pd
import tensorflow as tf
from transformers import (
    DistilBertTokenizer, TFDistilBertForSequenceClassification,
    RobertaTokenizer, TFRobertaForSequenceClassification,
)
from openai import OpenAI
import re
from Levenshtein import distance as levenshtein_distance
from typing import Dict, Any, Tuple

class CovidNewsClassifier:
    def __init__(self, openai_api_key: str):
        # Initialize models and tokenizers (unchanged)
        self.distilbert_tokenizer = DistilBertTokenizer.from_pretrained('Models/distilbert_finetuned_1')
        self.distilbert_model = TFDistilBertForSequenceClassification.from_pretrained('Models/distilbert_finetuned_1')
        
        self.roberta_tokenizer_v1 = RobertaTokenizer.from_pretrained('Models/roberta_v1_finetuned_1')
        self.roberta_model_v1 = TFRobertaForSequenceClassification.from_pretrained('Models/roberta_v1_finetuned_1')
        
        self.roberta_tokenizer_v2 = RobertaTokenizer.from_pretrained('Models/roberta_v2_finetuned_1')
        self.roberta_model_v2 = TFRobertaForSequenceClassification.from_pretrained('Models/roberta_v2_finetuned_1')
        
        self.openai_client = OpenAI(api_key=openai_api_key)

    def classify_text(self, text: str) -> Tuple[Dict[str, bool], float]:
        # Prepare inputs and get predictions for all models
        distilbert_prediction = self._get_model_prediction(
            text, self.distilbert_tokenizer, self.distilbert_model)
        roberta_v1_prediction = self._get_model_prediction(
            text, self.roberta_tokenizer_v1, self.roberta_model_v1)
        roberta_v2_prediction = self._get_model_prediction(
            text, self.roberta_tokenizer_v2, self.roberta_model_v2)
        
        # Calculate ensemble prediction and confidence
        ensemble_probabilities = (distilbert_prediction[1] + 
                                 roberta_v1_prediction[1] + 
                                 roberta_v2_prediction[1]) / 3
        ensemble_prediction = bool(tf.argmax(ensemble_probabilities, axis=-1).numpy()[0])
        
        predictions = {
            'distilbert': bool(distilbert_prediction[0]),
            'roberta_v1': bool(roberta_v1_prediction[0]),
            'roberta_v2': bool(roberta_v2_prediction[0]),
            'ensemble': ensemble_prediction
        }
        
        confidence = self._calculate_ensemble_confidence(predictions)
        
        return predictions, confidence

    def _get_model_prediction(self, text: str, tokenizer, model) -> Tuple[bool, tf.Tensor]:
        inputs = tokenizer(text, return_tensors="tf", truncation=True, 
                          padding='max_length', max_length=512)
        outputs = model(inputs)
        probabilities = tf.nn.softmax(outputs.logits, axis=-1)
        predicted_class = tf.argmax(probabilities, axis=-1).numpy()[0]
        return bool(predicted_class), probabilities

    def _calculate_ensemble_confidence(self, predictions: Dict[str, bool]) -> float:
        # Exclude the ensemble prediction itself from the confidence calculation
        individual_predictions = [predictions[key] for key in predictions if key != 'ensemble']
        agreeing_models = sum(1 for pred in individual_predictions if pred == predictions['ensemble'])
        return agreeing_models / len(individual_predictions)

    def generate_counterfactual(self, text: str, predictions: Dict[str, bool], confidence: float) -> Dict[str, Any]:
        original_label = 'fake news' if predictions['ensemble'] else 'real news'
        
        # Get latent features (unchanged)
        latent_features = self._get_gpt_response(
            f"Given the classifier's label of '{original_label}' for the following headline, "
            f"identify the key factors (latent features) that likely influenced this classification decision. "
            f"List the latent features as a comma-separated list.\n\n"
            f"Headline: '{text}'\nLabel: {original_label}"
        )
        
        # Get identified words (unchanged)
        identified_words = self._get_gpt_response(
            f"Based on the previously identified latent features, now identify and list "
            f"the specific words in the headline related to these features. "
            f"List the words as a comma-separated list.\n\n"
            f"Latent features: {latent_features}\nHeadline: '{text}'"
        )
        
        # Generate counterfactual (unchanged)
        new_label = 'real news' if original_label == 'fake news' else 'fake news'
        counterfactual = self._get_gpt_response(
            f"Generate an edited version of the original headline to alter the classification "
            f"from '{original_label}' to '{new_label}'.\n\n"
            f"Original headline: '{text}'\n"
            f"Latent features: {latent_features}\n"
            f"Identified words: {identified_words}"
        )
        
        # Clean counterfactual text (unchanged)
        counterfactual = re.sub(r"<new>|</new>|\"", "", counterfactual)
        
        # Classify counterfactual
        counterfactual_predictions, counterfactual_confidence = self.classify_text(counterfactual)
        
        # Get latent features and words for counterfactual
        counterfactual_label = 'fake news' if counterfactual_predictions['ensemble'] else 'real news'
        counterfactual_latent_features = self._get_gpt_response(
            f"Given the classifier's label of '{counterfactual_label}' for the following headline, "
            f"identify the key factors (latent features) that likely influenced this classification decision. "
            f"List the latent features as a comma-separated list.\n\n"
            f"Headline: '{counterfactual}'\nLabel: {counterfactual_label}"
        )
        counterfactual_identified_words = self._get_gpt_response(
            f"Based on the previously identified latent features, now identify and list "
            f"the specific words in the headline related to these features. "
            f"List the words as a comma-separated list.\n\n"
            f"Latent features: {counterfactual_latent_features}\nHeadline: '{counterfactual}'"
        )
        
        # Calculate Levenshtein distance (unchanged)
        levenshtein_dist = levenshtein_distance(text, counterfactual)
        
        return {
            'original_text': text,
            'counterfactual_text': counterfactual,
            'original_latent_features': latent_features,
            'original_identified_words': identified_words,
            'counterfactual_latent_features': counterfactual_latent_features,
            'counterfactual_identified_words': counterfactual_identified_words,
            'levenshtein_distance': levenshtein_dist,
            'original_predictions': predictions,
            'original_confidence': confidence,
            'counterfactual_predictions': counterfactual_predictions,
            'counterfactual_confidence': counterfactual_confidence
        }

    def _get_gpt_response(self, prompt: str) -> str:
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

def process_text(text: str, api_key: str) -> pd.DataFrame:
    classifier = CovidNewsClassifier(api_key)
    predictions, confidence = classifier.classify_text(text)
    result = classifier.generate_counterfactual(text, predictions, confidence)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Text': ['Original', 'Counterfactual'],
        'Content': [result['original_text'], result['counterfactual_text']],
        'Ensemble': [result['original_predictions']['ensemble'], result['counterfactual_predictions']['ensemble']],
        'Confidence': [result['original_confidence'], result['counterfactual_confidence']],
        'DistilBERT': [result['original_predictions']['distilbert'], result['counterfactual_predictions']['distilbert']],
        'RoBERTa v1': [result['original_predictions']['roberta_v1'], result['counterfactual_predictions']['roberta_v1']],
        'RoBERTa v2': [result['original_predictions']['roberta_v2'], result['counterfactual_predictions']['roberta_v2']],
        'Latent Features': [result['original_latent_features'], result['counterfactual_latent_features']],
        'Identified Words': [result['original_identified_words'], result['counterfactual_identified_words']]
    })
    
    return df

if __name__ == "__main__":
    # Example usage
    with open('Data/apiKey.txt', 'r') as file:
        api_key = file.readline().strip()
    
    sample_text = "Stay health from COVID-19 by drinking bleach"
    result_df = process_text(sample_text, api_key)
    print(result_df.to_string(index=False))
