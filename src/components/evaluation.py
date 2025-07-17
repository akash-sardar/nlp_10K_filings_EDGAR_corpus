from src.entity.artifact_entity import GeneratedArtifact
from RAG_Validation_Results import VALIDATION_DATA
import pandas as pd
from pathlib import Path
import os
from bert_score import score

class Evaluator:
   def __init__(self, generated_artifact: GeneratedArtifact):
       self.generated_artifact = generated_artifact
       self.validation_path = "RAG_Validation_Results"


   def bert_f1(self, reference, generated):
       _, _, F1 = score([generated], [reference], lang='en')
       return F1.item()
   
   
   def evaluate(self):
       queries = self.generated_artifact.queries
       responses = self.generated_artifact.responses
       
       results = []
       correct_answers = 0
       correct_chunks = 0
       top1_chunks = 0
       bert_scores = []
       
       for i, (query, response) in enumerate(zip(queries, responses)):
           expected = VALIDATION_DATA.validation[i]['expected_value']
           expected_chunk = VALIDATION_DATA.validation[i]['chunk_id']
           
           chunk_ids = [row["chunk_id"] for row in query["context"]]
           scores = [row["score"] for row in query["context"]]
           
           # Evaluating answer for keyword match vs semantic match
           use_bert = len(expected) > 15 and '$' not in expected
           if use_bert:
               bert_f1 = self.bert_f1(expected, response)
               bert_scores.append(bert_f1)
               answer_correct = bert_f1 > 0.8
           else:
               bert_f1 = None
               answer_correct = expected.strip() == response.strip()
           
           # Evaluating chunks
           # Only works if Validation set used same chunked dataset from /Knowledge
           chunk_correct = expected_chunk in chunk_ids if expected_chunk else True
           top1_correct = chunk_ids[0] == expected_chunk if chunk_ids and expected_chunk else False
           
           # Count metrics
           correct_answers += answer_correct
           correct_chunks += chunk_correct
           top1_chunks += top1_correct
           
           # Store results
           results.append({
               'Query': query['query'],
               'Expected_Answer': expected,
               'Generated_Answer': response,
               'Expected_Chunk_ID': expected_chunk,
               'Retrieved_Chunk_IDs': ', '.join(chunk_ids),
               'Answer_Correct': answer_correct,
               'Chunk_Retrieved': chunk_correct,
               'Top1_Chunk_Match': top1_correct,
               'BERT_F1_Score': bert_f1,
               'Avg_Retrieval_Score': f"{sum(scores)/len(scores):.3f}" if scores else "0.000"
           })
           
           # Print details
           print(f"\nQuery: {query['query']}")
           print(f"Expected: {expected}")
           print(f"Generated: {response}")
           print(f"Expected Chunk: {expected_chunk}")
           print(f"Retrieved Chunks: {chunk_ids}")
           print(f"Retrieval Scores: {[f'{s:.3f}' for s in scores]}")
           if bert_f1:
               print(f"BERT F1: {bert_f1:.3f}")
           print(f"Answer Correct: {answer_correct}")
       
       # metrics
       total = len(queries)
       metrics = {
           'Answer_Accuracy': f"{correct_answers/total:.1%}",
           'Chunk_Recall': f"{correct_chunks/total:.1%}",
           'Top1_Chunk_Accuracy': f"{top1_chunks/total:.1%}",
           'Avg_BERT_F1': f"{sum(bert_scores)/len(bert_scores):.3f}" if bert_scores else "N/A",
           'BERT_Pass_Rate': f"{sum(1 for s in bert_scores if s > 0.8)/len(bert_scores):.1%}" if bert_scores else "N/A"
       }
       
       # Print summary
       print(f"\n{'='*50}")
       print("EVALUATION SUMMARY\n")
       print(f"{'='*50}")
       for key, value in metrics.items():
           print(f"{key}: {value}")
       print(f"{'='*50}")
       
       # Save CSV
       os.makedirs(self.validation_path, exist_ok=True)
       pd.DataFrame(results).to_csv(Path(self.validation_path) / "evaluation_results.csv", index=False)
       print(f"\nResults saved to: {self.validation_path}/evaluation_results.csv")