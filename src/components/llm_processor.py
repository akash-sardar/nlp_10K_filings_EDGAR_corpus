from src.entity.artifact_entity import RetrieverArtifact, GeneratedArtifact

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


class LLM_processer:
   """
   This class processes retrieved document contexts and generates intelligent responses using OpenAI's GPT-4 model.
   
   Attributes:
       retrieved_artifact (RetrieverArtifact): Contains retrieved contexts and queries for processing
   """
   
   def __init__(self, retrieved_artifact: RetrieverArtifact):
       try:
           self.retrieved_artifact = retrieved_artifact                         
       except Exception as e:
           raise(e)


   def build_prompt(self, question: str, context: str, query_type: str = "general") -> str:
       """
       This method creates tailored prompts that guide the LLM to provide appropriate responses
       based on the type of financial query (financial, risk, operational, regulatory, or general).
       
       Args:
           question (str): The user's question about the SEC filing
           context (str): Relevant document context retrieved from the knowledge base
           query_type (str, optional): Classification of the query type.
       
       Returns:
           str: A formatted prompt string optimized for the specific query type
       """
       
       # Base instructions
       base_instruction = """
   You are an expert financial analyst reviewing SEC 10-K filings. 
   Analyze the provided context to answer the specific question below.

   INSTRUCTIONS:
   - Base your answer ONLY on the information provided in the context
   - If the answer is not found in the context, respond with "not found"
   - Be precise and cite specific sections when possible
   - For questions on numerical values and amounts, respond with just the exact amounts with units (millions, billions, etc.)
   - If multiple relevant pieces of information exist, provide all of them
   - The context does not provide information on any question reply with just 'not found'
   """

       # Specialized instructions for different query categories
       query_instructions = {

           "financial": """
   - Focus on financial metrics, dollar amounts, and quantitative data
   - Include timeframes and comparison periods where mentioned
   - Note any significant changes or trends
   - For questions on numerical values and amounts, respond with just the exact amounts with units (millions, billions, etc.). Nothing else is needed.
   """,
           "risk": """
   - Identify specific risk factors and their potential impacts
   - Distinguish between current risks and future uncertainties
   - Note any risk mitigation strategies mentioned
   - For questions on numerical values and amounts, respond with just the exact amounts with units (millions, billions, etc.). Nothing else is needed.
   """,
           "operational": """
   - Focus on business operations, processes, and organizational changes
   - Include geographic locations and business segments
   - Note any operational improvements or challenges
   - For questions on numerical values and amounts, respond with just the exact amounts with units (millions, billions, etc.). Nothing else is needed.
   """,
           "regulatory": """
   - Identify regulatory requirements and compliance matters
   - Note any legal proceedings or regulatory changes
   - Include relevant dates and jurisdictions
   - For questions on numerical values and amounts, respond with just the exact amounts with units (millions, billions, etc.). Nothing else is needed.
   """,
           "general": """
   - Provide comprehensive information relevant to the question
   - Include supporting details and context
   - Note any qualifications or limitations mentioned
   - For questions on numerical values and amounts, respond with just the exact amounts with units (millions, billions, etc.). Nothing else is needed.
   """
       }
       
       # Final prompt
       context_instruction = f"""
   CONTEXT FROM SEC 10-K FILING:
   {context}

   QUESTION: {question}

   ANALYSIS or ANSWER:"""
       
       # Combine all components into a complete prompt
       full_prompt = base_instruction + query_instructions.get(query_type, query_instructions["general"]) + context_instruction
       
       return full_prompt


       
   def generate(self) -> GeneratedArtifact:
       """
       This method implements a Query Classification and Response Generation: Creates tailored responses based on the classification and context
      
       Returns:
           GeneratedArtifact: Contains the generated responses and processed queries with contexts
       
       Raises:
           Exception: If API calls fail or response generation encounters errors
       """
       try:
           # Extract contexts and queries from the retrieval artifact
           all_contexts = self.retrieved_artifact.all_contexts
           queries = self.retrieved_artifact.queries
           
           # Associate each query with its corresponding context
           for index, query in enumerate(queries):
               query["context"] = all_contexts[index]
           
           responses = []

           # Initialize OpenAI client for API interactions
           llm_client = OpenAI()

           # Process each query individually
           for query in queries:
               # Combine retrieved chunks into a single context string
               context = "\n\n".join([row["chunk"] for row in query["context"]])
               question = query["query"]

               # Query Type Classification
               query_type_prompt = f"""
               This is a question on SEC 10-K filing of a company: {question}. 
               As an expert on SEC 10-K forms categorize the question into one of the below categories:
               financial, risk, operation, regulatory and general. No Other categories exist. If it fit into none of them reply 'general'
               Answer with one of the five categories given above only in lower cases.
               Answer:"""
               
               # API call for query classification
               try:
                   response = llm_client.chat.completions.create(
                       model = "gpt-4",
                       messages=[{"role": "system", "content": "You are a financial analyst specializing in SEC filing analysis."},
                                 {"role":"user", "content":query_type_prompt}
                                 ],
                       max_tokens=10,
                   )
                   query_type = response.choices[0].message.content
               except Exception as e:
                   print("API call failed for query type")
                   raise(e)                

               # default to 'general' if invalid
               if query_type not in ["financial", "risk", "operation", "general"]:
                   query_type = "general"

               # Response Generation
               prompt = self.build_prompt(question, context, query_type)

               # API call for final response generation
               try:
                   response = llm_client.chat.completions.create(
                       model="gpt-4",
                       messages=[
                           {"role": "system", "content": "You are a financial analyst specializing in SEC filing analysis."},
                           {"role": "user", "content": prompt}
                       ],
                       temperature=0.1,
                       max_tokens=250,
                   )
                   answer = response.choices[0].message.content
               except Exception as e:
                   print("API call failed")
                   raise(e)

               # default to 'not found'
               if answer:
                   if answer.lower() == "notfound":
                       answer = 'not found'
                   responses.append(answer)
               else:
                   responses.append('not found')
           
           # results into a structured artifact
           generated_artifact = GeneratedArtifact(responses=responses, queries=queries)

           return generated_artifact


       except Exception as e:
           raise(e)