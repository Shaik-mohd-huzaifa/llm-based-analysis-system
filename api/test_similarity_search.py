from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import json
from typing import List, Dict
from pprint import pprint

# Load environment variables
load_dotenv()

def test_similarity_search(subtheme: str, limit: int = 100) -> List[Dict]:
    """
    Test similarity search for a given subtheme.
    
    Args:
        subtheme (str): The subtheme to search for
        limit (int): Maximum number of results to return
    
    Returns:
        List[Dict]: List of relevant reviews with metadata
    """
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # Initialize vectorstore with correct path
    vectorstore = Chroma(
        persist_directory="../jewelry_reviews_vectorstore",  # Updated path
        embedding_function=embeddings
    )
    
    # Create a comprehensive search query
    search_query = f"""
    Find customer reviews about {subtheme}:
    - Direct mentions of {subtheme}
    - Customer experiences related to {subtheme}
    - Specific examples of {subtheme}
    - Impact of {subtheme} on customer satisfaction
    - Quality of {subtheme}
    - Customer feedback regarding {subtheme}
    """
    
    print(f"\nExecuting search for: {subtheme}")
    print("Search query:", search_query)
    
    # Perform similarity search with increased limit
    results = vectorstore.similarity_search_with_score(
        search_query,
        k=limit  # Increased limit to get more results
    )
    
    # Process results
    reviews = []
    seen_texts = set()  # To avoid duplicates
    
    for doc, score in results:
        if doc.page_content in seen_texts:
            continue
            
        # Calculate relevance score (convert distance to similarity)
        relevance_score = 1 - min(score, 1)
        
        # Include all results without filtering by relevance
        review = {
            'text': doc.page_content,
            'relevance_score': relevance_score,
            'metadata': {
                'rating': doc.metadata.get('score_Overall_Rating'),
                'date': doc.metadata.get('date_Date_Created'),
                'city': doc.metadata.get('string_City'),
                'state': doc.metadata.get('string_State')
            }
        }
        reviews.append(review)
        seen_texts.add(doc.page_content)
    
    # Sort by relevance
    reviews.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    return reviews

def main():
    # Test subthemes
    test_cases = [
        "Staff Knowledge and Expertise",
        "Customer Service Quality",
        "Product Selection"
    ]
    
    for subtheme in test_cases:
        print(f"\n{'='*80}")
        print(f"Testing search for: {subtheme}")
        print('='*80)
        
        reviews = test_similarity_search(subtheme)
        
        print(f"\nFound {len(reviews)} reviews")
        print("\nAll reviews by relevance score:")
        
        for i, review in enumerate(reviews, 1):
            print(f"\n{i}. Relevance Score: {review['relevance_score']:.3f}")
            print(f"Rating: {review['metadata']['rating']}")
            print(f"Location: {review['metadata']['city']}, {review['metadata']['state']}")
            print(f"Text: {review['text']}")
            print("-" * 40)

if __name__ == "__main__":
    main() 