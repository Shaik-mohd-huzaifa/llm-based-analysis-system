from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from dotenv import load_dotenv
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from datetime import datetime
from pathlib import Path
from tabulate import tabulate

# Pydantic models
class ChatRequest(BaseModel):
    query: str
    role: str

class ChatResponse(BaseModel):
    response: str
    metadata: Dict

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Review Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Allow both React and Vite dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize shared resources
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="../jewelry_reviews_vectorstore", embedding_function=embeddings)
llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo")

# Load data and cluster analysis
df = pd.read_csv('../Jewelry Store Google Map Reviews.csv')
df['date_Date Created'] = pd.to_datetime(df['date_Date Created'])

with open('../cluster_analysis.json', 'r') as f:
    cluster_analysis = json.load(f)

# Pydantic models for request/response
class Review(BaseModel):
    text: str
    rating: float
    city: Optional[str] = None
    state: Optional[str] = None
    date: Optional[str] = None

class AnalysisRequest(BaseModel):
    reviews: List[Review]
    n_clusters: Optional[int] = 5

class SubthemeSearchRequest(BaseModel):
    subtheme: str
    cluster_id: int
    limit: Optional[int] = 10

class SubthemeReviewsRequest(BaseModel):
    subtheme: str
    min_rating: Optional[float] = None
    max_rating: Optional[float] = None
    sort_by: Optional[str] = "relevance"  # Options: relevance, rating, date
    limit: Optional[int] = 50

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def get_detailed_cluster_themes(texts: List[str], llm: ChatOpenAI) -> dict:
    """Get detailed themes and sub-themes for a cluster using LLM."""
    # Sample up to 15 texts to get better coverage
    sample_texts = texts[:15] if len(texts) > 15 else texts
    
    prompt = f"""Analyze these customer reviews and extract business-focused themes and subthemes.

Key Requirements:
1. Main theme should be a clear business category (e.g., Customer Service, Product Quality, Store Experience)
2. Subthemes should be specific, actionable insights under that category
3. All themes must come directly from the review text
4. Each subtheme should be something a business can act upon

For example:
Main Theme: "Customer Service Quality"
Subthemes:
- "Staff Knowledge and Expertise"
- "Response Time to Inquiries"
- "Problem Resolution Effectiveness"

Reviews to analyze:
{json.dumps(sample_texts)}

Provide your analysis in this JSON format:
{{
    "main_theme": "string (clear business category)",
    "sub_themes": [
        {{
            "text": "string (specific, actionable finding)",
            "color": "#hex_color (matching the sentiment)",
            "action_item": "string (suggested business action)",
            "impact_level": "string (high/medium/low)",
            "frequency": "string (common/occasional/rare finding)"
        }},
        // 6-8 subthemes total
    ],
    "sentiment": {{
        "type": "string (positive/negative/mixed)",
        "explanation": "string (business impact explanation)",
        "key_drivers": ["string (main factors)"]
    }},
    "theme_color": "#hex_color",
    "business_category": "string (e.g., Operations, Service, Product, Ambiance)",
    "priority_level": "string (high/medium/low)",
    "suggested_focus_areas": ["string (2-3 key areas to focus on)"]
}}
"""
    
    try:
        response = llm.predict(prompt)
        return json.loads(response.strip())
    except Exception as e:
        print(f"Error in theme extraction: {str(e)}")
        return {
            "main_theme": "Theme extraction failed",
            "sub_themes": [
                {
                    "text": "Unknown",
                    "color": "#808080",
                    "action_item": "Review system error",
                    "impact_level": "unknown",
                    "frequency": "unknown"
                }
            ] * 6,
            "sentiment": {
                "type": "unknown",
                "explanation": "Failed to analyze",
                "key_drivers": ["error in analysis"]
            },
            "theme_color": "#808080",
            "business_category": "Unknown",
            "priority_level": "unknown",
            "suggested_focus_areas": ["System needs review"]
        }

def analyze_cluster_statistics(texts: List[str], ratings: List[float]) -> dict:
    """Generate statistical analysis for a cluster."""
    avg_rating = np.mean(ratings) if ratings else 0
    rating_distribution = {
        '1 star': sum(1 for r in ratings if r == 1),
        '2 stars': sum(1 for r in ratings if r == 2),
        '3 stars': sum(1 for r in ratings if r == 3),
        '4 stars': sum(1 for r in ratings if r == 4),
        '5 stars': sum(1 for r in ratings if r == 5)
    }
    
    # Calculate trend indicators
    total_reviews = len(texts)
    positive_reviews = sum(1 for r in ratings if r >= 4)
    negative_reviews = sum(1 for r in ratings if r <= 2)
    
    sentiment_ratio = positive_reviews / total_reviews if total_reviews > 0 else 0
    
    # Calculate average review length and identify detailed reviews
    review_lengths = [len(text.split()) for text in texts]
    avg_review_length = np.mean(review_lengths)
    detailed_reviews = sum(1 for length in review_lengths if length > 50)
    
    return {
        'average_rating': round(avg_rating, 2),
        'rating_distribution': rating_distribution,
        'average_review_length': round(avg_review_length, 2),
        'total_reviews': total_reviews,
        'sentiment_metrics': {
            'positive_ratio': round(sentiment_ratio * 100, 1),
            'detailed_feedback_count': detailed_reviews,
            'rating_trend': 'positive' if sentiment_ratio > 0.7 else 'mixed' if sentiment_ratio > 0.3 else 'negative'
        }
    }

@app.post("/analyze")
async def analyze_reviews(request: AnalysisRequest):
    try:
        # Prepare data
        texts = [review.text for review in request.reviews]
        ratings = [review.rating for review in request.reviews]

        # Generate embeddings
        embedding_vectors = embeddings.embed_documents(texts)
        embedding_array = np.array(embedding_vectors)

        # Standardize features
        scaler = StandardScaler()
        embedding_array_scaled = scaler.fit_transform(embedding_array)

        # Perform clustering
        kmeans = KMeans(n_clusters=request.n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embedding_array_scaled)

        # Generate analysis for each cluster
        cluster_analysis = {}
        for i in range(request.n_clusters):
            cluster_indices = [idx for idx, label in enumerate(cluster_labels) if label == i]
            cluster_texts = [texts[idx] for idx in cluster_indices]
            cluster_ratings = [ratings[idx] for idx in cluster_indices]
            
            themes = get_detailed_cluster_themes(cluster_texts, llm)
            stats = analyze_cluster_statistics(cluster_texts, cluster_ratings)
            
            cluster_analysis[i] = {
                'themes': themes,
                'statistics': stats,
                'review_indices': cluster_indices,
                'texts': cluster_texts,
                'ratings': cluster_ratings
            }

        return {
            "status": "success",
            "cluster_analysis": cluster_analysis
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_subtheme")
async def search_subtheme(request: SubthemeSearchRequest):
    try:
        # Create a more comprehensive search query
        search_query = f"""
        Find reviews that discuss or relate to: {request.subtheme}
        Consider:
        - Direct mentions and variations
        - Customer experiences and outcomes
        - Specific examples and situations
        - Impact on business performance
        - Customer satisfaction indicators
        """

        # Get all reviews for the cluster
        all_results = vectorstore.similarity_search_with_score(
            search_query,
            k=100,
            filter={"cluster_id": request.cluster_id}
        )
        
        def highlight_text(text, subtheme, phrases_to_highlight):
            """Add HTML tags to highlight matching phrases in text."""
            text_lower = text.lower()
            highlighted_text = text
            
            # Create variations of the subtheme
            subtheme_variations = [
                subtheme,
                *[word for word in subtheme.lower().split() if len(word) > 3],
                *phrases_to_highlight
            ]
            
            # Sort by length (longest first) to avoid nested highlights
            subtheme_variations.sort(key=len, reverse=True)
            
            for phrase in subtheme_variations:
                phrase_lower = phrase.lower()
                start_idx = text_lower.find(phrase_lower)
                
                if start_idx != -1:
                    original_text = text[start_idx:start_idx + len(phrase)]
                    highlighted_text = highlighted_text.replace(
                        original_text,
                        f"<mark style='background-color: #fff3cd'>{original_text}</mark>"
                    )
            
            return highlighted_text

        def calculate_relevance(text, subtheme):
            """Calculate detailed relevance metrics for a review."""
            text_lower = text.lower()
            subtheme_lower = subtheme.lower()
            
            # Split into words and phrases
            subtheme_words = set(subtheme_lower.split())
            text_words = set(text_lower.split())
            
            # Generate phrase variations
            subtheme_phrases = [
                subtheme_lower,
                *[w for w in subtheme_words if len(w) > 3],
                # Add business-focused variations
                *[f"{w}ing" for w in subtheme_words if len(w) > 3],
                *[f"{w}ed" for w in subtheme_words if len(w) > 3],
                *[f"{w}s" for w in subtheme_words if len(w) > 3]
            ]
            
            # Calculate matches
            direct_matches = sum(1 for phrase in subtheme_phrases if phrase in text_lower)
            word_overlap = len(subtheme_words.intersection(text_words))
            word_match_score = word_overlap / len(subtheme_words) if subtheme_words else 0
            
            # Calculate context relevance
            context_phrases = [
                "because",
                "when",
                "after",
                "during",
                "experience",
                "customer",
                "service",
                "quality"
            ]
            context_score = sum(1 for phrase in context_phrases if phrase in text_lower) / len(context_phrases)
            
            return {
                'direct_matches': direct_matches,
                'word_match_score': word_match_score,
                'context_score': context_score,
                'matching_phrases': [p for p in subtheme_phrases if p in text_lower],
                'has_context': context_score > 0.2
            }

        # Process and enhance results
        processed_reviews = []
        seen_texts = set()
        
        for doc, semantic_score in all_results:
            if doc.page_content in seen_texts:
                continue
                
            # Calculate relevance metrics
            relevance = calculate_relevance(doc.page_content, request.subtheme)
            
            # Calculate combined score with business focus
            semantic_similarity = 1 - min(semantic_score, 1)
            combined_score = (
                semantic_similarity * 0.3 +  # Semantic relevance
                (relevance['word_match_score'] * 0.3) +  # Direct content match
                (min(relevance['direct_matches'], 3) / 3 * 0.2) +  # Specific mentions
                (relevance['context_score'] * 0.2)  # Business context
            )
            
            # Only include if sufficiently relevant
            if combined_score > 0.2 or relevance['direct_matches'] > 0:
                # Highlight matching text
                highlighted_text = highlight_text(
                    doc.page_content,
                    request.subtheme,
                    relevance['matching_phrases']
                )
                
                processed_reviews.append({
                    "text": doc.page_content,
                    "highlighted_text": highlighted_text,
                    "metadata": doc.metadata,
                    "relevance_score": combined_score,
                    "relevance_details": {
                        "semantic_score": semantic_similarity,
                        "word_match_score": relevance['word_match_score'],
                        "direct_matches": relevance['direct_matches'],
                        "matching_phrases": relevance['matching_phrases'],
                        "has_business_context": relevance['has_context']
                    }
                })
                seen_texts.add(doc.page_content)
        
        # Sort by relevance
        processed_reviews.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Group reviews by relevance with business focus
        grouped_reviews = {
            "highly_relevant": [],  # Clear examples with business impact
            "moderately_relevant": [],  # Related experiences
            "somewhat_relevant": []  # Indirect mentions
        }
        
        for review in processed_reviews:
            if review["relevance_score"] > 0.7 or (
                review["relevance_score"] > 0.5 and 
                review["relevance_details"]["has_business_context"]
            ):
                grouped_reviews["highly_relevant"].append(review)
            elif review["relevance_score"] > 0.4:
                grouped_reviews["moderately_relevant"].append(review)
            else:
                grouped_reviews["somewhat_relevant"].append(review)

        return {
            "status": "success",
            "grouped_reviews": grouped_reviews,
            "total_found": len(processed_reviews),
            "subtheme": request.subtheme,
            "relevance_stats": {
                "highly_relevant": len(grouped_reviews["highly_relevant"]),
                "moderately_relevant": len(grouped_reviews["moderately_relevant"]),
                "somewhat_relevant": len(grouped_reviews["somewhat_relevant"])
            },
            "business_metrics": {
                "impact_score": sum(r["relevance_score"] for r in grouped_reviews["highly_relevant"]) / len(grouped_reviews["highly_relevant"]) if grouped_reviews["highly_relevant"] else 0,
                "context_coverage": sum(1 for r in processed_reviews if r["relevance_details"]["has_business_context"]) / len(processed_reviews) if processed_reviews else 0,
                "average_rating": np.mean([r["metadata"]["score_Overall_Rating"] for r in processed_reviews]) if processed_reviews else 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reviews/subtheme")
async def get_subtheme_reviews(request: SubthemeReviewsRequest):
    """Get reviews related to a specific subtheme with filtering and sorting options."""
    try:
        print(f"\nProcessing request for subtheme: {request.subtheme}")
        print(f"Min rating: {request.min_rating}")
        print(f"Max rating: {request.max_rating}")
        print(f"Sort by: {request.sort_by}")
        print(f"Limit: {request.limit}")
        
        # Create a comprehensive search query for better semantic matching
        search_query = f"""
        Find customer reviews about {request.subtheme}:
        - Direct mentions of {request.subtheme}
        - Customer experiences related to {request.subtheme}
        - Specific examples of {request.subtheme}
        - Impact of {request.subtheme} on customer satisfaction
        - Quality of {request.subtheme}
        - Customer feedback regarding {request.subtheme}
        """
        
        print("\nExecuting search with query:", search_query)

        # Get relevant reviews using similarity search with increased limit
        results = vectorstore.similarity_search_with_score(
            search_query,
            k=100  # Increased limit to get more results
        )
        
        print(f"\nFound {len(results)} initial results")
        
        # Process results
        reviews = []
        seen_texts = set()  # To avoid duplicates
        
        for doc, score in results:
            if doc.page_content in seen_texts:
                continue
                
            # Calculate relevance score (convert distance to similarity)
            relevance_score = 1 - min(score, 1)
            
            # Get rating for filtering
            rating = doc.metadata.get('score_Overall_Rating')
            
            print(f"\nProcessing review:")
            print(f"Relevance score: {relevance_score:.3f}")
            print(f"Rating: {rating}")
            print(f"Text preview: {doc.page_content[:100]}...")
            
            # Apply rating filters if specified
            if request.min_rating is not None and (rating is None or rating < request.min_rating):
                print("Skipping due to min_rating filter")
                continue
            if request.max_rating is not None and (rating is None or rating > request.max_rating):
                print("Skipping due to max_rating filter")
                continue
            
            # Include all results without filtering by relevance score
            review = {
                'text': doc.page_content,
                'relevance_score': relevance_score,
                'metadata': {
                    'rating': rating,
                    'date': doc.metadata.get('date_Date_Created'),
                    'city': doc.metadata.get('string_City'),
                    'state': doc.metadata.get('string_State'),
                    'helpful_count': doc.metadata.get('People_Found_Review_Helpful', 0)
                }
            }
            reviews.append(review)
            seen_texts.add(doc.page_content)
        
        print(f"\nProcessed {len(reviews)} reviews after filtering")
        
        # Sort reviews based on the specified criteria
        if request.sort_by == "rating":
            reviews.sort(key=lambda x: (x['metadata']['rating'] or 0, x['relevance_score']), reverse=True)
        elif request.sort_by == "date":
            reviews.sort(key=lambda x: (x['metadata']['date'] or '', x['relevance_score']), reverse=True)
        else:  # default to relevance
            reviews.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Apply limit after sorting
        reviews = reviews[:request.limit]
        
        print(f"\nReturning {len(reviews)} reviews after applying limit")
        
        # Calculate statistics for the filtered reviews
        ratings = [r['metadata']['rating'] for r in reviews if r['metadata']['rating'] is not None]
        stats = {
            'total_reviews': len(reviews),
            'average_rating': round(np.mean(ratings), 2) if ratings else 0,
            'rating_distribution': {
                '5_star': sum(1 for r in ratings if r == 5),
                '4_star': sum(1 for r in ratings if r == 4),
                '3_star': sum(1 for r in ratings if r == 3),
                '2_star': sum(1 for r in ratings if r == 2),
                '1_star': sum(1 for r in ratings if r == 1)
            },
            'relevance_metrics': {
                'high_relevance': sum(1 for r in reviews if r['relevance_score'] > 0.7),
                'medium_relevance': sum(1 for r in reviews if 0.4 <= r['relevance_score'] <= 0.7),
                'low_relevance': sum(1 for r in reviews if r['relevance_score'] < 0.4)
            }
        }
        
        return {
            "status": "success",
            "subtheme": request.subtheme,
            "reviews": reviews,
            "statistics": stats,
            "filters_applied": {
                "min_rating": request.min_rating,
                "max_rating": request.max_rating,
                "sort_by": request.sort_by,
                "limit": request.limit
            }
        }
    except Exception as e:
        print(f"\nError in get_subtheme_reviews: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/themes")
async def get_themes():
    """Get all themes and their subthemes from the cluster analysis."""
    try:
        # Load the cluster analysis from the JSON file in root directory
        with open('../cluster_analysis.json', 'r') as f:
            cluster_data = json.load(f)
        
        # Transform the data into a more usable format
        themes = []
        for cluster_id, data in cluster_data.items():
            themes_data = data.get('themes', {})
            stats_data = data.get('statistics', {})
            
            # Handle subthemes safely
            subthemes = []
            for subtheme in themes_data.get('sub_themes', []):
                if isinstance(subtheme, dict):
                    subthemes.append({
                        'text': subtheme.get('text', 'Unknown'),
                        'impact_level': subtheme.get('impact_level', 'Unknown'),
                        'frequency': subtheme.get('frequency', 'Unknown'),
                        'action_item': subtheme.get('action_item', 'No action specified')
                    })
                else:
                    # Handle case where subtheme is a string
                    subthemes.append({
                        'text': subtheme,
                        'impact_level': 'Unknown',
                        'frequency': 'Unknown',
                        'action_item': 'No action specified'
                    })

            theme_info = {
                'id': int(cluster_id),
                'main_theme': themes_data.get('main_theme', 'Unknown Theme'),
                'sentiment': themes_data.get('sentiment', {}).get('type', 'Unknown'),
                'sentiment_explanation': themes_data.get('sentiment', {}).get('explanation', ''),
                'subthemes': subthemes,
                'statistics': {
                    'average_rating': stats_data.get('average_rating', 0),
                    'total_reviews': stats_data.get('total_reviews', 0),
                    'rating_distribution': stats_data.get('rating_distribution', {}),
                    'sentiment_metrics': stats_data.get('sentiment_metrics', {})
                }
            }
            themes.append(theme_info)
        
        return {
            "status": "success",
            "themes": themes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reviews/by-subtheme/{subtheme}")
async def get_reviews_by_subtheme(subtheme: str, limit: int = 50):
    """Get reviews related to a specific subtheme."""
    try:
        # Create a comprehensive search query
        search_query = f"""
        Find reviews that discuss or relate to: {subtheme}
        Consider:
        - Direct mentions and variations
        - Customer experiences and outcomes
        - Specific examples and situations
        """

        # Search for relevant reviews
        results = vectorstore.similarity_search_with_score(
            search_query,
            k=limit
        )
        
        # Process and format the results
        reviews = []
        for doc, score in results:
            review = {
                'text': doc.page_content,
                'relevance_score': 1 - min(score, 1),  # Convert distance to similarity score
                'metadata': {
                    'rating': doc.metadata.get('score_Overall_Rating'),
                    'date': doc.metadata.get('date_Date_Created'),
                    'city': doc.metadata.get('string_City'),
                    'state': doc.metadata.get('string_State')
                }
            }
            reviews.append(review)
        
        # Sort by relevance score
        reviews.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return {
            "status": "success",
            "subtheme": subtheme,
            "total_reviews": len(reviews),
            "reviews": reviews
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def preprocess_text(text: str) -> str:
    """Preprocess text for keyword analysis."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return ' '.join(tokens)

def extract_keywords(texts: List[str], top_n: int = 40) -> List[Dict]:
    """Extract important keywords from a list of texts."""
    # Preprocess texts
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=100,
        min_df=2,
        max_df=0.95
    )
    
    # Fit and transform the texts
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    
    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Calculate average TF-IDF scores for each word
    avg_tfidf_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
    
    # Create word-score pairs and sort by score
    word_scores = list(zip(feature_names, avg_tfidf_scores))
    word_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N keywords with their scores
    top_keywords = []
    for word, score in word_scores[:top_n]:
        # Count occurrences in original texts
        occurrence_count = sum(1 for text in texts if word in text.lower())
        frequency_percentage = (occurrence_count / len(texts)) * 100
        
        keyword_info = {
            'word': word,
            'importance_score': float(score),
            'occurrence_count': occurrence_count,
            'frequency_percentage': round(frequency_percentage, 2)
        }
        top_keywords.append(keyword_info)
    
    return top_keywords

@app.get("/keywords")
async def get_keywords():
    """Get important keywords from all reviews."""
    try:
        # Load the cluster analysis to get all reviews
        with open('../cluster_analysis.json', 'r') as f:
            cluster_data = json.load(f)
        
        # Collect all review texts
        all_texts = []
        for cluster_info in cluster_data.values():
            if 'texts' in cluster_info:
                all_texts.extend(cluster_info['texts'])
        
        # Extract keywords
        keywords = extract_keywords(all_texts)
        
        # Group keywords by importance
        grouped_keywords = {
            'high_importance': [],
            'medium_importance': [],
            'low_importance': []
        }
        
        for keyword in keywords:
            if keyword['importance_score'] > 0.1:
                grouped_keywords['high_importance'].append(keyword)
            elif keyword['importance_score'] > 0.05:
                grouped_keywords['medium_importance'].append(keyword)
            else:
                grouped_keywords['low_importance'].append(keyword)
        
        return {
            "status": "success",
            "total_reviews_analyzed": len(all_texts),
            "keywords": keywords,
            "grouped_keywords": grouped_keywords
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def format_executive_response(results: Dict) -> str:
    """Format response for executives - high level, action-oriented."""
    response = "📊 Executive Summary\n\n"
    
    # Key Findings
    response += "🎯 Key Findings:\n"
    for finding in results.get('key_findings', [])[:3]:
        response += f"• {finding}\n"
    
    # Critical Actions
    response += "\n⚡ Critical Actions:\n"
    for action in results.get('recommended_actions', [])[:3]:
        response += f"• {action}\n"
    
    # Business Impact
    if 'impact' in results:
        response += f"\n💼 Business Impact:\n{results['impact']}\n"
    
    return response

def format_analyst_response(results: Dict) -> str:
    """Format response for analysts - detailed, data-driven."""
    response = "📈 Detailed Analysis\n\n"
    
    # Statistical Overview
    if 'statistics' in results:
        response += "📊 Statistical Overview:\n"
        response += tabulate(
            results['statistics'],
            headers='keys',
            tablefmt='grid'
        ) + "\n\n"
    
    # Trend Analysis
    if 'trends' in results:
        response += "📈 Trend Analysis:\n"
        for trend in results['trends']:
            response += f"• {trend}\n"
    
    # Detailed Findings
    response += "\n🔍 Detailed Findings:\n"
    for finding in results.get('key_findings', []):
        response += f"• {finding}\n"
    
    return response

def format_store_rep_response(results: Dict) -> str:
    """Format response for store representatives - actionable, customer-focused."""
    response = "🏪 Store Insights\n\n"
    
    # Customer Experience Highlights
    response += "👥 Customer Experience:\n"
    for highlight in results.get('customer_highlights', []):
        response += f"• {highlight}\n"
    
    # Action Items
    response += "\n✅ Action Items:\n"
    for action in results.get('recommended_actions', []):
        response += f"• {action}\n"
    
    # Quick Tips
    if 'quick_tips' in results:
        response += "\n💡 Quick Tips:\n"
        for tip in results['quick_tips']:
            response += f"• {tip}\n"
    
    return response

def analyze_monthly_issues(month: int, year: int) -> Dict:
    """Analyze customer issues for a specific month."""
    # Filter reviews for the specified month
    month_mask = (df['date_Date Created'].dt.month == month) & \
                (df['date_Date Created'].dt.year == year)
    month_reviews = df[month_mask]
    
    # Get negative reviews (rating <= 3)
    negative_reviews = month_reviews[month_reviews['score_Overall Rating'] <= 3]
    
    # Use vectorstore to find similar issues
    if not negative_reviews.empty:
        issues = []
        for text in negative_reviews['Text'].tolist():
            similar_docs = vectorstore.similarity_search_with_score(
                text,
                k=3
            )
            for doc, score in similar_docs:
                if score < 0.8:  # Only include relevant matches
                    issues.append({
                        'text': doc.page_content,
                        'score': score,
                        'metadata': doc.metadata
                    })
    
    # Analyze issues and group them
    results = {
        'key_findings': [],
        'statistics': {
            'total_reviews': len(month_reviews),
            'negative_reviews': len(negative_reviews),
            'average_rating': month_reviews['score_Overall Rating'].mean()
        },
        'trends': [],
        'recommended_actions': []
    }
    
    # Add findings based on the analysis
    if not negative_reviews.empty:
        # Group by themes
        theme_counts = negative_reviews.groupby('Concepts').size()
        top_themes = theme_counts.nlargest(3)
        
        for theme, count in top_themes.items():
            results['key_findings'].append(
                f"{theme}: {count} negative reviews ({count/len(negative_reviews)*100:.1f}%)"
            )
    
    return results

def compare_store_feedback(store1: str, store2: str) -> Dict:
    """Compare feedback between two stores."""
    # Filter reviews for each store
    store1_reviews = df[df['string_Place_Location'].str.contains(store1, case=False)]
    store2_reviews = df[df['string_Place_Location'].str.contains(store2, case=False)]
    
    results = {
        'statistics': {
            store1: {
                'total_reviews': len(store1_reviews),
                'average_rating': store1_reviews['score_Overall Rating'].mean(),
                'positive_ratio': len(store1_reviews[store1_reviews['score_Overall Rating'] >= 4]) / len(store1_reviews)
            },
            store2: {
                'total_reviews': len(store2_reviews),
                'average_rating': store2_reviews['score_Overall Rating'].mean(),
                'positive_ratio': len(store2_reviews[store2_reviews['score_Overall Rating'] >= 4]) / len(store2_reviews)
            }
        },
        'key_findings': [],
        'recommended_actions': []
    }
    
    # Compare themes
    for store_name, store_data in [(store1, store1_reviews), (store2, store2_reviews)]:
        top_themes = store_data['Concepts'].value_counts().head(3)
        results['key_findings'].append(
            f"{store_name} top themes: " + ", ".join(f"{theme} ({count})" for theme, count in top_themes.items())
        )
    
    return results

def get_recommended_actions(analysis_results: Dict) -> List[str]:
    """Generate recommended actions based on analysis results."""
    actions = []
    
    # Add actions based on key findings
    for finding in analysis_results.get('key_findings', []):
        if 'negative' in finding.lower():
            actions.append(f"Address issue: {finding}")
    
    # Add actions based on statistics
    stats = analysis_results.get('statistics', {})
    if isinstance(stats, dict):
        for metric, value in stats.items():
            if isinstance(value, (int, float)) and metric == 'average_rating' and value < 4.0:
                actions.append(f"Improve {metric}: Current value {value:.1f}")
    
    return actions

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Process a chat request and return a formatted response."""
    try:
        print(f"Received chat request - Query: {request.query}, Role: {request.role}")
        
        # Look for CSV in the same directory as main.py
        current_dir = Path(__file__).parent
        csv_path = current_dir / "Jewelry Store Google Map Reviews.csv"
        
        print(f"Looking for CSV file at: {csv_path}")
        
        # Load data if not already loaded
        if not hasattr(app.state, 'df'):
            try:
                print("Loading CSV data...")
                app.state.df = pd.read_csv(csv_path)
                app.state.df['date_Date Created'] = pd.to_datetime(app.state.df['date_Date Created'])
                print(f"Data loaded successfully. Shape: {app.state.df.shape}")
            except Exception as e:
                print(f"Error loading CSV: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error loading data: {str(e)}"
                )

        query_lower = request.query.lower()
        
        # Handle different types of queries
        if "issues" in query_lower and ("month" in query_lower or "monthly" in query_lower):
            # Extract month from query or use current month
            current_date = datetime.now()
            month = current_date.month
            year = current_date.year
            
            # Filter reviews for the specified month
            month_mask = (app.state.df['date_Date Created'].dt.month == month) & \
                        (app.state.df['date_Date Created'].dt.year == year)
            month_reviews = app.state.df[month_mask]
            
            # Get negative reviews (rating <= 3)
            negative_reviews = month_reviews[month_reviews['score_Overall Rating'] <= 3]
            
            results = {
                'key_findings': [],
                'statistics': {
                    'total_reviews': len(month_reviews),
                    'negative_reviews': len(negative_reviews),
                    'average_rating': month_reviews['score_Overall Rating'].mean()
                },
                'trends': [],
                'recommended_actions': []
            }
            
            # Add findings based on the analysis
            if not negative_reviews.empty:
                # Group by cities instead of themes
                city_ratings = month_reviews.groupby('string_City')['score_Overall Rating'].agg(['mean', 'count'])
                low_rated_cities = city_ratings[city_ratings['mean'] < 4.0]
                
                for city, stats in low_rated_cities.iterrows():
                    results['key_findings'].append(
                        f"{city}: {stats['count']} reviews, average rating {stats['mean']:.1f}"
                    )
                    
                # Add recommended actions
                if len(results['key_findings']) > 0:
                    results['recommended_actions'].append(
                        "Focus on improving customer experience in cities with lower ratings"
                    )
            
        elif "difference" in query_lower or "compare" in query_lower:
            # Extract cities from the query if possible, otherwise use top cities
            cities = app.state.df['string_City'].value_counts().head(2).index.tolist()
            city1, city2 = cities[0], cities[1]
            
            # Compare cities
            city1_reviews = app.state.df[app.state.df['string_City'] == city1]
            city2_reviews = app.state.df[app.state.df['string_City'] == city2]
            
            results = {
                'statistics': {
                    city1: {
                        'total_reviews': len(city1_reviews),
                        'average_rating': city1_reviews['score_Overall Rating'].mean(),
                        'positive_ratio': len(city1_reviews[city1_reviews['score_Overall Rating'] >= 4]) / len(city1_reviews)
                    },
                    city2: {
                        'total_reviews': len(city2_reviews),
                        'average_rating': city2_reviews['score_Overall Rating'].mean(),
                        'positive_ratio': len(city2_reviews[city2_reviews['score_Overall Rating'] >= 4]) / len(city2_reviews)
                    }
                },
                'key_findings': [],
                'recommended_actions': []
            }
            
            # Add comparison findings
            for city, reviews in [(city1, city1_reviews), (city2, city2_reviews)]:
                avg_rating = reviews['score_Overall Rating'].mean()
                results['key_findings'].append(
                    f"{city}: {len(reviews)} reviews, average rating {avg_rating:.1f}"
                )
            
        elif "action" in query_lower or "recommend" in query_lower:
            # Get recent reviews (last 30 days)
            recent_date = app.state.df['date_Date Created'].max() - pd.Timedelta(days=30)
            recent_reviews = app.state.df[app.state.df['date_Date Created'] >= recent_date]
            
            results = {
                'key_findings': [],
                'recommended_actions': []
            }
            
            # Analyze recent reviews by city
            city_ratings = recent_reviews.groupby('string_City')['score_Overall Rating'].agg(['mean', 'count'])
            for city, stats in city_ratings.iterrows():
                results['key_findings'].append(
                    f"{city}: {stats['count']} recent reviews, average rating {stats['mean']:.1f}"
                )
            
            # Add general recommendations
            low_rated_cities = city_ratings[city_ratings['mean'] < 4.0]
            if not low_rated_cities.empty:
                results['recommended_actions'].extend([
                    f"Improve customer service in {city}" for city in low_rated_cities.index
                ])
            
        else:
            # Get overall statistics by city
            city_stats = app.state.df.groupby('string_City').agg({
                'score_Overall Rating': ['count', 'mean'],
                'score_Count People Found Review Helpful': 'sum'
            }).round(2)
            
            results = {
                'key_findings': [
                    f"Total reviews analyzed: {len(app.state.df)}",
                    f"Average rating across all cities: {app.state.df['score_Overall Rating'].mean():.1f}"
                ],
                'statistics': {
                    'total_reviews': len(app.state.df),
                    'cities_analyzed': len(city_stats),
                    'average_rating': app.state.df['score_Overall Rating'].mean()
                },
                'recommended_actions': [
                    "Specify your query to get more detailed insights:",
                    "- Ask about monthly issues",
                    "- Compare specific cities",
                    "- Request recent trends and actions"
                ]
            }
        
        # Format response based on role
        if request.role == "executive":
            formatted_response = format_executive_response(results)
        elif request.role == "analyst":
            formatted_response = format_analyst_response(results)
        else:  # store representative
            formatted_response = format_store_rep_response(results)
        
        return ChatResponse(
            response=formatted_response,
            metadata=results
        )
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        ) 