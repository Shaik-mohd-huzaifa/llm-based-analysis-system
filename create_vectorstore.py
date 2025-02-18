import pandas as pd
import numpy as np
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import json
import sys
from openai import OpenAIError
from sklearn.feature_extraction.text import TfidfVectorizer

# Load environment variables from .env file
load_dotenv()

# Verify OpenAI API key is loaded and properly formatted
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY not found in environment variables. Please check your .env file.")
    sys.exit(1)
if api_key.count("OPENAI_API_KEY=") > 0:
    print("Error: OPENAI_API_KEY contains duplicate prefix. Please remove 'OPENAI_API_KEY=' from the value in .env file.")
    sys.exit(1)

try:
    # Test API key with a small request
    embeddings = OpenAIEmbeddings(api_key=api_key)
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    _ = embeddings.embed_query("test")
    print("API key validated successfully!")
except OpenAIError as e:
    print(f"Error: Invalid OpenAI API key or API error: {str(e)}")
    sys.exit(1)

def get_detailed_cluster_themes(texts: List[str], llm: ChatOpenAI) -> Dict:
    """Get detailed themes and sub-themes for a cluster using LLM."""
    # Sample up to 15 texts to get better coverage
    sample_texts = texts[:15] if len(texts) > 15 else texts
    
    # Analyze the dominant topics in the reviews
    topic_analysis_prompt = f"""Analyze these reviews and determine which ONE category they primarily discuss:
1. Store Experience & Layout
2. Product Quality & Selection
3. Customer Service & Staff
4. Piercing Services
5. Store Operations & Inventory
6. Price & Value
7. Shopping Experience
8. Product Display & Merchandising

Reviews:
{json.dumps(sample_texts)}

Return ONLY the number (1-8) of the most relevant category."""

    try:
        category_response = llm.predict(topic_analysis_prompt)
        category_num = int(category_response.strip())
    except:
        category_num = 3  # Default to Customer Service if parsing fails
    
    # Map category number to theme type
    theme_categories = {
        1: "Store Experience & Layout",
        2: "Product Quality & Selection",
        3: "Customer Service & Staff",
        4: "Piercing Services",
        5: "Store Operations & Inventory",
        6: "Price & Value",
        7: "Shopping Experience",
        8: "Product Display & Merchandising"
    }
    
    main_theme = theme_categories.get(category_num, "Customer Service & Staff")
    
    # Now get detailed analysis based on the identified theme
    prompt = f"""Analyze these customer reviews focusing on {main_theme}.

Key Requirements:
1. Main theme is: {main_theme}

2. Subthemes should be specific and detailed. Include 6-8 subthemes from these categories based on the main theme:

For Store Experience & Layout:
- Store design and atmosphere
- Traffic flow and space utilization
- Lighting and ambiance
- Seating and comfort
- Store cleanliness
- Display organization
- Navigation ease
- Fitting room experience

For Product Quality & Selection:
- Product durability
- Material quality
- Style variety
- Size range
- Collection diversity
- Product packaging
- Product presentation
- Seasonal offerings

For Customer Service & Staff:
- Staff knowledge
- Friendliness
- Response time
- Problem resolution
- Personal attention
- Communication clarity
- Follow-up service
- Staff availability

For Piercing Services:
- Piercer expertise
- Hygiene standards
- Pain management
- Aftercare guidance
- Jewelry selection
- Appointment system
- Recovery support
- Safety protocols

For Store Operations & Inventory:
- Stock availability
- Inventory organization
- Price labeling
- Return process
- Store hours
- Queue management
- Payment options
- Special orders

For Price & Value:
- Price competitiveness
- Value for money
- Discount offerings
- Loyalty programs
- Price transparency
- Special promotions
- Bundle deals
- Price matching

For Shopping Experience:
- Ease of shopping
- Transaction speed
- Store accessibility
- Product testing
- Gift wrapping
- Mobile payment
- Online-offline integration
- Shopping comfort

For Product Display & Merchandising:
- Visual merchandising
- Product placement
- Display creativity
- Category organization
- Seasonal displays
- Window displays
- Product information
- Brand storytelling

Reviews to analyze:
{json.dumps(sample_texts)}

Provide your analysis in this JSON format:
{{
    "main_theme": "{main_theme}",
    "sub_themes": [
        {{
            "text": "string (specific, detailed finding)",
            "color": "#hex_color (matching the sentiment)",
            "action_item": "string (specific improvement action)",
            "impact_level": "string (high/medium/low)",
            "frequency": "string (common/occasional/rare)"
        }},
        // Include 6-8 subthemes
    ],
    "sentiment": {{
        "type": "string (positive/negative/mixed)",
        "explanation": "string (detailed business impact explanation)"
    }}
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
                "explanation": "Failed to analyze"
            }
        }

def analyze_cluster_statistics(texts: List[str], ratings: List[float]) -> Dict:
    """Generate statistical analysis for a cluster."""
    avg_rating = np.mean(ratings) if ratings else 0
    rating_distribution = {
        '1 star': sum(1 for r in ratings if r == 1),
        '2 stars': sum(1 for r in ratings if r == 2),
        '3 stars': sum(1 for r in ratings if r == 3),
        '4 stars': sum(1 for r in ratings if r == 4),
        '5 stars': sum(1 for r in ratings if r == 5)
    }
    
    avg_review_length = np.mean([len(text.split()) for text in texts])
    
    return {
        'average_rating': round(avg_rating, 2),
        'rating_distribution': rating_distribution,
        'average_review_length': round(avg_review_length, 2),
        'total_reviews': len(texts)
    }

# Read the CSV file
df = pd.read_csv('Jewelry Store Google Map Reviews.csv')

# Initialize the embeddings model and LLM
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

print("Generating embeddings...")
# Generate embeddings for all texts
texts = df['Text'].tolist()
embedding_vectors = embeddings.embed_documents(texts)

# Standardize the features
scaler = StandardScaler()

# Before clustering, let's create feature vectors that emphasize different aspects
print("Preparing specialized feature vectors...")

# Define keyword sets for different aspects with sentiment indicators
aspect_keywords = {
    'customer_service_positive': [
        'helpful', 'friendly', 'knowledgeable', 'professional', 'excellent service',
        'great staff', 'attentive', 'patient', 'accommodating', 'wonderful service',
        'amazing staff', 'best service', 'caring', 'went above and beyond'
    ],
    'customer_service_negative': [
        'rude', 'unhelpful', 'unprofessional', 'poor service', 'bad service',
        'unfriendly', 'unresponsive', 'dismissive', 'terrible service', 'worst service',
        'poor customer service', 'bad attitude', 'not helpful', 'disrespectful'
    ],
    'product_quality': [
        'quality', 'material', 'durability', 'craftsmanship', 'authentic',
        'genuine', 'well-made', 'premium', 'high-quality', 'superior'
    ],
    'store_atmosphere': [
        'clean', 'organized', 'layout', 'atmosphere', 'ambiance',
        'decor', 'display', 'lighting', 'spacious', 'comfortable'
    ],
    'pricing_value': [
        'price', 'expensive', 'affordable', 'value', 'worth',
        'overpriced', 'reasonable', 'fair price', 'cost', 'budget'
    ],
    'product_selection': [
        'variety', 'selection', 'collection', 'options', 'range',
        'choices', 'inventory', 'styles', 'designs', 'assortment'
    ],
    'shopping_experience': [
        'experience', 'convenient', 'easy', 'quick', 'smooth',
        'hassle-free', 'efficient', 'pleasant', 'enjoyable', 'satisfying'
    ]
}

# Create sentiment-aware feature vectors
combined_features = []
for text in texts:
    text_lower = text.lower()
    
    # Calculate aspect scores
    aspect_scores = []
    for keywords in aspect_keywords.values():
        # Count keyword matches and weight by position (earlier mentions get higher weight)
        score = 0
        words = text_lower.split()
        for i, word in enumerate(words):
            position_weight = 1 - (i / len(words))  # Earlier words get higher weight
            for keyword in keywords:
                if keyword in word:
                    score += position_weight
        aspect_scores.append(score)
    
    # Add sentiment bias for customer service categories
    if any(keyword in text_lower for keyword in aspect_keywords['customer_service_positive']):
        aspect_scores[0] += 2  # Boost positive service score
    if any(keyword in text_lower for keyword in aspect_keywords['customer_service_negative']):
        aspect_scores[1] += 2  # Boost negative service score
    
    # Combine with embeddings
    combined_features.append(np.concatenate([embedding_vectors[len(combined_features)], aspect_scores]))

# Convert to numpy array and standardize
combined_array = np.array(combined_features)
combined_array_scaled = scaler.fit_transform(combined_array)

# Perform K-Means clustering with combined features
print("Performing clustering with specialized features...")
n_clusters = 7  # Customer Service (Positive & Negative) + 5 other themes
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(combined_array_scaled)

# Get detailed cluster themes and statistics
print("Generating detailed cluster analysis...")

# Helper function to determine the dominant theme for a cluster
def get_cluster_theme(texts: List[str], aspect_keywords: Dict) -> Tuple[str, float]:
    theme_scores = {theme: 0 for theme in aspect_keywords.keys()}
    
    for text in texts:
        text_lower = text.lower()
        for theme, keywords in aspect_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            theme_scores[theme] += score
    
    # Normalize scores by number of texts
    for theme in theme_scores:
        theme_scores[theme] /= len(texts)
    
    # Get theme with highest score
    dominant_theme = max(theme_scores.items(), key=lambda x: x[1])
    return dominant_theme[0], dominant_theme[1]

# Analyze each cluster
cluster_analysis = {}
for i in range(n_clusters):
    cluster_texts = [text for text, label in zip(texts, cluster_labels) if label == i]
    cluster_ratings = [rating for rating, label in zip(df['score_Overall Rating'], cluster_labels) if label == i]
    
    # Get dominant theme
    dominant_theme, theme_score = get_cluster_theme(cluster_texts, aspect_keywords)
    
    # Get themes and sentiment
    themes = get_detailed_cluster_themes(cluster_texts, llm)
    
    # Get statistics
    stats = analyze_cluster_statistics(cluster_texts, cluster_ratings)
    
    cluster_analysis[i] = {
        'themes': themes,
        'statistics': stats,
        'dominant_theme': dominant_theme,
        'theme_score': theme_score,
        'texts': cluster_texts,
        'ratings': cluster_ratings
    }
    
    print(f"\nCluster {i}:")
    print(f"Dominant Theme: {dominant_theme}")
    print(f"Theme Score: {theme_score:.2f}")
    print(f"Main Theme: {themes['main_theme']}")
    print(f"Average Rating: {stats['average_rating']:.2f}")
    print(f"Total Reviews: {stats['total_reviews']}")

# Prepare the texts and metadata
metadatas = []

for idx, row in df.iterrows():
    cluster_id = int(cluster_labels[idx])
    metadata = {
        'string_City': row['string_City'],
        'People_Found_Review_Helpful': row['score_Count People Found Review Helpful'],
        'date_Date_Created': row['date_Date Created'],
        'string_Name': row['string_Name'],
        'score_Overall_Rating': row['score_Overall Rating'],
        'string_Place_Location': row['string_Place Location'],
        'string_State': row['string_State'],
        'string_User_Name': row['string_User Name'],
        'Concepts': row['Concepts'],
        'cluster_id': cluster_id,
        'main_theme': cluster_analysis[cluster_id]['themes']['main_theme'],
        'sub_themes': json.dumps([theme['text'] for theme in cluster_analysis[cluster_id]['themes']['sub_themes']]),
        'sentiment_type': cluster_analysis[cluster_id]['themes']['sentiment']['type'],
        'sentiment_explanation': cluster_analysis[cluster_id]['themes']['sentiment']['explanation']
    }
    metadatas.append(metadata)

# Create the vectorstore
print("\nCreating vectorstore...")
vectorstore = Chroma.from_texts(
    texts=texts,
    metadatas=metadatas,
    embedding=embeddings,
    persist_directory="jewelry_reviews_vectorstore"
)

# Persist the vectorstore
vectorstore.persist()

# Save detailed cluster information
with open('cluster_analysis.json', 'w') as f:
    json.dump(cluster_analysis, f, indent=2)

print("\nVectorstore created and persisted successfully!")
print("\nDetailed Cluster Analysis saved to cluster_analysis.json") 