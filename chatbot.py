import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from datetime import datetime
import json
from typing import List, Dict, Optional
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class ReviewAnalysisChatbot:
    def __init__(self):
        # Initialize models and data
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo")
        self.vectorstore = Chroma(
            persist_directory="jewelry_reviews_vectorstore",
            embedding_function=self.embeddings
        )
        
        # Load cluster analysis
        with open('cluster_analysis.json', 'r') as f:
            self.cluster_analysis = json.load(f)
        
        # Load raw data
        self.df = pd.read_csv('Jewelry Store Google Map Reviews.csv')
        self.df['date_Date Created'] = pd.to_datetime(self.df['date_Date Created'])
        
    def format_response_for_role(self, analysis_results: Dict, role: str) -> str:
        """Format the analysis results based on the user's role."""
        if role == "executive":
            return self._format_executive_response(analysis_results)
        elif role == "analyst":
            return self._format_analyst_response(analysis_results)
        else:  # store representative
            return self._format_store_rep_response(analysis_results)

    def _format_executive_response(self, results: Dict) -> str:
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

    def _format_analyst_response(self, results: Dict) -> str:
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

    def _format_store_rep_response(self, results: Dict) -> str:
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

    def analyze_monthly_issues(self, month: int, year: int) -> Dict:
        """Analyze customer issues for a specific month."""
        # Filter reviews for the specified month
        month_mask = (self.df['date_Date Created'].dt.month == month) & \
                    (self.df['date_Date Created'].dt.year == year)
        month_reviews = self.df[month_mask]
        
        # Get negative reviews (rating <= 3)
        negative_reviews = month_reviews[month_reviews['score_Overall Rating'] <= 3]
        
        # Use vectorstore to find similar issues
        if not negative_reviews.empty:
            issues = []
            for text in negative_reviews['Text'].tolist():
                similar_docs = self.vectorstore.similarity_search_with_score(
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

    def compare_store_feedback(self, store1: str, store2: str) -> Dict:
        """Compare feedback between two stores."""
        # Filter reviews for each store
        store1_reviews = self.df[self.df['string_Place_Location'].str.contains(store1, case=False)]
        store2_reviews = self.df[self.df['string_Place_Location'].str.contains(store2, case=False)]
        
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

    def get_recommended_actions(self, analysis_results: Dict) -> List[str]:
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

    def process_query(self, query: str, role: str = "analyst") -> str:
        """Process a user query and return a formatted response."""
        # Extract key information from query
        query_lower = query.lower()
        
        try:
            # Handle different types of queries
            if "issues" in query_lower and ("month" in query_lower or "monthly" in query_lower):
                # Extract month from query or use current month
                current_date = datetime.now()
                month = current_date.month
                year = current_date.year
                
                results = self.analyze_monthly_issues(month, year)
                results['recommended_actions'] = self.get_recommended_actions(results)
                
            elif "difference" in query_lower or "compare" in query_lower:
                # Extract store locations from query
                # This is a simplified version - you might want to add more robust location extraction
                store1 = "Boston"  # Example - you should extract this from the query
                store2 = "New York"  # Example - you should extract this from the query
                
                results = self.compare_store_feedback(store1, store2)
                results['recommended_actions'] = self.get_recommended_actions(results)
                
            elif "action" in query_lower:
                # Generate action items based on recent reviews
                recent_reviews = self.df.sort_values('date_Date Created').tail(100)
                results = {
                    'key_findings': [],
                    'recommended_actions': []
                }
                
                # Analyze recent reviews
                for theme, count in recent_reviews['Concepts'].value_counts().head(5).items():
                    results['key_findings'].append(f"{theme}: {count} mentions")
                
                results['recommended_actions'] = self.get_recommended_actions(results)
                
            else:
                # Generic analysis
                results = {
                    'key_findings': ["Query not specific enough. Please ask about monthly issues, store comparisons, or specific actions."],
                    'recommended_actions': ["Refine your query to get more specific insights."]
                }
            
            # Format response based on role
            return self.format_response_for_role(results, role)
            
        except Exception as e:
            return f"Error processing query: {str(e)}\nPlease try rephrasing your question."

def main():
    # Initialize chatbot
    chatbot = ReviewAnalysisChatbot()
    
    print("🤖 Review Analysis Chatbot")
    print("Type 'quit' to exit")
    print("\nAvailable roles: executive, analyst, store_rep")
    
    # Get initial role
    role = input("\nWhat is your role? ").lower()
    while role not in ['executive', 'analyst', 'store_rep']:
        print("Invalid role. Please choose from: executive, analyst, store_rep")
        role = input("What is your role? ").lower()
    
    print(f"\nWelcome! You're now in {role} mode.")
    print("Ask me questions about customer reviews, store performance, and recommended actions.")
    
    while True:
        query = input("\nYour question: ")
        if query.lower() == 'quit':
            break
            
        response = chatbot.process_query(query, role)
        print("\n" + response)

if __name__ == "__main__":
    main() 