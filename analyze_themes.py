import json
from typing import Dict, List
from collections import defaultdict
import pandas as pd
from tabulate import tabulate
from colorama import init, Fore, Style

# Initialize colorama for colored output
init()

def load_cluster_data(file_path: str) -> Dict:
    """Load and parse the cluster analysis JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_themes(cluster_data: Dict) -> Dict:
    """Analyze themes and subthemes from cluster data."""
    theme_analysis = defaultdict(list)
    
    for cluster_id, cluster in cluster_data.items():
        main_theme = cluster['themes']['main_theme']
        sentiment = cluster['themes']['sentiment']['type']
        total_reviews = cluster['statistics']['total_reviews']
        avg_rating = cluster['statistics']['average_rating']
        
        # Get subthemes
        subthemes = cluster['themes']['sub_themes']
        if isinstance(subthemes[0], dict):
            # New format with detailed subtheme info
            for subtheme in subthemes:
                theme_analysis[main_theme].append({
                    'subtheme': subtheme['text'],
                    'impact': subtheme.get('impact_level', 'Unknown'),
                    'frequency': subtheme.get('frequency', 'Unknown'),
                    'action': subtheme.get('action_item', 'No action specified')
                })
        else:
            # Old format with simple subtheme strings
            for subtheme in subthemes:
                theme_analysis[main_theme].append({
                    'subtheme': subtheme,
                    'impact': 'Unknown',
                    'frequency': 'Unknown',
                    'action': 'No action specified'
                })
        
        # Add metadata to the first subtheme
        if theme_analysis[main_theme]:
            theme_analysis[main_theme][0].update({
                'sentiment': sentiment,
                'total_reviews': total_reviews,
                'avg_rating': avg_rating
            })
    
    return theme_analysis

def display_theme_hierarchy(theme_analysis: Dict):
    """Display the theme hierarchy in a structured format."""
    print(f"\n{Fore.CYAN}=== Customer Feedback Theme Analysis ==={Style.RESET_ALL}\n")
    
    for main_theme, subthemes in theme_analysis.items():
        # Display main theme with metadata
        print(f"\n{Fore.GREEN}Main Theme: {main_theme}{Style.RESET_ALL}")
        if subthemes:
            metadata = subthemes[0]
            print(f"Sentiment: {metadata.get('sentiment', 'Unknown')}")
            print(f"Total Reviews: {metadata.get('total_reviews', 0)}")
            print(f"Average Rating: {metadata.get('avg_rating', 0):.2f}")
        
        # Display subthemes in a table
        subtheme_data = [
            [
                s['subtheme'],
                s.get('impact', 'Unknown'),
                s.get('frequency', 'Unknown'),
                s.get('action', 'No action specified')
            ]
            for s in subthemes
        ]
        
        print(f"\n{Fore.YELLOW}Subthemes:{Style.RESET_ALL}")
        print(tabulate(
            subtheme_data,
            headers=['Subtheme', 'Impact', 'Frequency', 'Action Item'],
            tablefmt='grid'
        ))
        print("\n" + "-"*80)

def main():
    try:
        # Load cluster data
        cluster_data = load_cluster_data('cluster_analysis.json')
        
        # Analyze themes
        theme_analysis = analyze_themes(cluster_data)
        
        # Display results
        display_theme_hierarchy(theme_analysis)
        
    except FileNotFoundError:
        print(f"{Fore.RED}Error: cluster_analysis.json not found{Style.RESET_ALL}")
    except json.JSONDecodeError:
        print(f"{Fore.RED}Error: Invalid JSON format in cluster_analysis.json{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")

if __name__ == "__main__":
    main() 