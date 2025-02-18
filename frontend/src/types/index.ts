export interface Theme {
  id: number;
  main_theme: string;
  sentiment: string;
  sentiment_explanation: string;
  subthemes: Subtheme[];
  statistics: ThemeStatistics;
}

export interface Subtheme {
  text: string;
  impact_level: string;
  frequency: string;
  action_item: string;
}

export interface ThemeStatistics {
  average_rating: number;
  total_reviews: number;
  rating_distribution: {
    '1_star': number;
    '2_star': number;
    '3_star': number;
    '4_star': number;
    '5_star': number;
  };
  sentiment_metrics: Record<string, any>;
}

export interface Review {
  text: string;
  relevance_score: number;
  metadata: {
    rating: number;
    date: string;
    city: string;
    state: string;
  };
}

export interface SubthemeReviewsResponse {
  status: string;
  subtheme: string;
  reviews: Review[];
  statistics: {
    total_reviews: number;
    average_rating: number;
    rating_distribution: Record<string, number>;
    relevance_metrics: {
      high_relevance: number;
      medium_relevance: number;
      low_relevance: number;
    };
  };
  filters_applied: {
    min_rating: number | null;
    max_rating: number | null;
    sort_by: string;
    limit: number;
  };
} 