import axios from 'axios';
import { Theme, SubthemeReviewsResponse } from '../types';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getThemes = async (): Promise<{ status: string; themes: Theme[] }> => {
  const response = await api.get('/themes');
  return response.data;
};

export const getSubthemeReviews = async (
  subtheme: string,
  minRating?: number,
  maxRating?: number,
  sortBy: string = 'relevance',
  limit: number = 20
): Promise<SubthemeReviewsResponse> => {
  const response = await api.post('/reviews/subtheme', {
    subtheme,
    min_rating: minRating,
    max_rating: maxRating,
    sort_by: sortBy,
    limit,
  });
  return response.data;
};

export interface ChatRequest {
  query: string;
  role: 'executive' | 'analyst' | 'store_rep';
}

export interface ChatResponse {
  response: string;
  metadata?: Record<string, any>;
}

export const sendChatMessage = async (request: ChatRequest): Promise<ChatResponse> => {
  const response = await api.post('/chat', request);
  return response.data;
}; 