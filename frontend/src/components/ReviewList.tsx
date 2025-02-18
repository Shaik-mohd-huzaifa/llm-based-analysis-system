import React from 'react';
import {
  List,
  ListItem,
  ListItemText,
  Typography,
  Paper,
  Rating,
  Box,
  Chip,
  Divider,
  IconButton,
} from '@mui/material';
import { Review } from '../types';
import MoreHorizIcon from '@mui/icons-material/MoreHoriz';

interface ReviewListProps {
  reviews: Review[];
  subtheme: string;
}

const getRelevanceColor = (score: number): string => {
  if (score >= 0.7) return '#4caf50';
  if (score >= 0.4) return '#ff9800';
  return '#f44336';
};

const ReviewList: React.FC<ReviewListProps> = ({ reviews, subtheme }) => {
  return (
    <List
      sx={{
        width: '100%',
        bgcolor: 'background.paper',
        position: 'relative',
      }}
    >
      {reviews.map((review, index) => (
        <React.Fragment key={index}>
          {index > 0 && <Divider />}
          <ListItem
            component={Paper}
            elevation={0}
            sx={{
              mb: 2,
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'flex-start',
              transition: 'all 0.2s ease-in-out',
              border: '1px solid #e9ecef',
              '&:hover': {
                backgroundColor: '#f8f9fa',
                borderColor: '#666CFF',
              },
            }}
          >
            <Box
              display="flex"
              justifyContent="space-between"
              alignItems="center"
              width="100%"
              mb={1.5}
            >
              <Box display="flex" alignItems="center" gap={1}>
                <Rating 
                  value={review.metadata.rating}
                  readOnly
                  precision={0.5}
                  size="small"
                  sx={{
                    color: '#666CFF',
                  }}
                />
                <Typography variant="body2" color="text.secondary">
                  ({review.metadata.rating})
                </Typography>
              </Box>
              <Box display="flex" alignItems="center" gap={1}>
                <Chip
                  label={`${(review.relevance_score * 100).toFixed(0)}% Relevant`}
                  size="small"
                  sx={{
                    backgroundColor: `${getRelevanceColor(review.relevance_score)}15`,
                    color: getRelevanceColor(review.relevance_score),
                    fontWeight: 500,
                    fontSize: '0.75rem',
                  }}
                />
                <IconButton size="small">
                  <MoreHorizIcon fontSize="small" />
                </IconButton>
              </Box>
            </Box>

            <Typography
              variant="body2"
              sx={{
                lineHeight: 1.6,
                color: 'text.primary',
                mb: 1.5,
              }}
            >
              {review.text}
            </Typography>

            <Box
              display="flex"
              justifyContent="space-between"
              alignItems="center"
              width="100%"
            >
              <Box display="flex" alignItems="center" gap={0.5}>
                <Box
                  component="span"
                  sx={{
                    width: 6,
                    height: 6,
                    backgroundColor: 'text.secondary',
                    borderRadius: '50%',
                    opacity: 0.4,
                  }}
                />
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ fontStyle: 'italic' }}
                >
                  {review.metadata.city}
                  {review.metadata.state && `, ${review.metadata.state}`}
                </Typography>
              </Box>
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ fontWeight: 500 }}
              >
                {new Date(review.metadata.date).toLocaleDateString(undefined, {
                  year: 'numeric',
                  month: 'short',
                  day: 'numeric',
                })}
              </Typography>
            </Box>
          </ListItem>
        </React.Fragment>
      ))}
    </List>
  );
};

export default ReviewList; 