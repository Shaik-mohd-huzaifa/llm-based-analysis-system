import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Chip,
  Box,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  Divider,
  IconButton,
  Collapse,
  Menu,
  MenuItem,
} from '@mui/material';
import { Theme, Review } from '../types';
import MoreHorizIcon from '@mui/icons-material/MoreHoriz';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';
import KeyboardArrowUpIcon from '@mui/icons-material/KeyboardArrowUp';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import * as XLSX from 'xlsx';

interface ThemeCardProps {
  theme: Theme;
  onSubthemeClick: (subtheme: string) => void;
}

const getSentimentColor = (sentiment: string): string => {
  switch (sentiment.toLowerCase()) {
    case 'positive':
      return '#4caf50';
    case 'negative':
      return '#f44336';
    case 'mixed':
      return '#ff9800';
    default:
      return '#757575';
  }
};

const getSubthemeReviews = async (subtheme: string, limit: number = 50) => {
  try {
    const response = await fetch('http://localhost:8000/reviews/subtheme', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        subtheme,
        limit,
        sort_by: 'relevance'
      }),
    });
    return await response.json();
  } catch (error) {
    console.error('Error fetching subtheme reviews:', error);
    return null;
  }
};

const ThemeCard: React.FC<ThemeCardProps> = ({ theme, onSubthemeClick }) => {
  const [expanded, setExpanded] = useState(false);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const open = Boolean(anchorEl);
  const sentimentColor = getSentimentColor(theme.sentiment);
  const totalReviews = theme.statistics.total_reviews;
  const ratingTrend = theme.statistics.average_rating >= 4 ? 'up' : 'down';
  
  const handleMenuClick = (event: React.MouseEvent<HTMLElement>) => {
    event.stopPropagation();
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = (event: React.MouseEvent<HTMLElement>) => {
    event.stopPropagation();
    setAnchorEl(null);
  };

  const handleExport = (event: React.MouseEvent<HTMLElement>) => {
    event.stopPropagation();
    setAnchorEl(null);

    // Create a new workbook
    const wb = XLSX.utils.book_new();

    // Theme Overview Sheet
    const overviewData = [
      ['Theme Analysis Report'],
      [],
      ['Theme Information'],
      ['Main Theme', theme.main_theme],
      ['Sentiment', theme.sentiment],
      ['Sentiment Explanation', theme.sentiment_explanation],
      [],
      ['Statistics'],
      ['Total Reviews', theme.statistics.total_reviews],
      ['Average Rating', theme.statistics.average_rating.toFixed(2)],
      [],
      ['Rating Distribution'],
      ['Rating', 'Count'],
      ['5 stars', theme.statistics.rating_distribution['5_star'] || 0],
      ['4 stars', theme.statistics.rating_distribution['4_star'] || 0],
      ['3 stars', theme.statistics.rating_distribution['3_star'] || 0],
      ['2 stars', theme.statistics.rating_distribution['2_star'] || 0],
      ['1 star', theme.statistics.rating_distribution['1_star'] || 0],
      [],
      ['Sentiment Metrics'],
      ['Positive Reviews Ratio', `${theme.statistics.sentiment_metrics?.positive_ratio || 0}%`],
      ['Detailed Reviews Count', theme.statistics.sentiment_metrics?.detailed_feedback_count || 0],
      ['Rating Trend', theme.statistics.sentiment_metrics?.rating_trend || 'N/A']
    ];

    // Subthemes Sheet
    const subthemesData = [
      ['Subtheme Analysis'],
      [],
      ['Subtheme', 'Impact Level', 'Frequency', 'Action Item'],
      ...theme.subthemes.map(subtheme => [
        subtheme.text,
        subtheme.impact_level,
        subtheme.frequency,
        subtheme.action_item
      ])
    ];

    // Create sheets
    const wsOverview = XLSX.utils.aoa_to_sheet(overviewData);
    const wsSubthemes = XLSX.utils.aoa_to_sheet(subthemesData);

    // Style the sheets
    const overviewMerge = { s: { c: 0, r: 0 }, e: { c: 3, r: 0 } };
    if (!wsOverview['!merges']) wsOverview['!merges'] = [];
    wsOverview['!merges'].push(overviewMerge);

    const subthemesMerge = { s: { c: 0, r: 0 }, e: { c: 3, r: 0 } };
    if (!wsSubthemes['!merges']) wsSubthemes['!merges'] = [];
    wsSubthemes['!merges'].push(subthemesMerge);

    // Set column widths
    wsOverview['!cols'] = [
      { wch: 25 }, // Column A
      { wch: 50 }, // Column B
      { wch: 15 }, // Column C
      { wch: 15 }  // Column D
    ];

    wsSubthemes['!cols'] = [
      { wch: 40 }, // Column A
      { wch: 15 }, // Column B
      { wch: 15 }, // Column C
      { wch: 50 }  // Column D
    ];

    // Add sheets to workbook
    XLSX.utils.book_append_sheet(wb, wsOverview, 'Theme Overview');
    XLSX.utils.book_append_sheet(wb, wsSubthemes, 'Subthemes');

    // Add individual sheets for each subtheme with their reviews
    theme.subthemes.forEach((subtheme, index) => {
      // Fetch reviews for this subtheme
      getSubthemeReviews(subtheme.text, 50).then(reviewsData => {
        if (reviewsData && reviewsData.reviews) {
          const reviewsSheet = [
            [`Reviews for Subtheme: ${subtheme.text}`],
            [],
            ['Review Text', 'Rating', 'Date', 'Location', 'Relevance Score'],
            ...reviewsData.reviews.map((review: Review) => [
              review.text,
              review.metadata.rating,
              new Date(review.metadata.date).toLocaleDateString(),
              `${review.metadata.city}${review.metadata.state ? `, ${review.metadata.state}` : ''}`,
              review.relevance_score.toFixed(3)
            ])
          ];

          const wsReviews = XLSX.utils.aoa_to_sheet(reviewsSheet);
          
          // Style the reviews sheet
          const reviewsMerge = { s: { c: 0, r: 0 }, e: { c: 4, r: 0 } };
          if (!wsReviews['!merges']) wsReviews['!merges'] = [];
          wsReviews['!merges'].push(reviewsMerge);

          wsReviews['!cols'] = [
            { wch: 100 }, // Review Text
            { wch: 10 },  // Rating
            { wch: 15 },  // Date
            { wch: 25 },  // Location
            { wch: 15 }   // Relevance Score
          ];

          // Add the reviews sheet to the workbook
          XLSX.utils.book_append_sheet(wb, wsReviews, `${subtheme.text.slice(0, 28)}_Reviews`);

          // If this is the last subtheme, generate the file
          if (index === theme.subthemes.length - 1) {
            // Generate and download the Excel file
            XLSX.writeFile(wb, `${theme.main_theme.replace(/[^a-zA-Z0-9]/g, '_')}_analysis.xlsx`);
          }
        }
      });
    });
  };

  return (
    <Card 
      sx={{ 
        height: 'auto',
        display: 'flex',
        flexDirection: 'column',
        transition: 'all 0.2s ease-in-out',
        '&:hover': {
          borderColor: expanded ? '#666CFF' : 'inherit',
        },
      }}
    >
      <CardContent sx={{ p: 2.5 }}>
        {/* Header Section - Always Visible */}
        <Box 
          display="flex" 
          justifyContent="space-between" 
          alignItems="flex-start"
          onClick={() => setExpanded(!expanded)}
          sx={{ cursor: 'pointer' }}
        >
          <Box flex={1}>
            <Box display="flex" alignItems="center" gap={1} mb={1}>
              <Typography variant="subtitle1" fontWeight={600}>
                {theme.main_theme}
              </Typography>
              <IconButton 
                size="small"
                onClick={(e) => {
                  e.stopPropagation();
                  setExpanded(!expanded);
                }}
              >
                {expanded ? <KeyboardArrowUpIcon /> : <KeyboardArrowDownIcon />}
              </IconButton>
            </Box>
            <Typography 
              variant="body2" 
              color="text.secondary"
              sx={{
                display: '-webkit-box',
                WebkitLineClamp: expanded ? 'none' : 2,
                WebkitBoxOrient: 'vertical',
                overflow: 'hidden',
                mb: expanded ? 2 : 0,
              }}
            >
              {theme.sentiment_explanation}
            </Typography>
          </Box>
          <Box display="flex" alignItems="center" gap={1}>
            <Chip
              label={theme.sentiment}
              size="small"
              sx={{
                backgroundColor: `${sentimentColor}15`,
                color: sentimentColor,
                fontWeight: 600,
                fontSize: '0.75rem',
              }}
            />
            <IconButton 
              size="small"
              onClick={handleMenuClick}
            >
              <MoreHorizIcon fontSize="small" />
            </IconButton>
          </Box>
        </Box>

        <Menu
          anchorEl={anchorEl}
          open={open}
          onClose={handleMenuClose}
          onClick={handleMenuClose}
        >
          <MenuItem onClick={handleExport}>
            <FileDownloadIcon fontSize="small" sx={{ mr: 1 }} />
            Export to Excel
          </MenuItem>
        </Menu>

        {/* Expandable Content */}
        <Collapse in={expanded} timeout="auto">
          <Box mt={2}>
            <Box mb={3}>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                <Typography variant="body2" color="text.secondary">
                  Average Rating
                </Typography>
                <Box display="flex" alignItems="center" gap={0.5}>
                  <Typography variant="subtitle2" fontWeight={600}>
                    {theme.statistics.average_rating.toFixed(1)}
                  </Typography>
                  {ratingTrend === 'up' ? (
                    <TrendingUpIcon fontSize="small" color="success" />
                  ) : (
                    <TrendingDownIcon fontSize="small" color="error" />
                  )}
                </Box>
              </Box>
              <LinearProgress
                variant="determinate"
                value={(theme.statistics.average_rating / 5) * 100}
                sx={{
                  height: 6,
                  borderRadius: 3,
                  backgroundColor: `${sentimentColor}15`,
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: sentimentColor,
                    borderRadius: 3,
                  },
                }}
              />
              <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                Based on {totalReviews} reviews
              </Typography>
            </Box>

            <Typography variant="subtitle2" fontWeight={600} gutterBottom>
              Key Subthemes
            </Typography>
            <List 
              dense 
              disablePadding
              sx={{
                maxHeight: 300,
                overflowY: 'auto',
                '&::-webkit-scrollbar': {
                  width: '6px',
                },
                '&::-webkit-scrollbar-track': {
                  background: '#f1f1f1',
                  borderRadius: '3px',
                },
                '&::-webkit-scrollbar-thumb': {
                  background: '#888',
                  borderRadius: '3px',
                  '&:hover': {
                    background: '#666',
                  },
                },
              }}
            >
              {theme.subthemes.map((subtheme, index) => (
                <React.Fragment key={index}>
                  {index > 0 && <Divider />}
                  <ListItem
                    onClick={() => onSubthemeClick(subtheme.text)}
                    sx={{
                      px: 0,
                      cursor: 'pointer',
                      '&:hover': {
                        backgroundColor: 'rgba(0, 0, 0, 0.02)',
                      },
                    }}
                  >
                    <ListItemText
                      primary={
                        <Typography variant="body2" fontWeight={500}>
                          {subtheme.text}
                        </Typography>
                      }
                      secondary={
                        <Typography variant="caption" color="text.secondary">
                          Impact: {subtheme.impact_level} • Frequency: {subtheme.frequency}
                        </Typography>
                      }
                    />
                  </ListItem>
                </React.Fragment>
              ))}
            </List>
          </Box>
        </Collapse>
      </CardContent>
    </Card>
  );
};

export default ThemeCard; 