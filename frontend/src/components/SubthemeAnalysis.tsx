import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
  CircularProgress,
  IconButton,
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { SubthemeReviewsResponse } from '../types';
import ReviewList from './ReviewList';
import MoreHorizIcon from '@mui/icons-material/MoreHoriz';
import FilterListIcon from '@mui/icons-material/FilterListOutlined';

interface SubthemeAnalysisProps {
  data: SubthemeReviewsResponse;
  isLoading: boolean;
  onSortChange: (sortBy: string) => void;
}

const SubthemeAnalysis: React.FC<SubthemeAnalysisProps> = ({
  data,
  isLoading,
  onSortChange,
}) => {
  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
      </Box>
    );
  }

  const ratingDistributionData = Object.entries(data.statistics.rating_distribution).map(
    ([key, value]) => ({
      rating: key.replace('_star', ' Star'),
      count: value,
    })
  );

  const handleSortChange = (event: SelectChangeEvent) => {
    onSortChange(event.target.value);
  };

  return (
    <Box
      sx={{
        height: 'calc(100vh - 140px)',
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
      }}
    >
      {/* Fixed Header Section */}
      <Paper sx={{ p: 2.5 }}>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box>
            <Typography variant="subtitle1" fontWeight={600}>
              Analysis: {data.subtheme}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Detailed review analysis and insights
            </Typography>
          </Box>
          <IconButton size="small">
            <MoreHorizIcon fontSize="small" />
          </IconButton>
        </Box>
      </Paper>

      {/* Scrollable Content */}
      <Box
        sx={{
          flexGrow: 1,
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
        {/* Overview and Charts Section */}
        <Paper sx={{ p: 2.5, mb: 2 }}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Box mb={3}>
                <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                  Overview
                </Typography>
                <Box display="flex" gap={3}>
                  <Box>
                    <Typography variant="h5" fontWeight={600}>
                      {data.statistics.total_reviews}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Total Reviews
                    </Typography>
                  </Box>
                  <Box>
                    <Typography variant="h5" fontWeight={600}>
                      {data.statistics.average_rating.toFixed(1)}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Average Rating
                    </Typography>
                  </Box>
                </Box>
              </Box>
              <Box height={300}>
                <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                  Rating Distribution
                </Typography>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={ratingDistributionData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="rating" fontSize={12} />
                    <YAxis fontSize={12} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#666CFF" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box mb={3}>
                <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                  Relevance Metrics
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={4}>
                    <Paper
                      sx={{
                        p: 1.5,
                        textAlign: 'center',
                        backgroundColor: '#4caf5015',
                      }}
                    >
                      <Typography variant="h6" color="success.main" fontWeight={600}>
                        {data.statistics.relevance_metrics.high_relevance}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        High Relevance
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={4}>
                    <Paper
                      sx={{
                        p: 1.5,
                        textAlign: 'center',
                        backgroundColor: '#ff980015',
                      }}
                    >
                      <Typography variant="h6" color="warning.main" fontWeight={600}>
                        {data.statistics.relevance_metrics.medium_relevance}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Medium
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={4}>
                    <Paper
                      sx={{
                        p: 1.5,
                        textAlign: 'center',
                        backgroundColor: '#f4433615',
                      }}
                    >
                      <Typography variant="h6" color="error.main" fontWeight={600}>
                        {data.statistics.relevance_metrics.low_relevance}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Low
                      </Typography>
                    </Paper>
                  </Grid>
                </Grid>
              </Box>
              <Box>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                  <Typography variant="subtitle2" fontWeight={600}>
                    Sort Reviews
                  </Typography>
                  <IconButton size="small">
                    <FilterListIcon fontSize="small" />
                  </IconButton>
                </Box>
                <FormControl fullWidth size="small">
                  <InputLabel>Sort By</InputLabel>
                  <Select
                    value={data.filters_applied.sort_by}
                    label="Sort By"
                    onChange={handleSortChange}
                  >
                    <MenuItem value="relevance">Relevance</MenuItem>
                    <MenuItem value="rating">Rating</MenuItem>
                    <MenuItem value="date">Date</MenuItem>
                  </Select>
                </FormControl>
              </Box>
            </Grid>
          </Grid>
        </Paper>

        {/* Reviews Section */}
        <Paper sx={{ p: 2.5 }}>
          <Typography variant="subtitle2" fontWeight={600} gutterBottom>
            Reviews
          </Typography>
          <ReviewList reviews={data.reviews} subtheme={data.subtheme} />
        </Paper>
      </Box>
    </Box>
  );
};

export default SubthemeAnalysis; 