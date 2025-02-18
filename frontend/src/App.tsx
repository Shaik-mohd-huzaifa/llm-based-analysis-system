import React, { useState } from 'react';
import {
  Container,
  Grid,
  Typography,
  Box,
  CssBaseline,
  AppBar,
  Toolbar,
  useTheme,
  useMediaQuery,
  CircularProgress,
  Button,
  Divider,
  Dialog,
  IconButton,
} from '@mui/material';
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query';
import { getThemes, getSubthemeReviews } from './api';
import ThemeCard from './components/ThemeCard';
import SubthemeAnalysis from './components/SubthemeAnalysis';
import ChatInterface from './components/ChatInterface';
import DownloadIcon from '@mui/icons-material/FileDownloadOutlined';
import FilterListIcon from '@mui/icons-material/FilterListOutlined';
import ChatIcon from '@mui/icons-material/ChatOutlined';
import CloseIcon from '@mui/icons-material/Close';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <CssBaseline />
      <ReviewAnalysisDashboard />
    </QueryClientProvider>
  );
}

function ReviewAnalysisDashboard() {
  const [selectedSubtheme, setSelectedSubtheme] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState('relevance');
  const [isChatOpen, setIsChatOpen] = useState(false);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  const {
    data: themesData,
    isLoading: isLoadingThemes,
    error: themesError,
  } = useQuery({
    queryKey: ['themes'],
    queryFn: getThemes,
  });

  const {
    data: subthemeData,
    isLoading: isLoadingSubtheme,
  } = useQuery({
    queryKey: ['subtheme', selectedSubtheme, sortBy],
    queryFn: () =>
      selectedSubtheme
        ? getSubthemeReviews(selectedSubtheme, undefined, undefined, sortBy)
        : null,
    enabled: !!selectedSubtheme,
  });

  const handleSubthemeClick = (subtheme: string) => {
    setSelectedSubtheme(subtheme);
  };

  const handleSortChange = (newSortBy: string) => {
    setSortBy(newSortBy);
  };

  if (themesError) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="100vh"
      >
        <Typography color="error">
          Error loading themes. Please try again later.
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ backgroundColor: 'background.default', minHeight: '100vh' }}>
      <AppBar position="fixed" elevation={0}>
        <Toolbar>
          <Box display="flex" alignItems="center" gap={3} sx={{ flexGrow: 1 }}>
            <Box
              display="flex"
              alignItems="center"
              gap={1}
            >
              <Box
                component="span"
                sx={{
                  width: 8,
                  height: 8,
                  backgroundColor: 'secondary.main',
                  borderRadius: '50%',
                  display: 'inline-block',
                }}
              />
              <Typography
                variant="h6"
                component="div"
                sx={{ 
                  fontWeight: 600,
                  display: 'flex',
                  alignItems: 'center',
                }}
              >
                Review Analysis
              </Typography>
            </Box>

            <Box
              display="flex"
              alignItems="center"
              gap={1}
              sx={{ 
                cursor: 'pointer',
                '&:hover': {
                  opacity: 0.8,
                },
              }}
              onClick={() => setIsChatOpen(true)}
            >
              <Box
                component="span"
                sx={{
                  width: 8,
                  height: 8,
                  backgroundColor: '#ff9800',
                  borderRadius: '50%',
                  display: 'inline-block',
                }}
              />
              <Typography
                variant="h6"
                component="div"
                sx={{ 
                  fontWeight: 600,
                  display: 'flex',
                  alignItems: 'center',
                }}
              >
                Chatbot
              </Typography>
            </Box>
          </Box>
        </Toolbar>
      </AppBar>
      <Toolbar /> {/* Spacer for fixed AppBar */}

      <Container maxWidth="xl" sx={{ mt: 3, mb: 4 }}>
        {/* Action Bar */}
        <Box
          display="flex"
          justifyContent="space-between"
          alignItems="center"
          mb={3}
        >
          <Box>
            <Typography variant="h5" gutterBottom>
              Customer Reviews
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Analyze and manage customer feedback across different themes
            </Typography>
          </Box>
          <Box display="flex" gap={2}>
            <Button
              variant="outlined"
              color="inherit"
              startIcon={<FilterListIcon />}
              size="small"
            >
              Filters
            </Button>
            <Button
              variant="outlined"
              color="inherit"
              startIcon={<DownloadIcon />}
              size="small"
            >
              Export
            </Button>
            <Button
              variant="contained"
              color="secondary"
              startIcon={<ChatIcon />}
              size="small"
              onClick={() => setIsChatOpen(true)}
            >
              Chat
            </Button>
          </Box>
        </Box>

        <Divider sx={{ mb: 3 }} />

        <Grid container spacing={3}>
          {/* Themes Section */}
          <Grid item xs={12} md={selectedSubtheme && !isMobile ? 5 : 12}>
            <Box
              sx={{
                height: 'calc(100vh - 280px)',
                display: 'flex',
                flexDirection: 'column',
              }}
            >
              <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                Themes & Subthemes
              </Typography>
              {isLoadingThemes ? (
                <Box display="flex" justifyContent="center" p={4}>
                  <CircularProgress />
                </Box>
              ) : (
                <Box
                  sx={{
                    flexGrow: 1,
                    overflowY: 'auto',
                    pr: 1,
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
                  <Grid container spacing={2}>
                    {themesData?.themes.map((theme) => (
                      <Grid item xs={12} key={theme.id}>
                        <ThemeCard theme={theme} onSubthemeClick={handleSubthemeClick} />
                      </Grid>
                    ))}
                  </Grid>
                </Box>
              )}
            </Box>
          </Grid>

          {/* Subtheme Analysis Section */}
          {selectedSubtheme && (
            <Grid item xs={12} md={isMobile ? 12 : 7}>
              {subthemeData && (
                <SubthemeAnalysis
                  data={subthemeData}
                  isLoading={isLoadingSubtheme}
                  onSortChange={handleSortChange}
                />
              )}
            </Grid>
          )}
        </Grid>
      </Container>

      {/* Chat Dialog */}
      <Dialog
        open={isChatOpen}
        onClose={() => setIsChatOpen(false)}
        maxWidth="md"
        fullWidth
        sx={{
          '& .MuiDialog-paper': {
            height: '80vh',
            maxHeight: '800px',
          },
        }}
      >
        <ChatInterface onClose={() => setIsChatOpen(false)} />
      </Dialog>
    </Box>
  );
}

export default App;
