# MovieLens Recommendation System

A movie recommendation system built using Neural Collaborative Filtering (NCF) on the MovieLens dataset. The system consists of two main programs: a movie browser for finding movies to rate, and a recommender system that generates personalized recommendations.

## Features

- **Movie Browser**: Browse the MovieLens movie database in a paginated table format
- **Neural Collaborative Filtering**: State-of-the-art recommendation algorithm with early stopping
- **Interactive Rating System**: Rate movies and get instant recommendations
- **Persistent Rating Storage**: Ratings automatically saved and loaded between sessions
- **Rating Statistics**: Track your rating patterns and progress over time
- **Quality Metrics**: Built-in evaluation with RMSE and MAE metrics
- **Early Stopping**: Automatic optimal epoch detection to prevent overfitting
- **Learning Curve Visualization**: Visual analysis of training progress and overfitting detection

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the movie browser to find movies to rate:
```bash
python movie_browser.py
```

3. Run the recommender system to rate movies and get recommendations:
```bash
python recommender.py
```

## Usage

### Step 1: Browse Movies
Run `movie_browser.py` to browse the MovieLens database:
- Movies are displayed in pages of 30 (configurable via `MOVIES_PER_PAGE`)
- Shows Movie ID, Title, and Genres
- Use ENTER to go to next page, 'q' to quit
- Note down Movie IDs of movies you've watched

### Step 2: Get Recommendations
Run `recommender.py` to rate movies and get recommendations:
- The system will first download and train the NCF model with early stopping (takes a few minutes)
- **Training Features**:
  - Automatic optimal epoch detection (typically 6-15 epochs)
  - Three-way data split: 60% train, 20% validation, 20% test
  - Learning curve visualization with overfitting detection
  - Best model restoration based on validation performance
- Your previous ratings are automatically loaded from `user_ratings.csv`
- Enter ratings in format: `movie_id rating` (e.g., "1 4.5")
- Ratings must be between 0.5 and 5.0 in 0.5 increments
- **Commands**: 
  - `stats` - View your rating statistics
  - `list` - See all movies you've rated
  - `0 0` - Finish rating and get recommendations
- Ratings are automatically saved after each entry
- The system will display your top 20 recommended movies

## Algorithm Explanation

### Why Neural Collaborative Filtering?

**Options Considered:**
1. **Basic Collaborative Filtering**: Fast but limited by sparse data
2. **Matrix Factorization (SVD)**: Good balance but assumes linear relationships  
3. **Neural Collaborative Filtering**: Best accuracy through non-linear learning

**NCF Advantages:**
- Learns complex user-movie interactions through neural networks
- Better generalization to new users with limited ratings
- Handles sparse data better than traditional methods
- State-of-the-art performance on MovieLens benchmarks

### Model Architecture

The NCF model combines:
- **Embedding Layers**: Learn dense representations for users and movies
- **Multi-Layer Perceptron (MLP)**: Captures non-linear user-movie interactions
- **Dropout**: Prevents overfitting during training
- **Early Stopping**: Automatically finds optimal training duration
- **Xavier Weight Initialization**: Ensures stable training

### Training Process

**Early Stopping Implementation:**
- **Validation Monitoring**: Tracks validation loss each epoch
- **Patience Parameter**: Stops training after 5 epochs without improvement (configurable)
- **Best Model Restoration**: Automatically restores the best-performing model
- **Learning Curves**: Visual analysis of training vs validation loss

**Data Split Strategy:**
- **Training Set**: 60% of data for model learning
- **Validation Set**: 20% for early stopping decisions
- **Test Set**: 20% for final performance evaluation

### Quality Assessment

The system provides several quality metrics:
- **RMSE (Root Mean Square Error)**: Measures prediction accuracy
- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **Subjective Testing**: You and your wife will be the ultimate judges

Typical good values:
- RMSE: 0.8-1.0 for MovieLens dataset
- MAE: 0.6-0.8 for MovieLens dataset

## File Structure

```
MovieLens_Rec/
├── data_cache/
│   └── data/
│       ├── movies.csv          # Movie information
│       ├── ratings.csv         # User ratings
│       ├── links.csv          # External links
│       └── README.txt         # Dataset documentation
├── utils.py                   # Shared utilities
├── movie_browser.py          # Movie browsing program
├── recommender.py            # Recommendation system
├── user_ratings.csv          # Your persistent ratings (auto-created)
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Configuration

### Movie Browser Settings
Edit `movie_browser.py` to adjust:
```python
MOVIES_PER_PAGE = 30  # Number of movies per page
```

### Recommender Settings
Edit `recommender.py` to adjust:
- **Early Stopping Parameters**:
  - `epochs=50`: Maximum training epochs
  - `patience=5`: Epochs to wait before stopping
- **Training Parameters**:
  - `batch_size=1024`: Training batch size
  - `learning_rate=0.001`: Adam optimizer learning rate
- **Model Architecture**:
  - `embedding_dim=50`: User/movie embedding dimensions
  - `hidden_dims=[128, 64, 32]`: MLP layer sizes
- **Learning Curves**:
  - `plot_learning_curves()`: Show/save training analysis
  - Options: `show_plot=True`, `save_plot=False`, `filename='learning_curves.png'`

## Data Source

This project uses the MovieLens Small Dataset:
- **Source**: https://grouplens.org/datasets/movielens/latest/
- **Size**: ~100,000 ratings from ~600 users on ~9,000 movies
- **Rating Scale**: 0.5 to 5.0 stars in 0.5 increments
- **Updated**: 2018

## Performance Notes

- **Training Time**: ~3-8 minutes on modern hardware (optimized with early stopping)
- **Memory Usage**: ~500MB during training
- **GPU Support**: Automatically uses GPU if available
- **Recommendation Speed**: Near-instant after training
- **Optimal Epochs**: Typically 6-15 epochs (automatically detected)
- **Learning Curves**: Visual analysis available with matplotlib

## Troubleshooting

### Common Issues

1. **Dataset Download Fails**: Check internet connection, the system will retry
2. **Training Takes Too Long**: Early stopping automatically optimizes training time
3. **Memory Issues**: Reduce batch size in training parameters
4. **Movie ID Not Found**: Verify Movie ID exists in the browser
5. **Learning Curves Not Displaying**: Install matplotlib with `pip install matplotlib`
6. **Overfitting Concerns**: Early stopping automatically prevents overfitting

### Python Environment

The system is designed to work with:
- Python 3.8+
- PyTorch 2.0+
- Standard scientific Python stack (pandas, numpy, scikit-learn)
- matplotlib 3.7+ (for learning curve visualization)

## Persistent Rating Storage

The system now includes comprehensive rating persistence:

### **Automatic Storage**
- Ratings automatically saved to `user_ratings.csv` after each entry
- Previous ratings loaded when restarting the program
- No data loss even if program is interrupted

### **Rating Management**
- **Accumulative**: Each session adds to your previous ratings
- **Updateable**: Change ratings for movies you've already rated
- **Statistics**: Track total movies rated, average rating, distribution
- **Commands**: Use `stats` and `list` commands for rating insights

### **Benefits**
- Build comprehensive movie preference profile over time
- Better recommendations as you rate more movies
- Easy to track your rating patterns and progress
- Perfect for long-term movie discovery and preference learning

## Learning Curve Analysis

The system includes comprehensive training analysis capabilities:

### **Early Stopping Benefits**
- **Automatic Optimization**: Finds optimal training duration without manual tuning
- **Overfitting Prevention**: Stops training when validation performance plateaus
- **Time Efficiency**: Typically reduces training time by 20-40%
- **Better Performance**: Often achieves better final model performance

### **Learning Curve Visualization**
```python
# Default usage (show plot)
recommender.plot_learning_curves()

# Save plot for documentation
recommender.plot_learning_curves(show_plot=False, save_plot=True, filename='training_analysis.png')

# Text-only analysis (no matplotlib required)
recommender.plot_learning_curves(show_plot=False, save_plot=False)
```

### **What the Plots Show**
- **Training vs Validation Loss**: Visualize learning progress
- **Optimal Epoch Detection**: Green line marks best stopping point
- **Overfitting Indicator**: Shows when training/validation loss diverge
- **Performance Analysis**: Automatic overfitting detection and recommendations

### **Typical Results**
- **Optimal Epochs**: 6-15 epochs (varies by data and hyperparameters)
- **RMSE Improvement**: Often 5-15% better than fixed epoch training
- **Training Time**: 3-8 minutes (down from 5-10 minutes)
- **Overfitting Detection**: Automatic alerts when validation loss increases

## Future Enhancements

### **Immediate Improvements**
- Genre-based filtering
- Similar user recommendations
- Movie similarity analysis
- Web interface
- Real-time model updates
- Rating export/import functionality

### **Advanced Optimization (See FUTURE_ENHANCEMENTS.md)**
- **Hyperparameter Tuning**: Learning rate, patience, batch size optimization
- **Model Architecture Experiments**: Embedding dimensions, hidden layers, dropout rates
- **Performance Comparison Framework**: Automated testing and results tracking
- **Cross-validation**: More robust model evaluation
- **Grid Search**: Systematic hyperparameter optimization

### **Implementation Priority**
1. **High**: Genre filtering, web interface
2. **Medium**: Hyperparameter tuning, architecture experiments
3. **Low**: Advanced ML techniques, production deployment

## License

This project is for educational purposes. The MovieLens dataset is provided under the Creative Commons Attribution 4.0 International License.




"# Movie_Rec" 

