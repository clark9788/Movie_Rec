"""
Movie Browser - Display MovieLens movies in a paginated table format.
Use this to find movie IDs for rating in the recommender system.
"""

import sys
from utils import initialize_data, format_genres


# Configuration
MOVIES_PER_PAGE = 30  # Adjust this value based on your screen size


def display_movies_page(movies_df, page_num, movies_per_page):
    """
    Display a single page of movies.
    
    Args:
        page_num (int): Page number (0-indexed)
        movies_per_page (int): Number of movies per page
    
    Returns:
        bool: True if there are more pages, False otherwise
    """
    start_idx = page_num * movies_per_page
    end_idx = start_idx + movies_per_page
    
    page_movies = movies_df.iloc[start_idx:end_idx]
    
    if len(page_movies) == 0:
        return False
    
    # Clear screen (works on Windows)
    import os
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Header
    print("=" * 120)
    print(f"MOVIE BROWSER - Page {page_num + 1} (Movies {start_idx + 1}-{min(end_idx, len(movies_df))} of {len(movies_df)})")
    print("=" * 120)
    print(f"{'Movie ID':<10} {'Title':<70} {'Genres':<38}")
    print("-" * 120)
    
    # Display movies
    for _, movie in page_movies.iterrows():
        movie_id = str(movie['movieId'])
        title = movie['title'][:68]  # Truncate if too long
        genres = format_genres(movie['genres'])[:36]  # Truncate if too long
        
        print(f"{movie_id:<10} {title:<70} {genres:<38}")
    
    # Footer
    print("-" * 120)
    has_more = end_idx < len(movies_df)
    if has_more:
        print("Press ENTER for next page, or 'q' + ENTER to quit")
    else:
        print("End of movie list. Press ENTER to return to first page, or 'q' + ENTER to quit")
    print("=" * 120)
    
    return has_more


def main():
    """
    Main function to run the movie browser.
    """
    print("Initializing movie browser...")
    print()
    
    # Load data
    movies_df, _ = initialize_data()
    
    print()
    print("Starting movie browser...")
    print()
    
    # Calculate total pages
    total_pages = (len(movies_df) + MOVIES_PER_PAGE - 1) // MOVIES_PER_PAGE
    
    # Get starting page number from user
    while True:
        try:
            page_input = input(f"Enter Page Number to Start (1-{total_pages}): ").strip()
            if not page_input:
                page_num = 0  # Default to first page if empty input
                break
            page_num = int(page_input) - 1  # Convert to 0-indexed
            if 0 <= page_num < total_pages:
                break
            else:
                print(f"Please enter a page number between 1 and {total_pages}")
        except ValueError:
            print("Please enter a valid number")
    
    # Browse movies
    
    while True:
        has_more = display_movies_page(movies_df, page_num, MOVIES_PER_PAGE)
        
        # Get user input
        user_input = input().strip().lower()
        
        if user_input == 'q':
            print("\nExiting movie browser. Goodbye!")
            break
        
        # Move to next page or loop back to first
        if has_more:
            page_num += 1
        else:
            page_num = 0  # Loop back to first page


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMovie browser interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


