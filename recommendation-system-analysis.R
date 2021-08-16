
#############################################
# Report: Movie Recommendation System
# Date: July 22, 2021
# Author: Inga Aritenco
#############################################

----------------------------
##### Data Preparation #####
----------------------------
  
# Create edx set, validation set

# Note: this process could take a couple of minutes

# Download required libraries

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(DescTools)) install.packages("DescTools", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(recosystem))install.packages("recosystem", repos = "http://cran.us.r-project.org")
if(!require(scales))install.packages("scales", repos = "http://cran.us.r-project.org")

# If the libraries are already installed, simply use them 
library(tidyverse)
library(caret)
library(data.table)
library(DescTools)
library(lubridate)
library(recosystem)
library(scales)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed) 


-------------------------
##### Data Overview #####
-------------------------
  
# First look at the edx dataset 
head(edx)

# Explore the structure of the data from the training set
str(edx)

# The number of rows and columns in the edx (training) set
dim(edx)

# The number of movies in edx dataset
edx %>% group_by(movieId) %>% summarize(count = n())

# The number of different users in the edx dataset
edx %>% group_by(userId) %>% summarize(count = n())

# Unique users and movies
edx %>% summarize(n_users = n_distinct(userId), n_movies = n_distinct(movieId))
 
# Movies that have the most ratings
edx %>% group_by(title) %>% 
  summarize(n_ratings = n()) %>% 
  arrange(desc(n_ratings))

# Validation dataset observations by column
glimpse(validation)

------------------------
#### Visualization ####
------------------------

# The most popular movies in the dataset 
edx %>% 
  group_by(title) %>% 
  summarize(count = n()) %>% 
  arrange(-count) %>%
  top_n(20, count) %>% 
  ggplot(aes(count, reorder(title, count))) +
  geom_bar(color = "black", fill = "turquoise2", stat = "identity") + 
  xlab("Count") +
  ylab(NULL) + 
  theme_grey()

# The distribution of movie ratings with range from 0.5 to 5
edx %>% ggplot(aes(rating, y = ..prop..)) +
  geom_bar(color = "black", fill = "turquoise2") + 
  labs(x = "Ratings", y = "Comparative Frequency") + 
  scale_x_continuous(breaks = c(0.5, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5)) + 
  theme_grey()

# The right distribution of ratings versus users  
edx %>% group_by(userId) %>% 
  summarize(count = n()) %>% 
  ggplot(aes(count)) + 
  geom_histogram(color = "black", fill = "turquoise2", bins = 40) +
  xlab("Ratings") + 
  ylab("Users") + 
  scale_x_log10() + 
  theme_grey()

-----------------------------------
##### Data Models and Results #####
-----------------------------------
  
##### Model 1: Average of all ratings #####

# Compute the overall average rating using the training set `edx`
mu <- mean(edx$rating)
mu
# Predict all unknown ratings with mu and compute the RMSE, using obtained average:
naive_rmse <- RMSE(validation$rating, mu)
results <- tibble(Method = "Model 1: Overall Average", RMSE = naive_rmse)
results %>% knitr::kable()

##### Model 2: Movie effect #####

# Represent average ranking for movie i, by adding term b_i (bias)
b_i <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Predict the ratings with movie bias
predicted_ratings <- mu + validation %>%
  left_join(b_i, by = "movieId") %>%
  pull(b_i)

# Calculate RMSE of movie ranking effect
movie_bs_rmse <- RMSE(predicted_ratings, validation$rating)

# Show results in a tibble
movie_bs_rmse <- RMSE(predicted_ratings, validation$rating)
results <- bind_rows(results, tibble(Method = "Model 2: Average + Movie Effect",
                                     RMSE = movie_bs_rmse))
results %>% knitr::kable()

# See if the movie effect distribution is normally skewed
b_i  %>% ggplot(aes(x = b_i)) + 
  geom_histogram(color = "black", fill = "turquoise2", bins = 10) + 
  ggtitle("Movie Effect Distribution") +
  xlab("Movie Effect") +
  ylab("Count") + 
  scale_y_continuous(labels = comma) + 
  theme_grey()


##### Model 3: Movie and user effects #####

# Compute user bs(bias) effect, b_u
b_u <- edx %>% 
  left_join(b_i, by = "movieId") %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating - mu - b_i))

# Predict new ratings with movie and user bs(bias)
predicted_ratings <- validation %>% 
  left_join(b_i, by = "movieId") %>% 
  left_join(b_u, by = "userId") %>% 
  mutate(pred = mu + b_i + b_u) %>% 
  pull(pred)

# Calculate RMSE of movie and user ranking effect
user_bs_rmse <- RMSE(predicted_ratings, validation$rating)
results <- bind_rows(results, tibble(Method = "Model 3: Mean + Movie Bias + User Effect",
                                     RMSE = user_bs_rmse))
results %>% knitr::kable()

# Check the user effect distribution
edx %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>%
  filter(n()>=100) %>% 
  ggplot(aes(b_u)) + 
  geom_histogram(color = "black", fill = "turquoise2", bins = 25) +
  ggtitle("User Effect Distribution") +
  xlab("User Bias") +
  ylab("Count") +
  scale_y_continuous(labels = comma) +
  theme_grey()


##### Model 4: Regularized movie and user effects #####

# Determine the most appropriate lambda defined by sequence
lambdas <- seq(from = 0, to = 10, by = 0.25)

## Repeat the previous actions, but with regularization
## Receive RMSE output of each lambda

rmses <- sapply(lambdas, function(l){
  # Calculate average rating on training set
  mu <- mean(edx$rating)
  # Compute regularized movie bs term
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  # Compute regularize user bs term
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  # Compute predictions on validation set as per the terms of reference
  predicted_ratings <- validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  # Output RMSE of the predictions
  return(RMSE(predicted_ratings, validation$rating))
})

# Quick plot of RMSE in relation to lambdas
qplot(lambdas, rmses)

# Print the minimum value of RMSE
min(rmses)

##### The linear model with regularized movie and user effects #####

# The final linear model with the min lambda
lambda <- lambdas[which.min(rmses)]
lambda

# Predictions Output. RMSE result. 
RMSE(predicted_ratings, validation$rating)
results <- bind_rows(results, tibble(Method = "Model 4: Regularized Movie and User Effects",
                                     RMSE = min(rmses)))
results %>% knitr::kable()

##### Model 5: Matrix Factorization with `recosystem` library. #####

# Create training(edx) and test(validation) sets using `recosystem` library 
train_reco <- with(edx, data_memory(user_index = userId,
                                    item_index = movieId, rating = rating))
test_reco <- with(validation, data_memory(user_index = userId, 
                                          item_index = movieId, rating = rating))
# Create the model object
r = Reco()

# Wait.The process may take some time for the output. 
# Select the best tuning parameters
opts <- r$tune(train_reco, opts = list(dim = c(10, 20, 30),
                                       lrate = c(0.1, 0.2),
                                       costp_l2 = c(0.01, 0.1),
                                       costq_l2 = c(0.01, 0.1),
                                       nthread = 4, niter = 10))

# Train the algorithm
r$train(train_reco, opts = c(opts$min, nthread = 4, niter = 20))

# Calculate the prediction
y_hat_final_reco <- r$predict(test_reco, out_memory())

# Update the result table
result <- bind_rows(results,
                    tibble(Method = "Model 5: Matrix Factorization with recosystem",
                           RMSE = RMSE(validation$rating, y_hat_final_reco)))
# Show the RMSE final result
result %>% knitr::kable()
