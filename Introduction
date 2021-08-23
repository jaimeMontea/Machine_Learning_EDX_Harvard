#Recomendation Systems

# Recommendation systems use ratings that users have given items to make specific recommendations.
# Items for which a high rating is predicted for a given user are then recommended to that user.
#
# The following project consists on a challenge offered by Netflix in October 2006: improve their
# recommendation algorithm by 10% and win a million dollars. 
# In September 2009 the winners were announced. The following articles show a detail explanation of their algorithm: 
# http://blog.echen.me/2011/10/24/winning-the-netflix-prize-a-summary/
# http://www.netflixprize.com/assets/GrandPrize2009_BPC_BellKor.pdf
#
# The Netflix data is not publicly available, but the GroupLens research lab112 generated their own database with 
# over 20 million ratings for over 27,000 movies by more than 138,000 users. A small dataset is available in the 
# dslabs package.

library("tidyverse")
library("dslabs")
data("movielens")

# We can see this table is in tidy format with thousands of rows:

movielens %>% as_tibble()

#> # A tibble: 100,004 x 7
#>   movieId title              year genres         userId rating timestamp
#>     <int> <chr>             <int> <fct>           <int>  <dbl>     <int>
#> 1      31 Dangerous Minds    1995 Drama               1    2.5    1.26e9
#> 2    1029 Dumbo              1941 Animation|Chi…      1    3      1.26e9
#> 3    1061 Sleepers           1996 Thriller            1    3      1.26e9
#> 4    1129 Escape from New …  1981 Action|Advent…      1    2      1.26e9
#> 5    1172 Cinema Paradiso …  1989 Drama               1    4      1.26e9
#> # … with 99,999 more rows

# We can see the number of unique users that provided ratings and how many unique movies were rated:

movielens %>% 
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

#>   n_users n_movies
#> 1     671     9066

# If we multiply those two numbers, we get a number larger than 5 million, yet our data table has about
# 100,000 rows. This implies that not every user rated every movie. So we can think of these data as a 
# very large matrix, with users on the rows and movies on the columns, with many empty cells. 
#
# This complicated machine learning challenge because each outcome Y has a different set of predictors. 
# To see this, note that if we are predicting the rating for movie i by user u, in principle, all other ratings
# related to movie i and by user u may be used as predictors, but different users rate different movies and a 
# different number of movies. Furthermore, we may be able to use information from other movies that we have 
# determined are similar to movie i or from users determined to be similar to user u. In essence, the entire 
# matrix can be used as predictors for each cell.
#
# Let’s create a test set to assess the accuracy of the models we implement.

set.seed(755)
index <- createDataPartition(y = movielens$rating, times=1, p=0.2, list=FALSE)
test_set <- movielens[index, ]
train_set <- movielens[-index, ]

# To make sure we don’t include users and movies in the test set that do not appear in the training set, 
# we remove these entries using the semi_join function:

test_set <- test_set %>% semi_join(train_set, by="movieId") %>%
                     semi_join(train_set, by="userId")

# The Netflix challenge used the typical error loss: they decided on a winner based on the residual 
# mean squared error (RMSE) on a test set.
# 
# Let’s write a function that computes the RMSE for vectors of ratings and their corresponding predictors:

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# The winners of the Netflix challenge used two types of models: one similar to k-nearest neighnors where
# you found movies that were similar to each other and users that were similar to each other.
# The other one was based on an approach called matrix factorization, which is going to be explainde in this section. 
#
# Matrix Factorization
#
# Let’s start by building the simplest possible recommendation system: we predict the same rating for all movies 
# regardless of user. What number should this prediction be? We can use a model based approach to answer this. 
# A model that assumes the same rating for all movies and users with all the differences explained by random 
# variation would look like this:
#
# Yu,i =  μ + εu,i
#
# with εi,u independent errors sampled from the same distribution centered at 0 and μ the “true” rating 
# for all movies. We know that the estimate that minimizes the RMSE is the least squares estimate of μ
# and, in this case, is the average of all ratings:

mu <- mean(train_set$rating)
naive_rmse <- RMSE(test_set$rating, mu_hat)

#> [1] 1.05

# As we go along, we will be comparing different approaches. Let’s start by creating a results table 
# with this naive approach:

rmse_results <- tibble(method = "Just the average", RMSE = naive_rmse)

# We know from experience that some movies are just generally rated higher than others. This intuition,
# that different movies are rated differently, is confirmed by data. We can augment our previous model by
# adding the term bi to represent average ranking for movie i:
# 
# Yu,i = μ + bi + εu,i
#
# In this particular situation, we know that the least squares estimate bi is just the average of  
# Yu,i − μ for each movie i. So we can compute them this way (we will drop the hat notation in the code 
# to represent estimates going forward):

movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# We can see that these estimates vary substantially:

qplot(b_i, data = movie_avgs, bins = 10, color = I("black"))

# Let’s see how much our prediction improves once we use:

predicted_ratings <- mu + test_set %>%
                    left_join(movie_avgs, by="movieId") %>%
                    pull(b_i)

movie_approach <- RMSE(predicted_ratings, test_set$rating)

#> [1] 0.989

# Let's add this result to the table:

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",
                                     RMSE = movie_approach ))

# Let’s compute the average rating for user u
# for those that have rated 100 or more movies:
  
train_set %>% group_by(userId) %>%
              filter(n()>100) %>%
              summarise(b_u=mean(rating)) %>%
              ggplot(aes(b_u)) +
              geom_histogram(bins=30, color="black")

# Notice that there is substantial variability across users as well: some users are very cranky and others
# love every movie. This implies that a further improvement to our model may be:
# 
# Yu,i = μ + bi + bu + εu,i
#
# We qre going to estimate bu as follows:

user_avgs <- train_set %>%
             left_join(movie_avgs, by="movieId") %>%
             group_by(userId) %>%
             summarize(b_u = mean(rating -mu - b_i))

# We can now construct predictors and see how much the RMSE improves:

predicted_ratings <- test_set %>%
                      left_join(movie_avgs, by='movieId') %>%
                      left_join(user_avgs, by='userId') %>%
                      mutate(pred = mu + b_i + b_u) %>%
                      pull(pred)

user_approach <- RMSE(predicted_ratings, test_set$rating)
#> [1] 0.905

# Let's add this result to the table:

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",
                                     RMSE = movie_approach ))
