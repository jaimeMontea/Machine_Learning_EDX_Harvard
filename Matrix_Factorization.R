# Regularisation
# 
# Penalized least squares
#
# The general idea behind regularization is to constrain the total variability of the effect sizes. 
# Why does this help? Consider a case in which we have movie i = 1 with 100 user ratings and 4 movies  
# i = 2, 3, 4, 5 with just one user rating. We intend to fit the model
#
# Yu,i = μ + bi + εu,i
#
# Suppose we know the average rating is, say, μ = 3. If we use least squares, the estimate for the first 
# movie effect b1 is the average of the 100 user ratings.which we expect to be a quite precise. However, 
# the estimate for movies 2, 3, 4, and 5 will simply be the observed deviation from the average rating  
# bi = Yu,i − μ which is an estimate based on just one number so it won’t be precise at all. 
# In fact, ignoring the one user and guessing that movies 2,3,4, and 5 are just average movies (bi = 0)
# might provide a better prediction. The general idea of penalized regression is to control the total 
# variability of the movie effects. Specifically, instead of minimizing the least squares equation, 
# we minimize an equation that adds a penalty
#
# ∑u,i(yu,i − μ − bi)2 + λ∑ib2i
#
# The first term is just the sum of squares and the second is a penalty that gets larger when many  
# bi are large. Using calculus we can actually show that the values of bi that minimize this equation are:
# 
# bi(λ) = 1/(λ + ni) * ∑(Yu,i − μ)
# 
# Let’s compute these regularized estimates of bi using λ = 3.

lambda <- 3
mu <- mean(train_set$rating)
movie_reg_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

#
# To see how the estimates shrink, let’s make a plot of the regularized estimates versus the least 
# squares estimates.

tibble(original = movie_avgs$b_i, 
       regularlized = movie_reg_avgs$b_i, 
       n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)

# Do we improve our results?

predicted_ratings <- test_set %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)
RMSE(predicted_ratings, test_set$rating)
#> [1] 0.97

# Note that λ is a tuning parameter and could be fonud by cross-validation. 

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas, rmses)  

# For the full model, the optimal λ is:

lambda <- lambdas[which.min(rmses)]

# ----------------------------------------------------------------------------------------

# Matrix factorization

# Matrix factorization is a widely used concept in machine learning. It is very much related to factor analysis, 
# singular value decomposition (SVD), and principal component analysis (PCA). Here we describe the concept in the 
# context of movie recommendation systems.
# 
# We have described how the model:
#
# Yu,i = μ + bi + bu + εu,i
# 
# accounts for movie to movie differences through the bi and user to user differences through the  
# bu. But this model leaves out an important source of variation related to the fact that groups of movies have similar
# rating patterns and groups of users have similar rating patterns as well. We will discover these patterns by studying the residuals:
#
# ru,i = yu,i − bi − bu
#
# To see this, we will convert the data into a matrix so that each user gets a row, each movie gets a column, and yu,i is the entry in row  
# u and column i. For illustrative purposes, we will only consider a small subset of movies with many ratings and users that have rated 
# many movies. We also keep Scent of a Woman (movieId == 3252) because we use it for a specific example:

train_small <- movielens %>% 
  group_by(movieId) %>%
  filter(n() >= 50 | movieId == 3252) %>% ungroup() %>% 
  group_by(userId) %>%
  filter(n() >= 50) %>% ungroup()

y <- train_small %>% 
  select(userId, movieId, rating) %>%
  pivot_wider(names_from = "movieId", values_from = "rating") %>%
  as.matrix()

# We add row names and column names:

rownames(y)<- y[,1]
y <- y[,-1]

movie_titles <- movielens %>% 
  select(movieId, title) %>%
  distinct()

colnames(y) <- with(movie_titles, title[match(colnames(y), movieId)])

# and convert them to residuals by removing the column and row effects:

y <- sweep(y, 2, colMeans(y, na.rm=TRUE))
y <- sweep(y, 1, rowMeans(y, na.rm=TRUE))

# If the model above explains all the signals, and the ε are just noise, then the residuals for different movies should 
# be independent from each other. But they are not. Here are some examples:

m_1 <- "Godfather, The"
m_2 <- "Godfather: Part II, The"
p1 <- qplot(y[ ,m_1], y[,m_2], xlab = m_1, ylab = m_2)

m_1 <- "Godfather, The"
m_3 <- "Goodfellas"
p2 <- qplot(y[ ,m_1], y[,m_3], xlab = m_1, ylab = m_3)

m_4 <- "You've Got Mail" 
m_5 <- "Sleepless in Seattle" 
p3 <- qplot(y[ ,m_4], y[,m_5], xlab = m_4, ylab = m_5)

gridExtra::grid.arrange(p1, p2 ,p3, ncol = 3)

# This plot says that users that liked The Godfather more than what the model expects them to, based on the movie and
# user effects, also liked The Godfather II more than expected. A similar relationship is seen when comparing The Godfather 
# and Goodfellas. Although not as strong, there is still correlation. We see correlations between You’ve Got Mail and Sleepless 
# in Seattle as well. 
#
# By looking at the correlation between movies, we can see a pattern (we rename the columns to save print space):

x <- y[, c(m_1, m_2, m_3, m_4, m_5)]
short_names <- c("Godfather", "Godfather2", "Goodfellas",
                 "You've Got", "Sleepless")
colnames(x) <- short_names
cor(x, use="pairwise.complete")
#>            Godfather Godfather2 Goodfellas You've Got Sleepless
#> Godfather      1.000      0.829      0.444     -0.440    -0.378
#> Godfather2     0.829      1.000      0.521     -0.331    -0.358
#> Goodfellas     0.444      0.521      1.000     -0.481    -0.402
#> You've Got    -0.440     -0.331     -0.481      1.000     0.533
#> Sleepless     -0.378     -0.358     -0.402      0.533     1.000

# There seems to be people that like romantic comedies more than expected, while others that like gangster movies more than expected.
# These results tell us that there is structure in the data. But how can we model this?

#  Factor analysis

# Here is an illustration, using a simulation, of how we can use some structure to predict the ru,i. 
# Suppose our residuals r look like this:

round(r, 1)
#>    Godfather Godfather2 Goodfellas You've Got Sleepless
#> 1        2.0        2.3        2.2       -1.8      -1.9
#> 2        2.0        1.7        2.0       -1.9      -1.7
#> 3        1.9        2.4        2.1       -2.3      -2.0
#> 4       -0.3        0.3        0.3       -0.4      -0.3
#> 5       -0.3       -0.4        0.3        0.2       0.3
#> 6       -0.1        0.1        0.2       -0.3       0.2
#> 7       -0.1        0.0       -0.2       -0.2       0.3
#> 8        0.2        0.2        0.1        0.0       0.4
#> 9       -1.7       -2.1       -1.8        2.0       2.4
#> 10      -2.3       -1.8       -1.7        1.8       1.7
#> 11      -1.7       -2.0       -2.1        1.9       2.3
#> 12      -1.8       -1.7       -2.1        2.3       2.0

# There seems to be a pattern here. In fact, we can see very strong correlation patterns:

cor(r) 
#>            Godfather Godfather2 Goodfellas You've Got Sleepless
#> Godfather      1.000      0.980      0.978     -0.974    -0.966
#> Godfather2     0.980      1.000      0.983     -0.987    -0.992
#> Goodfellas     0.978      0.983      1.000     -0.986    -0.989
#> You've Got    -0.974     -0.987     -0.986      1.000     0.986
#> Sleepless     -0.966     -0.992     -0.989      0.986     1.000

# We can create vectors q and p, that can explain much of the structure we see. The q would look like this:

t(q) 
#>      Godfather Godfather2 Goodfellas You've Got Sleepless
#> [1,]         1          1          1         -1        -1

# and it narrows down movies to two groups: gangster (coded with 1) and romance (coded with -1). We can also reduce the users to three groups:

t(p)
#>      1 2 3 4 5 6 7 8  9 10 11 12
#> [1,] 2 2 2 0 0 0 0 0 -2 -2 -2 -2

# those that like gangster movies and dislike romance movies (coded as 2), those that like romance movies and dislike gangster movies (coded as -2), 
# and those that don’t care (coded as 0). The main point here is that we can almost reconstruct r, which has 60 values, with a couple of vectors
# totaling 17 values. Note that p and q are equivalent to the patterns and weights we described in Section 33.5.4.
#
# If r contains the residuals for users u = 1,…,12 for movies i = 1,…,5 we can write the following mathematical formula for our residuals ru,i.
# 
# ru,i ≈ puqi
#
# This implies that we can explain more variability by modifying our previous model for movie recommendations to:
#
# Yu,i = μ + bi + bu + puqi + εu,i
# 
# However, we motivated the need for the puqi term with a simple simulation. The structure found in data is usually more complex. For example,
# in this first simulation we assumed there were was just one factor pu that determined which of the two genres movie u belongs to. But the
# structure in our movie data seems to be much more complicated than gangster movie versus romance. We may have many other factors. Here we 
# present a slightly more complex simulation. We now add a sixth movie.

round(r, 1)
#>    Godfather Godfather2 Goodfellas You've Got Sleepless Scent
#> 1        0.5        0.6        1.6       -0.5      -0.5  -1.6
#> 2        1.5        1.4        0.5       -1.5      -1.4  -0.4
#> 3        1.5        1.6        0.5       -1.6      -1.5  -0.5
#> 4       -0.1        0.1        0.1       -0.1      -0.1   0.1
#> 5       -0.1       -0.1        0.1        0.0       0.1  -0.1
#> 6        0.5        0.5       -0.4       -0.6      -0.5   0.5
#> 7        0.5        0.5       -0.5       -0.6      -0.4   0.4
#> 8        0.5        0.6       -0.5       -0.5      -0.4   0.4
#> 9       -0.9       -1.0       -0.9        1.0       1.1   0.9
#> 10      -1.6       -1.4       -0.4        1.5       1.4   0.5
#> 11      -1.4       -1.5       -0.5        1.5       1.6   0.6
#> 12      -1.4       -1.4       -0.5        1.6       1.5   0.6

# By exploring the correlation structure of this new dataset

colnames(r)[4:6] <- c("YGM", "SS", "SW")
cor(r)
#>            Godfather Godfather2 Goodfellas    YGM     SS     SW
#> Godfather      1.000      0.997      0.562 -0.997 -0.996 -0.571
#> Godfather2     0.997      1.000      0.577 -0.998 -0.999 -0.583
#> Goodfellas     0.562      0.577      1.000 -0.552 -0.583 -0.994
#> YGM           -0.997     -0.998     -0.552  1.000  0.998  0.558
#> SS            -0.996     -0.999     -0.583  0.998  1.000  0.588
#> SW            -0.571     -0.583     -0.994  0.558  0.588  1.000

# We note that we perhaps need a second factor to account for the fact that some users like Al Pacino, while others dislike him or
# don’t care. Notice that the overall structure of the correlation obtained from the simulated data is not that far off the real correlation:

six_movies <- c(m_1, m_2, m_3, m_4, m_5, m_6)
x <- y[, six_movies]
colnames(x) <- colnames(r)
cor(x, use="pairwise.complete")
#>            Godfather Godfather2 Goodfellas    YGM     SS      SW
#> Godfather     1.0000      0.829      0.444 -0.440 -0.378  0.0589
#> Godfather2    0.8285      1.000      0.521 -0.331 -0.358  0.1186
#> Goodfellas    0.4441      0.521      1.000 -0.481 -0.402 -0.1230
#> YGM          -0.4397     -0.331     -0.481  1.000  0.533 -0.1699
#> SS           -0.3781     -0.358     -0.402  0.533  1.000 -0.1822
#> SW            0.0589      0.119     -0.123 -0.170 -0.182  1.0000

# To explain this more complicated structure, we need two factors. For example something like this:

t(q) 
#>      Godfather Godfather2 Goodfellas You've Got Sleepless Scent
#> [1,]         1          1          1         -1        -1    -1
#> [2,]         1          1         -1         -1        -1     1

# With the first factor (the first row) used to code the gangster versus romance groups and a second factor (the second row) to explain the 
# Al Pacino versus no Al Pacino groups. We will also need two sets of coefficients to explain the variability introduced by the 3 × 3 types of groups:

t(p)
#>         1   2   3 4 5   6   7   8  9   10   11   12
#> [1,]  1.0 1.0 1.0 0 0 0.0 0.0 0.0 -1 -1.0 -1.0 -1.0
#> [2,] -0.5 0.5 0.5 0 0 0.5 0.5 0.5  0 -0.5 -0.5 -0.5

# The model with two factors has 36 parameters that can be used to explain much of the variability in the 72 ratings:
# 
# Yu,i = μ + bi + bu + pu,1q1,i + pu,2q2,i + εu,i
# 
# Note that in an actual data application, we need to fit this model to data. To explain the complex correlation we observe in real data,
# we usually permit the entries of p and q to be continuous values, rather than discrete ones as we used in the simulation. For example,
# rather than dividing movies into gangster or romance, we define a continuum. Also note that this is not a linear model and to fit it we 
# need to use an algorithm other than the one used by lm to find the parameters that minimize the least squares. The winning algorithms for
# the Netflix challenge fit a model similar to the above and used regularization to penalize for large values of p and q  rather than using 
# least squares. Implementing this approach is beyond the scope of this book.
#
# Connection to SVD and PCA
#
# The decomposition: 
#
# ru,i ≈ pu,1q1,i + pu,2q2,i
# 
# is very much related to SVD and PCA. SVD and PCA are complicated concepts, but one way to understand them is that SVD is an 
# algorithm that finds the vectors p and q that permit us to rewrite the matrix r with m rows and n columns as:
#
# ru,i = pu,1q1,i + pu,2q2,i + ⋯ + pu,nqn,i
#
# with the variability of each term decreasing and with the p s uncorrelated. The algorithm also computes this variability so
# that we can know how much of the matrices, total variability is explained as we add new terms. This may permit us to see that,
# with just a few terms, we can explain most of the variability.
# Let’s see an example with the movie data. To compute the decomposition, we will make the residuals with NAs equal to 0:

y[is.na(y)] <- 0
pca <- prcomp(y)

# The q vectors are called the principal components and they are stored in this matrix:
# 

dim(pca$rotation)
#> [1] 454 292

# While the p, or the user effects, are here:

dim(pca$x)
#> [1] 292 292

# We can see the variability of each of the vectors:

qplot(1:nrow(x), pca$sdev, xlab = "PC")

# ---------------------------------------------------------------------------------------------------------------------
# Comprehension Check: Matrix Factorization
#
# In this exercise set, we will be covering a topic useful for understanding matrix factorization: the singular value decomposition (SVD).
# SVD is a mathematical result that is widely used in machine learning, both in practice and to understand the mathematical properties of
# some algorithms. This is a rather advanced topic and to complete this exercise set you will have to be familiar with linear algebra 
# concepts such as matrix multiplication, orthogonal matrices, and diagonal matrices.
#
# The SVD tells us that we can decompose an N x p matrix Y with p < N as:
#
# Y = UDV
#
# with U and V orthogonal of dimensions N x p and p x p espectively and D a p x p diagonal matrix with the values of the diagonal decreasing: :
#
# d1,1 > d2,2 > d3,3 ...dp,p
#
# In this exercise, we will see one of the ways that this decomposition can be useful. To do this, we will construct a dataset that 
# represents grade scores for 100 students in 24 different subjects. The overall average has been removed so this data represents the 
# percentage point each student received above or below the average test score. So a 0 represents an average grade (C), a 25 is a high 
# grade (A+), and a -25 represents a low grade (F). You can simulate the data like this:

set.seed(1987, sample.kind="Rounding")
n <- 100
k <- 8
Sigma <- 64  * matrix(c(1, .75, .5, .75, 1, .5, .5, .5, 1), 3, 3) 
m <- MASS::mvrnorm(n, rep(0, 3), Sigma)
m <- m[order(rowMeans(m), decreasing = TRUE),]
y <- m %x% matrix(rep(1, k), nrow = 1) + matrix(rnorm(matrix(n*k*3)), n, k*3)
colnames(y) <- c(paste(rep("Math",k), 1:k, sep="_"),
                 paste(rep("Science",k), 1:k, sep="_"),
                 paste(rep("Arts",k), 1:k, sep="_"))
 
# Our goal is to describe the student performances as succinctly as possible. For example, we want to know if these test results are all 
# just a random independent numbers. Are all students just about as good? Does being good in one subject  imply you will be good in another? 
# How does the SVD help with all this? We will go step by step to show that with just three relatively small pairs of vectors we can explain
# much of the variability in this 100 x 24 dataset.
#
# Q1
# You can visualize the 24 test scores for the 100 students by plotting an image:

my_image <- function(x, zlim = range(x), ...){
  colors = rev(RColorBrewer::brewer.pal(9, "RdBu"))
  cols <- 1:ncol(x)
  rows <- 1:nrow(x)
  image(cols, rows, t(x[rev(rows),,drop=FALSE]), xaxt = "n", yaxt = "n",
        xlab="", ylab="",  col = colors, zlim = zlim, ...)
  abline(h=rows + 0.5, v = cols + 0.5)
  axis(side = 1, cols, colnames(x), las = 2)
}

my_image(y)

# How would you describe the data based on this figure?

# The students that test well are at the top of the image and there seem to be three groupings by subject.

# Q2

# You can examine the correlation between the test scores directly like this:

my_image(cor(y), zlim = c(-1,1))
range(cor(y))
axis(side = 2, 1:ncol(y), rev(colnames(y)), las = 2)

# There is correlation among all tests, but higher if the tests are in science and math and even higher within each subject.

# Q3

# Remember that orthogonality means that  and  are equal to the identity matrix. This implies that we can also rewrite the decomposition a
# We can think of  and  as two transformations of  that preserve the total variability of  since  and  are orthogonal.
# Use the function svd() to compute the SVD of y. This function will return , , and the diagonal entries of .

s <- svd(y)
names(s)

y_svd <- s$u %*% diag(s$d) %*% t(s$v)
max(abs(y - y_svd))

y_sq <- y*y 
ss_y <- colSums(y_sq)
sum(ss_y) 

y_svd_sq <- y_svd*y_svd 
ss_yv <- colSums(y_svd_sq)
sum(ss_yv) 

# Compute the sum of squares of the columns of  and store them in ss_y. Then compute the sum of squares of columns of the transformed
# and store them in ss_yv. Confirm that sum(ss_y) is equal to sum(ss_yv). What is the value of sum(ss_y) (and also the value of sum(ss_yv))?

# Q4

# We see that the total sum of squares is preserved. This is because  is orthogonal. Now to start understanding how  is useful, 
# plot ss_y against the column number and then do the same for ss_yv.

# What do you observe?

plot(ss_y) 
plot(ss_yv)

#Q5

# Now notice that we didn't have to compute ss_yv because we already have the answer. How? Remember that  and because  is orthogonal, 
# we know that the sum of squares of the columns of  are the diagonal entries of  squared. Confirm this by plotting the square root of
#ss_yv versus the diagonal entries of .

# Which of these plots is correct?

plot(sqrt(ss_yv), s$d)
abline(0,1)

plot(sqrt(ss_yv), s$d, xlab="x", ylab="y", xlim=c(0,max(ss_yv)), y_lim=c(0,max(s$d)))

#Q6

# So from the above we know that the sum of squares of the columns of  (the total sum of squares) adds up to the sum of s$d^2 and that 
# the transformation  gives us columns with sums of squares equal to s$d^2. Now compute the percent of the total variability that is 
# explained by just the first three columns of .

# What proportion of the total variability is explained by the first three columns of ?

sum(s$d[1:3]^2) / sum(s$d^2)

#Q7

identical(s$u %*% diag(s$d), sweep(s$u, 2, s$d, FUN = "*"))


#Q8

plot(-s$u[,1]*s$d[1], rowMeans(y))
