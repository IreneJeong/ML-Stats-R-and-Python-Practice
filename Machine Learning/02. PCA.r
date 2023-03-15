library(plyr)
library(ggplot2)
library(GGally)

ansur_data <- read.csv("ansur_2_male_public.csv", header=TRUE, sep=",")

summary(ansur_data)

X <- ansur_data$earlength
Y <- ansur_data$chestcircumference

dframe <- ansur_data[1:100, c(12:15)] #Select only 4 features

names(dframe)

ggpairs(dframe, upper=list(continuous="points"), lower=list(continuous="density"))

ggplot(dframe, mapping = aes(x=chestcircumference, y=earlength)) + geom_point()

cor(dframe)

dataZ = scale(dframe). # Scaling and Centering of Matrix-like Objects
?scale

colMeans(dataZ) # mean of sacled data
apply(dataZ, 2, sd) # Standard deviation calculationb(margin 2)
?apply

cor_mat <- cor(dframe) # Not changed
cor_mat
dataZ

ED <- eigen(cor_mat)
L <- ED$values
V <- ED$vectors
FVE <- L / sum(L) # variance explained

V
V[,1 ] 
# Check whether eigenvectors are in columns or rows
# Try columns - yes, that's correct
check1left <- cor_mat %*% V[, 1] # Covariance matrix %*% eigen vector
check1right <- L[1]*V[, 1] # Lambda * Eigen Vector
check1left
check1right

# Try rows - NO, that's wrong
check2left <- cor_mat %*% V[1, ] 
check2right <- L[1]*V[1, ]
check2left
check2right

FVE #normalized aigen value(lambda)
barplot(FVE, names.arg=c("PC1","PC2","PC3","PC4")) # variance 
loadings <- as.data.frame(V, row.names=names(data), col.names=c("PC1","PC2","PC3","PC4"))
loadings

ansur_pca <- prcomp(dataZ) # Principal Components Analysis
ansur_pca

?prcomp

sqrt(L)
loadings
summary(ansur_pca)

plot(ansur_pca)
biplot(ansur_pca)


Vinv <- solve(V)

Vt <- t(V)
Vt
t(V) %*% V[,1]

cumsum(FVE) # First two variance is good enough to explain the data (0.8678324)
