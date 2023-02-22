library(plyr)

auto_data <- readRDS(file="auto.rds")
auto_data
?readRDS

auto_data <- rename(auto_data, c(
  "V1"="MPG",
  "V2"="Cylinders",
  "V3"="Displacement",
  "V4"="Horsepower",
  "V5"="Weight",
  "V6"="Acceleration",
  "V7"="ModelYear",
  "V8"="Origin",
  "V9"="CarName"
  ))

# Code for PCA

auto_pca <- prcomp(auto_data[,c(1,3,5,6)], scale = TRUE) # PCA with scale
summary(auto_pca)
# pdf("../figures/auto_pc.pdf", width=6, height=3)
plot(auto_pca)
# dev.off()
# pdf("../figures/auto_bi.pdf", width=6, height=6)
biplot(auto_pca)
# dev.off()

# Code for CCA

auto_cca <- cancor(scale(auto_data[,c(3,5)]),scale(auto_data[,c(1,6)]))
?cancor

beta <- (auto_cca$xcoef[,1])/sqrt(sum(auto_cca$xcoef[,1]**2))
eta <- (auto_cca$ycoef[,1])/sqrt(sum(auto_cca$ycoef[,1]**2))
beta
eta
