setwd("/media/ggj/Files/mount/NvWA_Final/Predict_whole_genome/Pred_Mouse/")

library(ggthemes)
library(ggplot2)
library(ggpubr)

predicted_random <- read.csv("predicted_null/null_random_background.csv", row.names = 1)
predicted_null<- read.csv("predicted_null/preds_random_location_Mean.csv", row.names = 1)

# pred1 <- as.data.frame(predicted_null[, 1])
pred1 <- as.data.frame(apply(predicted_null, 1, mean))
colnames(pred1) <- ("value")
pred1$type <- "random_location"
dim(pred1);head(pred1)

# pred2 <- as.data.frame(predicted_blank[, 1])
# colnames(pred2) <- ("value")
# pred2$type <- "predicted_blank"
# dim(pred2);head(pred2)

# pred3 <- as.data.frame(predicted_random[, 1])
pred3 <- as.data.frame(apply(predicted_random, 1, mean))
colnames(pred3) <- ("value")
pred3$type <- "random_promoter"
dim(pred3);head(pred3)

pred <- rbind(pred1, pred3)
# sort factor
pred$type <- factor(pred$type, levels = c("random_promoter", "random_location"))
pred <- pred[order(pred$type),]
dim(pred); head(pred)

pdf("Null_Distribution_Boxplot.pdf", width = 6, height = 6)
ggplot(pred, aes(x=type, y=value, fill=type)) + 
  geom_boxplot() + theme_base() +
  theme(axis.text.x = element_text(angle = 90, vjust = .5, size=10)) +
  ylab("Prediction") +
  stat_compare_means(aes(group = type), label = "p.format", label.y = 1) 
dev.off()

pdf("Null_Distribution.pdf", width = 6, height = 4)
ggplot(pred, aes(x=value, fill=type)) + 
  geom_density(alpha=.4) + 
  theme_base() #添加密度曲线
dev.off()