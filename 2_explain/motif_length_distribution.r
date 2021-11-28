setwd("/media/ggj/Files/mount/NvWA_Final/TFBS/Dmel_scenic/")
data <- read.table("motif_length.txt", header = F)
# 设置插入片段长度的阈值，过滤掉太长的片段
length_cutoff <- 1200
fragment <- data$V1#[data$V1 <= length_cutoff]
# 利用直方图统计频数分布，设置柱子个数
breaks_num <- 50
res <- hist(fragment, breaks = breaks_num, plot = FALSE)
# 添加坐标原点
plot(x = c(0, res$breaks),
     y = c(0, 0, res$counts),
     type = "l", col = "red",
     xlab = "Fragment length(bp)",
     ylab = "counts",
     main = "Sample Fragment sizes ")