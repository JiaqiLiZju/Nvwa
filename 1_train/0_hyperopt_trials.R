setwd("/media/ggj/Files/NvWA/nvwa-pse-official/hyper_params_tune/")

library(latticeExtra)
library(ggplot2)
library(reshape2)

getresults = function(thisfile){
    sites = read.table(thisfile, sep='\t')
    colnames(sites)=c("leftpos","rightpos","loss")
    print(sites[which(sites$loss == min(sites$loss)),])
    # print(nrow(sites), "trials")
    return(sites)
}

c = getresults("Anneal-lr5-epoch10-EPOCH100/hyperopt_trails/params.txt")
t = getresults("Anneal-lr5-epoch10/hyperopt_trails/params.txt")
e = getresults("TPE-lr4-epoch5/hyperopt_trails/params.txt")
f = getresults("TPE-lr5-epoch10/hyperopt_trails/params.txt")

df = data.frame(Data = sapply(1:nrow(c), function(x) min(c[1:x, "loss"])), Idc="Anneal-lr5-epoch10-EPOCH100", Idx=1:nrow(c))
df = rbind(df, data.frame(Data = sapply(1:nrow(t), function(x) min(t[1:x, "loss"])), Idc="Anneal-lr5-epoch10", Idx=1:nrow(t)))
df = rbind(df, data.frame(Data = sapply(1:nrow(e), function(x) min(e[1:x, "loss"])), Idc="TPE-lr4-epoch5", Idx=1:nrow(e)))
df = rbind(df, data.frame(Data = sapply(1:nrow(f), function(x) min(f[1:x, "loss"])), Idc="TPE-lr5-epoch10", Idx=1:nrow(f)))
colnames(df) <- c("Validation Loss", "Hyperparameters Optimation Algorithms", "Number of Iterations")

pdf("Fig.pdf", width=5, height=4)
ggplot(df, aes(x=Idx, y=Data, color=Idc, group=Idc)) + 
  geom_line(size=0.8) +
  labs(title = "Hyperparameters Optimation",
    color='Algorithms',
    x='Validation Loss',
    y="Number of Iterations") + 
  theme(legend.position = c(0.7, 0.75),
        plot.title = element_text(color = "black", size = 14, face = "bold.italic"),
        axis.title.x =  element_text(color = "black", size = 8, face = "bold"),
        axis.title.y =  element_text(color = "black", size = 8, face = "bold")) +
  scale_fill_discrete(name="types")
dev.off()
