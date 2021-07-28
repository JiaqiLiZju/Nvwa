library(Seurat)
library(data.table)
as_matrix <- function(mat){
  
  tmp <- matrix(data=0L, nrow = mat@Dim[1], ncol = mat@Dim[2])
  
  row_pos <- mat@i+1
  col_pos <- findInterval(seq(mat@x)-1,mat@p[-1])+1
  val <- mat@x
  
  for (i in seq_along(val)){
    tmp[row_pos[i],col_pos[i]] <- val[i]
  }
  
  row.names(tmp) <- mat@Dimnames[[1]]
  colnames(tmp) <- mat@Dimnames[[2]]
  return(tmp)
}

load("seurat_object_merged_Singlet_used.rdata")
options(future.globals.maxSize = 100000 * 1024^2)

message("start SCTransform")
data <- SCTransform(use_data, vars.to.regress = "percent.mt", verbose = FALSE)
message("SCTransform is done")

message("start writing dge")
sct_dge<-as_matrix(data@assays$RNA@counts)
fwrite(sct_dge,"sct_dge.csv"))
write.table(rownames(sct_dge),"sct_gene.csv")
write.table(colnames(sct_dge),"sct_cell.csv")
message("writing dge is done")