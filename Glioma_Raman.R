library(tidyverse)
library(purrr)
library(magrittr)
library(ggfortify)
library(e1071)
library(caret)
library(Rtsne)
library(lsmeans)
library(pheatmap)
library(dendextend)
library(tidyverse)
library(psych)
library(readxl)
library(gbm)

# SVM
merge.svm<- read.csv('D:\\', check.names = F)

svm.tune <- train(group ~ ., data = merge.svm, 
                  method = 'svmRadial', 
                  trControl = trainControl(method='cv', number = 5))
svm.tune 

# LDA
merge.lda <- merge.train
lda.mod <- MASS::lda(as.factor(group) ~ ., merge.lda)
data.frame(type = merge.lda$group, lda = lda.mod %>% predict %>% .$x) %>%
  
  ggplot(aes(x=lda.LD1, y=lda.LD2, color=type)) + geom_point()
confusionMatrix(predict(lda.mod, merge.lda)$class%>% as.factor(), merge.lda$group %>% as.factor())
lda.mod

# grediant boost
gbm.mod <- train(as.factor(group) ~., data = merge.train,
                 metric = 'Accuracy',
                 method = 'gbm',
                 trControl = trainControl(method = 'cv', number = 10),
                 tuneGrid = expand.grid(`n.trees` = c(10, 50, 100),
                                        `interaction.depth` = 1:3,
                                        shrinkage = 0.1,
                                        `n.minobsinnode` = 3))
gbm.mod

# RF and importance map
rf.mod <- train(as.factor(group) ~., data = G_L,
                metric = 'Accuracy',
                method = 'rf',
                trControl = trainControl(method = 'cv', number = 10),
                tuneGrid = expand.grid(mtry = c(5, 10, 20)))  
plot(rf.mod) 
varImp(rf.mod, scale = T) %>% plot(top=50)
varimps_rf <- varImp(rf.mod)
write.csv(varimps_rf$importance, 'C:\\Users\\')
pred.rf <- predict(rf.mod, df2, type = 'raw')
table(pred.rf) %>% prop.table()
ggplot(df2, aes(x=X, y=-Y, fill = pred.rf)) + geom_tile(alpha=1) +
  scale_fill_manual(values = c('red','blue','black')) +
  theme_classic()
rf.mod 

# mapping
df2 <- read.delim('D:\\') %>% .[,1:4] %>%
  `names<-`(c('X', 'Y', 'wave', 'intensity')) %>% group_by(X, Y) %>% 
  pivot_wider(names_from = 'wave', values_from = 'intensity')

names(df2)[-(1:2)] <- names(merge)[-(1:3)] 
df2[,-(1:2)]<- df2[,-(1:2)] %>% mutate_if(is.double, ~ . /df2$`1002.551758`)
pred.svm <- predict(svm.tune, df2)
table(pred.svm) %>% prop.table()
ggplot(df2, aes(x=X, y=Y, fill = pred.svm)) + geom_tile(alpha=1) + scale_y_reverse()+
  scale_fill_manual(values = c('#FF3300','blue','#99FFFF','#FF7744','#FF7744')) +theme_classic()+labs(x='', y='')


# import spectral data
dat <- filename %>% 
  map(~read.delim(file.path('D:\\Glioma_file',.), sep = '\t')) %>% 
  lapply(function(t) t[,-(5:7)]) %>% 
  lapply(function(t) t %>% set_colnames(c('x','y', 'wave', 'intensity')))
dat <- lapply(dat, function(t) {
  t %>% group_by(x, y) %>%
    pivot_wider( names_from = wave, values_from = intensity) 
})
dat <- lapply(1:length(filename), function(t){
  dat[[t]] %>% mutate(type= (filename[t]%>% 
                               strsplit(split = '.txt') %>% unlist()), .before='x') %>% 
    ungroup %>% dplyr::select(-x, -y) %>% rownames_to_column(var = 'num') %>% 
    mutate_at(vars(num), as.double)
}) 

#HCA
dat[[1]] %>% 
  column_to_rownames(var='num') %>%
  dplyr::select(-type) %>% 
  dist(method = 'euclidean') %>% hclust(method = 'single')%>% plot # (method = 'single')  %>% color_branches(k=12)%>% set('labels_cex',1)
column_to_rownames(var='num') %>%
  dplyr::select(-type) %>% 
  dist(method = 'euclidean') %>% hclust(method = 'single') %>% plot # (method = 'single')
# Outlier elimination
dat[[6]] <- dat[[6]] %>% filter(!(num %in% c(1,33,57,59,63,79,73,55,60,62,76,77))) 

#PCA and PC loadings
pca.obj %>% 
  autoplot(data=merge.pca, colour='group', frame=T, frame.type='norm')
pc <- pca.obj$rotation

write.csv(pc,'C:\\Users\\dzhou\\Desktop/2_PCs0307.csv', row.names = T)#导出所有的PC1，PC2，PC3,PC4等等

qplot(rownames(pca.obj$rotation) %>% as.numeric(), 
      pca.obj$rotation[,5], color='red',geom = 'line', fill='#F7B6D2')+
  theme_bw() +  theme(panel.grid = element_blank())+ 
  scale_x_continuous(breaks=seq(600,2000,100))+
  scale_y_continuous(breaks=seq(-100,800,0.01))

#3DPCA
plot_ly(comp, x = ~PC1, y = ~PC2, z = ~PC3, color = ~df$group,colors = c('#FF5511', '#CCFF33','#DC71FA','#00FFFF','blue')) %>% #  colors = c('#BF382A', '#0C4B8E')替换color可换颜色
  add_markers(size= 12) %>%
  #  add ellipsoid
  add_trace(x = contour_args$M$vb[1,] + mean(mean_comp$PC1[3]),
            y = contour_args$M$vb[2,] + mean(mean_comp$PC2[3]),
            z = contour_args$M$vb[3,] + mean(mean_comp$PC3[3]),
            type = 'mesh3d', alphahull = 0, opacity = 0.02) %>%
  add_trace(x = contour_args$N$vb[1,] + mean(mean_comp$PC1[4]),
            y = contour_args$N$vb[2,] + mean(mean_comp$PC2[4]),
            z = contour_args$N$vb[3,] + mean(mean_comp$PC3[4]),
            type = 'mesh3d', alphahull = 0, opacity = 0.02) %>%
  add_trace(x = contour_args$G$vb[1,] + mean(mean_comp$PC1[1]),
            y = contour_args$G$vb[2,] + mean(mean_comp$PC2[1]),
            z = contour_args$G$vb[3,] + mean(mean_comp$PC3[1]),
            type = 'mesh3d', alphahull = 0, opacity = 0.02) %>%
  add_trace(x = contour_args$L$vb[1,] + mean(mean_comp$PC1[2]),
            y = contour_args$L$vb[2,] + mean(mean_comp$PC2[2]),
            z = contour_args$L$vb[3,] + mean(mean_comp$PC3[2]),
            type = 'mesh3d', alphahull = 0, opacity = 0.02)%>% 
  add_trace(x = contour_args$T$vb[1,] + mean(mean_comp$PC1[5]),
            y = contour_args$T$vb[2,] + mean(mean_comp$PC2[5]),
            z = contour_args$T$vb[3,] + mean(mean_comp$PC3[5]),
            type = 'mesh3d', alphahull = 0, opacity = 0.02) 

# linear fit
ref_files <- dir('D:\\Glioma_file\\total_data\\Raman_data\\jzStandard1019/', full.names = T)
ref <- map(ref_files, ~read.table(., sep = '\t', check.names = F))
ref <- lapply(1:length(ref), function(t) {
  pivot_wider(data = ref[[t]], names_from = 'V1', values_from = 'V2') 
}) %>% reduce(rbind) %>%  as.data.frame 
ref <- ref %>% sapply(function(t) t/ref$`1002.75293`) %>% 
  as.data.frame %>% dplyr::select(-c(`1002.75293`))
ref <- data.frame(t(ref)) %>% `colnames<-`(sapply(strsplit(ref_files, '/|\\.'),'[[',2))# 
pca_df <- read.csv('D:\\Glioma_file\\total_data\\Raman_data\\zqq0905//PC_G_N0209.csv')
coeff <- lm(PC2 ~ ., data.frame(scale(ref), PC2 = pca_df$PC2))$coefficients[-1]   #自己更换PC1
names(coeff)[1] <- 'actin'
barplot(coeff, las =2, col = ifelse(coeff > 0, '#FF6666', '#66FF99') , ylim = c(-0.03, 0.03), ylab = 'PC1')
box()
par(xpd =F)
abline(0, 0)

# SNR 
dat %>% dplyr::select(`775.407227`,`1764.771484`, `1763.808594`) %>% summarise_all(~mean(.)/sd(.))

dat %>% dplyr::select(`775.407227`,`1764.771484`, `1763.808594`) %>% summarise_all(~mean(.)/sd(.)) %>% pull %>% mean


