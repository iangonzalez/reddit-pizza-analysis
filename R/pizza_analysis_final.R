# Ian Gonzalez
# STAT 365 final project
# analysis of pizza data from kaggle.com
# https://www.kaggle.com/c/random-acts-of-pizza

cur_dir <- getwd()  # should be run from /R directory for paths to work

########### DATA WRANGLING ###########
require("rjson")

fname <- sprintf("%s/../data/train.json", cur_dir)

jsondata <- fromJSON(file=fname)

# remove longer feature vectors from json data
jsondata <- lapply(jsondata, function(lst){
  lst[sapply(lst, is.null)] <- "N/A"  # replace nulls with n/a character vectors
  lst
})
keep <- sapply(jsondata[[1]],length) == 1
prunedlst <- lapply(jsondata, function(lst) lst[keep])
jsonframe <- data.frame(matrix(unlist(prunedlst), nrow=length(prunedlst), byrow=T),
                        stringsAsFactors=F)

# convert numeric data back to numeric
numer_cols <- unname(which(sapply(prunedlst[[1]], is.numeric)))
for (i in numer_cols) {jsonframe[,i] <- as.numeric(jsonframe[,i])}
colnames(jsonframe) <- names(prunedlst[[1]]) # restore names

# now we have a data frame with 31 of the original json features


################################################
############## ADDING FEATURES #################
################################################
# most important feature to analyze is likely request text and request title

# add request length as a feature
req_length <- sapply(jsonframe$request_text, nchar)


## WORKING WITH REQUEST TEXT ########
# extract request text and titles to work with:
reqtext <- jsonframe$request_text

# do some preprocessing (convert to lowercase, remove weird characters)
reqtext <- tolower(reqtext)
reqtext <- gsub("([\\[\\]\\(\\):;@#\"\\$%,_/]|http)", " ", reqtext)

# use these packages for NLP
require("NLP")
require("openNLP")

# if the pos annotations havent been computed, compute them 
# (very time consuming):
if (!file.exists(sprintf("%s/pos_annots_train.rda", cur_dir))) {
  sent_annotator <- Maxent_Sent_Token_Annotator()
  word_annotator <- Maxent_Word_Token_Annotator()
  reqtext <- sapply(reqtext, as.String)
  
  
  # do word and sentence annotation:
  annotations <- lapply(reqtext, function(req) {
    if (req == "") {list()}
    else {annotate(req, list(sent_annotator, word_annotator))}
  })
  
  #do pos annotation:
  pos_annotator <- Maxent_POS_Tag_Annotator()
  pos_annots <- lapply(1:length(reqtext), function(i) {
    if (reqtext[i] == "") {list()}
    else {annotate(reqtext[i], pos_annotator, annotations[[i]])}
  })
  
  # save this so we don't have to do it again:
  save(pos_annots, file=sprintf("%s/pos_annots_train.rda", cur_dir))
} else {
  load(sprintf("%s/pos_annots_train.rda", cur_dir))
}



# get only the nouns from the input text:
getwords <- function(text, annotation) {
  lapply(annotation, function(ann) substr(text, ann$start, ann$end))
}
getpos <- function(words, anns, poslst) {
  posvec <- sapply(anns, function(ann) ann$features[[1]])
  onlypos <- sapply(posvec, function(pos) pos %in% poslst)
  words[onlypos]
}
nouns <- lapply(1:length(reqtext), function(i){
  if (reqtext[i] == "") {character(0)}
  else {    
    wordanns <- subset(pos_annots[[i]], type=="word")
    words <- getwords(reqtext[i], wordanns)
    only_nouns <- as.character(getpos(words, wordanns, c("NN","NNP","NNS")))
  }
})

all_nouns <- unique(unlist(nouns)) # get corpus of all nouns

#####################################################
#### CLUSTERING OF POSTS USING BAG OF WORDS VECTORS###
########################################################

nouns_unrolled <- unlist(nouns)
noun_rank <- sapply(all_nouns, function(nn){sum(nn==nouns_unrolled)})
# get top 100 nouns:
top_nouns <- all_nouns[order(noun_rank, decreasing=TRUE)][1:100]

# generate bag of word vectors by counting appearances of top nouns:
noun_bags <- matrix(0, nrow=length(nouns), ncol=length(top_nouns))
for (i in 1:length(nouns)) {
  noun_bags[i,] <- sapply(top_nouns, function(nn){sum(nouns[[i]] == nn)})
}

# reduce dimension with PCA and plot:
noun_PCA <- prcomp(noun_bags)

noun_PC1 <- noun_PCA$rotation[,1]
noun_PC2 <- noun_PCA$rotation[,2]

noun_dim_reduce <- cbind(noun_PC1, noun_PC2)
noun_reduced <- t(noun_dim_reduce) %*% t(noun_bags)
plot(t(noun_reduced), main="Post clusters by noun similarity")


# validate number of clusters:
require(clValid)

if(FALSE){
  valid_nn_clust<- clValid(t(noun_reduced), 5:10, clMethods = "kmeans", validation="internal")
  optimalScores(valid_nn_clust)
}
# the cluster validation shows k = 5 to have the highest connectivity
# for its clusters (72.8)

####################################
######## TOPIC MODELING ############
####################################
# Do LDA topic modeling to find the topics of the posts:
require("lda")


K <- 5
# remove nouns not in top 100:
restr_nouns <- lapply(nouns, function(nlist){
  nlist[nlist %in% top_nouns]
})
# make feature vectors for each post (simple word count in top 100):
docfeats <- lapply(restr_nouns, function(nlist) {
  sapply(unique(nlist), function(nn) {
    matchidx <- which(top_nouns == nn) 
    if (length(matchidx) > 0) { 
      as.integer(c(matchidx - 1, sum(nlist == nn))) 
    }
  })
})
docfeats <- docfeats[!(sapply(docfeats,length) == 0)]

# train LDA model
topmodel <- lda.collapsed.gibbs.sampler(docfeats, 
                                        K=5, 
                                        num.iterations = 50,
                                        vocab=top_nouns,
                                        alpha=0.1, eta=0.1)


# get most important words in each topic, add features corresponding
# to counts for each word
topwords <- top.topic.words(topmodel$topics, 20, by.score=T)

mywordcnt <- matrix(0, nrow=length(reqtext), ncol=K)
for (i in 1:K) {
  mywordcnt[,i] <- sapply(nouns, function(nns){
    sum(nns %in% topwords[,i])
  })
}

##################################
#### MAKE WORD COUNT MATRIX FROM PAPER FEATURES
##################################

# get paper topic word bags:
source(sprintf("%s/paper_feats.R", cur_dir))
paperwords <- list(top1,top2,top3,top4,top5)

# get all the words from the corpus:
reqwords <- lapply(1:length(reqtext), function(i){
  if (reqtext[i] == "") {character(0)}
  else {    
    wordanns <- subset(pos_annots[[i]], type=="word")
    words <- getwords(reqtext[i], wordanns)
  }
})
# make wordcount matrix, same as above:
wordcnt <- matrix(0, nrow=length(reqtext), ncol=K)
for (i in 1:K) {
  wordcnt[,i] <- sapply(reqwords, function(wrds){
    sum(wrds %in% paperwords[[i]]) / (length(wrds) + 1)
  })
}

# make the full feature frame (only paper feats for now)
featframe <- cbind(jsonframe[,numer_cols], req_length, wordcnt)
featframe$outcome <- factor(jsonframe$requester_received_pizza)

# remove date and time, they dont help:
featframe <- subset(featframe, select=-c(unix_timestamp_of_request,
                                         unix_timestamp_of_request_utc))

###############################
### ADD SPELLING ERROR FEATURES
###############################

# spelling errors in request text normalized by length
require(qdap)
has_errs <- sapply(1:length(jsonframe$request_title), function(i){
  length(which_misspelled(reqtext[i]))/(length(reqwords[[i]]) + 1)
})
featframe$has_errs <- has_errs

# spelling errors in the request title normalized by length
reqtitle <- jsonframe$request_title
title_errs <- sapply(1:length(reqtitle), function(i){
  length(which_misspelled(reqtitle[i]))/(word_count(reqtitle[i])+1)
})
featframe$title_errs <- title_errs


### ADD IMAGE LINK FEATURE ####
# Does the text contain an image file extension or "imgur.com"?
hasimg <- as.numeric(grepl("(imgur\\.com|\\.jpg|\\.png|\\.jpeg)", reqtext))
featframe$hasimg <- hasimg


##### DO FEATURE NORMALIZATION #########
normalize <- function(data) {(data - mean(data))/sqrt(var(data))}

# create featframe with my word features for comparison purposes:
myfeatframe <- featframe
myfeatframe[,c("1","2", "3", "4","5")] <- mywordcnt

# normalize both:
for (i in 1:length(featframe)) {
  if (names(featframe)[i] != "outcome") {
    myfeatframe[,i] <- normalize(myfeatframe[,i])
    featframe[,i] <- normalize(featframe[,i])
  }
}


#### SEPARATE TRAINING AND TEST SETS ####
set.seed(17)
total_pts <- length(featframe[,1])
train_size <- round(total_pts * 0.75, 0)
train_idxs <- sample(1:total_pts, size=train_size)
trainframe <- featframe[train_idxs,]
testframe <- featframe[-train_idxs,]

## Train boosted model:
require(adabag)
pizza.bt <- boosting(outcome ~ ., data=trainframe)
btpreds <- predict(pizza.bt, trainframe) 
sum(btpreds$class == trainframe$outcome)
# 3879, best so far

testpreds <- predict(pizza.bt, testframe)
sum(testpreds$class == testframe$outcome)

## Make ROC curve for boosted model:
require(ROCR)
btpredsobj <- prediction((testpreds$prob[,2]), testframe$outcome)
ROCobj <- performance(btpredsobj, measure="rec", x.measure="prec")
plot(ROCobj, main="ROC curve for boosted tree, paper features")
AUCobj <- performance(btpredsobj, measure="auc")
AUCobj@y.values[[1]]

# Train logistic model on the data, test effectiveness
pizza.glm4 <- glm(outcome ~ ., data=trainframe, family="binomial")
preds4 <- (predict(pizza.glm4, trainframe) > 0)
sum(preds4 == trainframe$outcome)
preds4 <- (predict(pizza.glm4, testframe) > 0)
sum(preds4 == testframe$outcome)
summary(pizza.glm4)


######## REPEAT PROCESS FOR MY FEATS:
mytrainframe <- myfeatframe[train_idxs,]
mytestframe <- myfeatframe[-train_idxs,]

# train boosted model 2
pizza.bt2 <- boosting(outcome ~ ., data=mytrainframe)
btpreds2 <- predict(pizza.bt2, mytrainframe) 
sum(btpreds2$class == mytrainframe$outcome)

# test boosted model 2
testpreds2 <- predict(pizza.bt2, mytestframe)
sum(testpreds2$class == mytestframe$outcome)

# ROC curve for boosted model 2
btpredsobj2 <- prediction((testpreds2$prob[,2]), mytestframe$outcome)
ROCobj2 <- performance(btpredsobj2, measure="rec", x.measure="prec")
plot(ROCobj2, main="ROC curve for boosted tree, my features")
AUCobj2 <- performance(btpredsobj2, measure="auc")
AUCobj2@y.values[[1]]




# Train logistic model on the my dataset, test it
pizza.glm5 <- glm(outcome ~ ., data=mytrainframe, family="binomial")
preds5 <- (predict(pizza.glm5, mytrainframe) > 0)
sum(preds5 == mytrainframe$outcome)
preds5 <- (predict(pizza.glm5, mytestframe) > 0)
sum(preds5 == mytestframe$outcome)
summary(pizza.glm5)






##################################
# PCA VISUALIZATION #############
##################################

# do PCA on paper feats frame:
featmat <- as.matrix(subset(featframe, select=-outcome))
train_pca <- prcomp(featmat)
PC1 <- train_pca$rotation[,1]
PC2 <- train_pca$rotation[,2]

# reduce dim to 2:
dim_reduce <- cbind(PC1, PC2)
data_reduced <- t(dim_reduce) %*% t(featmat)

# plot with correct colors
outcols <- sapply(as.logical(featframe$outcome), 
                  function(b) if(b){"green"}else{"red"})
plot(t(data_reduced), col=outcols, main="PCA plotted data, paper features")



# same code as above, but for generating plot for dataset 2.
featmat2 <- as.matrix(subset(myfeatframe, select=-outcome))
train_pca <- prcomp(featmat2)
PC1 <- train_pca$rotation[,1]
PC2 <- train_pca$rotation[,2]

dim_reduce <- cbind(PC1, PC2)
data_reduced <- t(dim_reduce) %*% t(featmat2)
outcols <- sapply(as.logical(myfeatframe$outcome), 
                  function(b) if(b){"green"}else{"red"})
plot(t(data_reduced), col=outcols, main="PCA plotted data, my features")

