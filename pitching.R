# Title: pitchingPredictor
# Author: Jonathan Lieberman
# Data Source: baseballsavant.mlb.com
# Style: lowerCamelCase


# This project is an experimental project to determine whether pitches can be predicted
# based on the status of the game and the way a pitch is thrown. The data from 
# baseballsavant.mlb.com is Cubs pitching data from the 2018 MLB season.


# Packages
library(tidyverse)
library(caret)
library(vtreat)

# Personal file path
basePath <- file.path("C://Users//david//OneDrive//Documents//Learning//MLB")
fileName <- "cubsPitching.csv"

# Read in the file
setwd(basePath)
cubsPitching <- read_csv(fileName)


# Should strings be read as factors?
stringsAsFactors <- FALSE

# Should pitching zones be read as factors?
zonesAsFactors <- TRUE

# Should player IDs be read as factors?
IDsAsFactors <- TRUE

# Should Feature Engineering be done?
addFeatures <- TRUE


# Read in pitching data
pitching <- readPitchingData(fileName = fileName
                             , stringsAsFactors = stringsAsFactors
                             , zonesAsFactors = zonesAsFactors
                             , IDsAsFactors = IDsAsFactors
                             )


# Removing Depreciated columns
pitching <- cleanColumns(pitching)


# Feature Engineering
if (addFeatures) {
  pitching <- simpleFeatures(pitching)
}


# Declare model variables
target <- "pitch_type"
vars <- c("player_name"
          , "release_pos_x"
          , "release_pos_y"
          , "release_pos_z"
          , "inning"
          , "p_throws"
          , "stand"
          , "same_side"
          , "balls"
          , "strikes"
          , "net_score"
          , "man_on_1b"
          , "man_on_2b"
          , "man_on_3b"
          ) 


# Subset data
pitchingSubset <- pitching %>%
  filter(pitch_type != "null") %>% # drop rows with no pitch type
  select(c(target, vars)) %>%
  droplevels() # Not every pitcher throws all pitch types


# Partition data
set.seed(255)
inTrain <- createDataPartition(y = pitchingSubset$pitch_type
                               , p = .75
                               , list = FALSE
)

train <- pitchingSubset[inTrain,]
test <- pitchingSubset[-inTrain,]



# Preprocessing
pOut <- preProcessPitching(train
                           , ignore = target
                           )
trainClean <- pOut$data
plan <- pOut$plan
testClean <- preProcessPitching(test
                                , plan = plan
                                , ignore = target
                                )$data


# Set training controls
ctrl <- trainControl(method = "cv"
                     , number = 3
                     , allowParallel = TRUE
                     , verboseIter = TRUE
                     , search = "random"
                     )


# Create grid for hyperparameter grid/random search
xgbTreeGrid <- expand.grid(nrounds = c(50, 75, 100)
                           , max_depth = c(5, 6)
                           , eta = c(.3, .4)
                           , gamma = c(0, 1)
                           , colsample_bytree = c(.4)
                           , min_child_weight = c(.5, .6)
                           , subsample = c(.75)
                           )


# Create model formula
fmla <- createFormula(target = target
                      , indVariables = colnames(trainClean)
                      , logy = FALSE)


# Train model
model <- train(fmla
               , data = trainClean
               , method = "xgbTree"
               , metric = "Accuracy"
               , preProc = c("nzv", "center", "scale")
               , na.action = na.pass
               , trControl = ctrl
               , tuneGrid = xgbTreeGrid
               , tuneLength = 1
)


# Prepare output
test <- test %>% 
  mutate(pred = predict(model, newdata = testClean))

# Confusion matrix
(confustionMatrix <- table(test$pitch_type, test$pred))

# Model accuracy
(modelAccuracy <- mean(test$pitch_type == test$pred))

# Aggregate pitches by count
trainCounts <- train %>%
  group_by(pitch_type) %>%
  summarize(count = n()) %>%
  mutate(percent = count/sum(count))


# Baseline = guess most frequent pitch
mostFrequentPitch <- trainCounts %>% 
  filter(count == max(count)) %>% 
  select(pitch_type) %>%
  as.character()
  
# Baseline accuracy
(baselineAccuracy <- mean(test$pitch_type == mostFrequentPitch))



# NEXT STEPS

# Find a model which can be understood by a batter
# Switch from caret to parsnip?
# Abstract the model to any team?
# Publish?